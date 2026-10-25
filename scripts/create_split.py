import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.worldsense_data import load_worldsense


JSON_PATH = Path("data/worldsense/worldsense_qa.json")

DEV_OUTPUT = Path("configs/dev_ids.json")
TEST_OUTPUT = Path("configs/test_ids.json")

SEED = 42
TOTAL_SIZE = 60
DEV_SIZE = 12
TEST_SIZE = 48
MAX_DURATION_SECONDS = 120


def parse_duration_seconds(duration: str) -> int | None:
    """
    Convert strings like:
        "60s"
        "105s"
        "120s"
    into integer seconds.

    Returns None if parsing fails.
    """
    if not duration:
        return None

    match = re.search(r"(\d+)", duration)

    if match is None:
        return None

    return int(match.group(1))


def check_overwrite(force: bool) -> None:
    """
    The split in configs/ is the canonical frozen split.
    Refuse to overwrite it unless --force is given.
    """
    existing = [
        path
        for path in (DEV_OUTPUT, TEST_OUTPUT)
        if path.exists()
    ]

    if not existing:
        return

    if not force:
        print(
            "ERROR: frozen split files already exist:\n"
            + "".join(f"  {path}\n" for path in existing)
            + "Refusing to overwrite them. The existing split is the "
            "canonical frozen split for this project.\n"
            "Re-run with --force only if you have explicit permission "
            "to regenerate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 70, file=sys.stderr)
    print(
        "WARNING: --force given. Overwriting the frozen split files:\n"
        + "".join(f"  {path}\n" for path in existing)
        + "configs/experiment_manifest.json hashes will no longer match "
        "unless the regenerated files are byte-identical.",
        file=sys.stderr,
    )
    print("=" * 70, file=sys.stderr)


def make_split():
    random.seed(SEED)

    # ---------------------------------------------------------
    # 1. Load all WorldSense QA examples
    # ---------------------------------------------------------
    examples = load_worldsense(JSON_PATH)

    print(f"Loaded {len(examples)} QA examples.")

    # ---------------------------------------------------------
    # 2. Keep only videos <= 120 seconds
    # ---------------------------------------------------------
    eligible = []

    for example in examples:

        duration = parse_duration_seconds(
            example.video_duration
        )

        if duration is None:
            continue

        if duration <= MAX_DURATION_SECONDS:
            eligible.append(example)

    print(
        f"Eligible QA examples "
        f"(video <= {MAX_DURATION_SECONDS}s): "
        f"{len(eligible)}"
    )

    # ---------------------------------------------------------
    # 3. Group examples by task type
    # ---------------------------------------------------------
    by_task_type = defaultdict(list)

    for example in eligible:
        by_task_type[example.task_type].append(example)

    task_types = sorted(by_task_type.keys())

    print(f"Number of task types available: {len(task_types)}")

    # Shuffle within every task type reproducibly.
    for task_type in task_types:
        random.shuffle(by_task_type[task_type])

    # Shuffle task-type order as well.
    random.shuffle(task_types)

    # ---------------------------------------------------------
    # 4. Select 60 QA examples.
    #
    # Requirements:
    #   - cover task types as evenly as possible
    #   - only one QA per video
    # ---------------------------------------------------------
    selected = []
    used_videos = set()

    # Keep moving round-robin over task types until
    # we have 60 examples.
    pointers = {
        task_type: 0
        for task_type in task_types
    }

    while len(selected) < TOTAL_SIZE:

        added_this_round = False

        for task_type in task_types:

            candidates = by_task_type[task_type]

            while pointers[task_type] < len(candidates):

                example = candidates[pointers[task_type]]
                pointers[task_type] += 1

                # One QA per video.
                if example.video_id in used_videos:
                    continue

                selected.append(example)
                used_videos.add(example.video_id)

                added_this_round = True

                break

            if len(selected) >= TOTAL_SIZE:
                break

        if not added_this_round:
            raise RuntimeError(
                "Could not find enough unique-video examples "
                "to create the requested split."
            )

    assert len(selected) == TOTAL_SIZE
    assert len(used_videos) == TOTAL_SIZE

    # ---------------------------------------------------------
    # 5. Split 60 -> 12 dev + 48 test
    #
    # Because selected examples are already round-robin over
    # task types, taking every 5th item gives us 12 reasonably
    # distributed dev examples.
    # ---------------------------------------------------------
    dev = []
    test = []

    for idx, example in enumerate(selected):

        if idx % 5 == 0 and len(dev) < DEV_SIZE:
            dev.append(example)
        else:
            test.append(example)

    # Safety fallback in case sizes somehow differ.
    while len(dev) < DEV_SIZE:
        dev.append(test.pop())

    while len(dev) > DEV_SIZE:
        test.append(dev.pop())

    assert len(dev) == DEV_SIZE
    assert len(test) == TEST_SIZE

    # ---------------------------------------------------------
    # 6. Make sure dev/test videos never overlap
    # ---------------------------------------------------------
    dev_videos = {
        example.video_id
        for example in dev
    }

    test_videos = {
        example.video_id
        for example in test
    }

    overlap = dev_videos & test_videos

    assert len(overlap) == 0, (
        f"Video leakage found: {overlap}"
    )

    # ---------------------------------------------------------
    # 7. Save enough metadata so we know exactly which
    #    examples belong to each split.
    # ---------------------------------------------------------
    def serialize(example):
        return {
            "uid": example.uid,
            "video_id": example.video_id,
            "task_id": example.task_id,
            "video_duration": example.video_duration,
            "domain": example.domain,
            "sub_category": example.sub_category,
            "audio_class": example.audio_class,
            "task_domain": example.task_domain,
            "task_type": example.task_type,
            "question": example.question,
            "candidates": example.candidates,
            "answer": example.answer,
        }

    DEV_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DEV_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [serialize(x) for x in dev],
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        TEST_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [serialize(x) for x in test],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # 8. Print statistics
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("Split created successfully")
    print("=" * 70)

    print(f"Dev examples:  {len(dev)}")
    print(f"Test examples: {len(test)}")

    print()
    print("Dev task domains:")
    print(
        Counter(
            x.task_domain
            for x in dev
        )
    )

    print()
    print("Test task domains:")
    print(
        Counter(
            x.task_domain
            for x in test
        )
    )

    print()
    print(
        "Task types represented in full 60:",
        len(
            set(
                x.task_type
                for x in selected
            )
        ),
    )

    print()
    print("Task type counts in full 60:")

    task_counts = Counter(
        x.task_type
        for x in selected
    )

    for task_type, count in sorted(
        task_counts.items()
    ):
        print(
            f"  {task_type}: {count}"
        )

    print()
    print("Domains in full 60:")

    domain_counts = Counter(
        x.domain
        for x in selected
    )

    for domain, count in sorted(
        domain_counts.items()
    ):
        print(
            f"  {domain}: {count}"
        )

    print()
    print("Audio classes in full 60:")

    audio_counts = Counter()

    for example in selected:
        for audio_class in example.audio_class:
            audio_counts[audio_class] += 1

    for audio_class, count in sorted(
        audio_counts.items()
    ):
        print(
            f"  {audio_class}: {count}"
        )

    print()
    print(f"Saved dev split to:  {DEV_OUTPUT}")
    print(f"Saved test split to: {TEST_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create the WorldSense dev/test split. "
            "Refuses to overwrite existing split files "
            "unless --force is given."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing frozen split files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_overwrite(args.force)
    make_split()