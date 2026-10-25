import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.create_split import (
    DEV_OUTPUT,
    DEV_SIZE,
    JSON_PATH,
    MAX_DURATION_SECONDS,
    TEST_OUTPUT,
    TEST_SIZE,
    TOTAL_SIZE,
    parse_duration_seconds,
)


MANIFEST_PATH = Path("configs/experiment_manifest.json")

EXPECTED_NUM_TASK_TYPES = 26


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    failures = []

    def check(condition: bool, message: str) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {message}")

        if not condition:
            failures.append(message)

    with open(DEV_OUTPUT, "r", encoding="utf-8") as f:
        dev = json.load(f)

    with open(TEST_OUTPUT, "r", encoding="utf-8") as f:
        test = json.load(f)

    full = dev + test

    # ---------------------------------------------------------
    # 1. Protocol invariants
    # ---------------------------------------------------------
    print("=" * 70)
    print("Protocol invariants")
    print("=" * 70)

    check(
        len(dev) == DEV_SIZE,
        f"dev has exactly {DEV_SIZE} examples (found {len(dev)})",
    )
    check(
        len(test) == TEST_SIZE,
        f"test has exactly {TEST_SIZE} examples (found {len(test)})",
    )

    video_ids = [x["video_id"] for x in full]
    check(
        len(set(video_ids)) == TOTAL_SIZE,
        f"{TOTAL_SIZE} unique video IDs "
        f"(found {len(set(video_ids))})",
    )

    overlap = (
        {x["video_id"] for x in dev}
        & {x["video_id"] for x in test}
    )
    check(
        len(overlap) == 0,
        f"dev/test video IDs do not overlap "
        f"(overlap: {sorted(overlap)})",
    )

    durations = {
        x["uid"]: parse_duration_seconds(x["video_duration"])
        for x in full
    }
    bad_durations = sorted(
        uid
        for uid, duration in durations.items()
        if duration is None or duration > MAX_DURATION_SECONDS
    )
    check(
        len(bad_durations) == 0,
        f"every video duration <= {MAX_DURATION_SECONDS}s "
        f"(violations: {bad_durations})",
    )

    task_types = {x["task_type"] for x in full}
    check(
        len(task_types) == EXPECTED_NUM_TASK_TYPES,
        f"all {EXPECTED_NUM_TASK_TYPES} task types represented "
        f"(found {len(task_types)})",
    )

    uid_counts = Counter(x["uid"] for x in full)
    duplicate_uids = sorted(
        uid
        for uid, count in uid_counts.items()
        if count > 1
    )
    check(
        len(duplicate_uids) == 0,
        f"no duplicate UIDs (duplicates: {duplicate_uids})",
    )

    # ---------------------------------------------------------
    # 2. Manifest hashes
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("Manifest hashes")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        check(False, f"{MANIFEST_PATH} exists")
    else:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        hashed_files = [
            ("dev_split_sha256", DEV_OUTPUT),
            ("test_split_sha256", TEST_OUTPUT),
            ("dataset_sha256", JSON_PATH),
        ]

        for key, path in hashed_files:
            if not path.exists():
                check(False, f"{path} exists (needed for {key})")
                continue

            actual = sha256_of(path)
            check(
                actual == manifest.get(key),
                f"{path} SHA-256 matches manifest ({actual})",
            )

    # ---------------------------------------------------------
    # 3. Summary statistics
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary statistics")
    print("=" * 70)

    known_durations = [d for d in durations.values() if d is not None]
    if known_durations:
        print(
            f"Video duration (s): "
            f"min={min(known_durations)} "
            f"max={max(known_durations)} "
            f"mean={sum(known_durations) / len(known_durations):.1f}"
        )

    for name, split in (("dev", dev), ("test", test)):
        print()
        print(f"[{name}] n={len(split)}")
        print(
            "  task domains: ",
            dict(Counter(x["task_domain"] for x in split)),
        )
        print(
            "  answer letters:",
            dict(sorted(Counter(x["answer"] for x in split).items())),
        )
        print(
            "  task types covered:",
            len({x["task_type"] for x in split}),
        )

    print()
    print("Task type counts in full 60:")
    for task_type, count in sorted(
        Counter(x["task_type"] for x in full).items()
    ):
        print(f"  {task_type}: {count}")

    print()
    print("Domains in full 60:")
    for domain, count in sorted(
        Counter(x["domain"] for x in full).items()
    ):
        print(f"  {domain}: {count}")

    print()
    print("Audio classes in full 60:")
    audio_counts = Counter(
        audio_class
        for x in full
        for audio_class in x["audio_class"]
    )
    for audio_class, count in sorted(audio_counts.items()):
        print(f"  {audio_class}: {count}")

    no_speech = sum("Speech" not in x["audio_class"] for x in full)
    print(f"Examples without 'Speech' audio class: {no_speech}")

    # ---------------------------------------------------------
    # 4. Result
    # ---------------------------------------------------------
    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed.")
        return 1

    print("VALIDATION PASSED: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
