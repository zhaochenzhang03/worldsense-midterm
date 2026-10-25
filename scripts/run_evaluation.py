import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.create_split import DEV_OUTPUT, JSON_PATH, TEST_OUTPUT
from scripts.download_frozen_videos import (
    MEDIA_MANIFEST,
    file_crc32_and_sha256,
)
from scripts.create_modality_variants import VARIANT_MANIFEST
from src.model_client import GatewayNotConfigured, make_client
from src.prompting import CONDITIONS, PROMPT_VERSION, build_prompt
from src.worldsense_data import load_worldsense


EVAL_CONFIG = Path("configs/eval_config.json")
SPLIT_FILES = {"dev": DEV_OUTPUT, "test": TEST_OUTPUT}
FORMAL_TEST_MARKER = "_formal_run.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen WorldSense evaluation (dev by default)."
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required together with --split test. Test must never be "
             "used for prompt development.",
    )
    conditions = parser.add_mutually_exclusive_group(required=True)
    conditions.add_argument(
        "--condition",
        choices=CONDITIONS,
        action="append",
        help="Condition to run (repeatable).",
    )
    conditions.add_argument("--all-conditions", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only the first N frozen examples of the split (frozen order).",
    )
    parser.add_argument(
        "--uids",
        nargs="+",
        help="Only these frozen UIDs (must belong to --split).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip (uid, condition, model, prompt_version) pairs that "
             "already have a successful result (default: from config).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt pairs whose saved records are all errors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print requests without calling the API or "
             "writing outputs.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line from an interrupted write is
                # ignored; that call is simply redone.
                print(
                    f"WARNING: {path}:{line_number} is not valid JSON; ignored",
                    file=sys.stderr,
                )
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """
    Append one record and force it to disk before returning, so a
    completed call survives an interruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def is_success(record: dict) -> bool:
    return record.get("error") is None and record.get("raw_response") is not None


def result_key(record: dict) -> tuple:
    return (
        record["uid"],
        record["condition"],
        record["model"],
        record["prompt_version"],
    )


def expected_media_hashes() -> dict[str, str]:
    """
    {media path: SHA-256} for every frozen original and variant.
    """
    hashes = {}

    with open(MEDIA_MANIFEST, "r", encoding="utf-8") as f:
        for v in json.load(f)["videos"]:
            hashes[v["local_path"]] = v["sha256"]

    with open(VARIANT_MANIFEST, "r", encoding="utf-8") as f:
        for v in json.load(f)["videos"]:
            hashes[v["video_only_path"]] = v["video_only_sha256"]
            hashes[v["audio_only_path"]] = v["audio_only_sha256"]

    return hashes


def main() -> int:
    args = parse_args()

    # ---------------------------------------------------------
    # 0. Test-split guard: before anything else, and before any
    #    API client exists.
    # ---------------------------------------------------------
    if args.split == "test" and not args.allow_test:
        print(
            "ERROR: --split test requires --allow-test. The test split "
            "is reserved for the final formal run and must never be "
            "used for prompt development.",
            file=sys.stderr,
        )
        return 2

    with open(EVAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config["prompt_version"] != PROMPT_VERSION:
        print(
            f"ERROR: config prompt_version {config['prompt_version']} != "
            f"src/prompting.py PROMPT_VERSION {PROMPT_VERSION}",
            file=sys.stderr,
        )
        return 1

    resume = config["resume"] if args.resume is None else args.resume
    conditions = CONDITIONS if args.all_conditions else args.condition
    model = config["model"]["model_id"]
    temperature = config["generation"]["temperature"]
    output_dir = Path(config["output_dir"]) / args.split

    # ---------------------------------------------------------
    # 1. Frozen examples only, in frozen order.
    # ---------------------------------------------------------
    with open(SPLIT_FILES[args.split], "r", encoding="utf-8") as f:
        examples = json.load(f)

    if args.uids:
        split_uids = {x["uid"] for x in examples}
        unknown = [uid for uid in args.uids if uid not in split_uids]
        if unknown:
            print(
                f"ERROR: not frozen {args.split} UIDs: {unknown}",
                file=sys.stderr,
            )
            return 1
        examples = [x for x in examples if x["uid"] in set(args.uids)]

    if args.limit is not None:
        examples = examples[:args.limit]

    # video_caption is never used; load it only to prove absence.
    captions = {
        x.video_id: x.video_caption
        for x in load_worldsense(JSON_PATH)
    }

    media_hashes = expected_media_hashes()
    verified_media = {}

    # ---------------------------------------------------------
    # 2. Plan: build every prompt and check it before any call.
    # ---------------------------------------------------------
    plan = []
    problems = []

    for condition in conditions:
        for example in examples:
            prompt = build_prompt(example, condition)
            caption = captions.get(example["video_id"], "")

            if "video_caption" in prompt.text or (caption and caption in prompt.text):
                problems.append(f"{example['uid']}/{condition}: video_caption in prompt")

            for media in prompt.media:
                if media.path not in verified_media:
                    path = Path(media.path)
                    verified_media[media.path] = (
                        path.exists()
                        and file_crc32_and_sha256(path)[1] == media_hashes.get(media.path)
                    )
                if not verified_media[media.path]:
                    problems.append(
                        f"{example['uid']}/{condition}: media {media.path} "
                        "missing or hash does not match manifest"
                    )

            plan.append((condition, example, prompt))

    if problems:
        print("ERROR: request checks failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    # ---------------------------------------------------------
    # 3. Dry run: show what would be sent, then stop.
    # ---------------------------------------------------------
    if args.dry_run:
        for condition, example, prompt in plan:
            print("=" * 70)
            print(f"[DRY RUN] {args.split} {example['uid']} condition={condition}")
            print("=" * 70)
            print(f"model={model} temperature={temperature} prompt_version={PROMPT_VERSION}")
            print(f"subtitle_used={prompt.subtitle_used}")
            if prompt.media:
                for media in prompt.media:
                    size = Path(media.path).stat().st_size / 1e6
                    print(f"media: {media.path} ({media.mime_type}, {size:.1f} MB, hash verified)")
            else:
                print("media: none")
            print("checks: video_caption absent, frozen UID, media hashes OK")
            print("--- prompt text ---")
            print(prompt.text)
            print()

        print(f"[DRY RUN] {len(plan)} request(s) planned; no API calls made, no files written.")
        return 0

    # ---------------------------------------------------------
    # 4. Real run.
    # ---------------------------------------------------------
    client = make_client(config)
    try:
        client.check_ready()
    except GatewayNotConfigured as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.split == "test":
        marker = output_dir / FORMAL_TEST_MARKER
        if not marker.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "formal_test_run": True,
                "started": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "prompt_version": PROMPT_VERSION,
            }, indent=2) + "\n")

    existing = {}
    for condition in conditions:
        for record in load_jsonl(output_dir / f"{condition}.jsonl"):
            existing.setdefault(result_key(record), []).append(record)

    called = skipped = failed = 0

    for index, (condition, example, prompt) in enumerate(plan, start=1):
        key = (example["uid"], condition, model, PROMPT_VERSION)
        prior = existing.get(key, [])
        prefix = f"[{index}/{len(plan)}] {example['uid']} {condition}"

        if any(is_success(r) for r in prior):
            if not resume:
                print(
                    f"ERROR: {prefix} already has a successful result and "
                    "--no-resume was given; refusing to duplicate it.",
                    file=sys.stderr,
                )
                return 1
            print(f"{prefix}: skipped (already completed)")
            skipped += 1
            continue

        if prior and not args.retry_failed:
            print(f"{prefix}: skipped (previous attempt failed; use --retry-failed)")
            skipped += 1
            continue

        media_paths = [m.path for m in prompt.media]
        response = client.generate(example, condition, prompt, media_paths)

        record = {
            "uid": example["uid"],
            "video_id": example["video_id"],
            "split": args.split,
            "condition": condition,
            "question": example["question"],
            "candidates": example["candidates"],
            "gold": example["answer"],
            "raw_response": response.raw_response,
            "parsed_prediction": response.parsed_prediction,
            "parse_status": response.parse_status,
            "correct": (
                response.parsed_prediction == example["answer"]
                if response.ok else None
            ),
            "model": model,
            "provider": response.provider,
            "gateway_model_id": config["model"].get("gateway_model_id"),
            "prompt_version": PROMPT_VERSION,
            "prompt_text": prompt.text,
            "prompt_sha256": prompt.text_sha256,
            "temperature": temperature,
            "media": [
                {"path": m.path, "mime_type": m.mime_type, "sha256": media_hashes[m.path]}
                for m in prompt.media
            ],
            "subtitle_used": prompt.subtitle_used,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_seconds": response.latency_seconds,
            "error": response.error,
            "request_metadata": response.request_metadata,
        }
        append_jsonl(output_dir / f"{condition}.jsonl", record)
        existing.setdefault(key, []).append(record)
        called += 1

        if response.ok:
            print(
                f"{prefix}: {response.raw_response!r} -> "
                f"{response.parsed_prediction} ({response.parse_status})"
            )
        else:
            failed += 1
            print(f"{prefix}: ERROR {response.error}")

    print()
    print(f"Calls: {called}  Skipped: {skipped}  Errors: {failed}")
    print(f"Outputs: {output_dir}/<condition>.jsonl")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
