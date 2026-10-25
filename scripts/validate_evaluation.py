import argparse
import json
import os
import sys
from pathlib import Path

from scripts.create_split import JSON_PATH
from scripts.run_evaluation import (
    EVAL_CONFIG,
    FORMAL_TEST_MARKER,
    SPLIT_FILES,
    is_success,
    load_jsonl,
    result_key,
)
from src.answer_parser import AMBIGUOUS, EXACT, INVALID, NORMALIZED, parse_answer
from src.prompting import CONDITIONS, PROMPT_VERSION, build_prompt
from src.worldsense_data import load_worldsense


# Offline analysis outputs (scripts/analyze_results.py), not records.
ANALYSIS_DIR = "analysis"

REQUIRED_FIELDS = [
    "uid", "video_id", "split", "condition", "question", "candidates",
    "gold", "raw_response", "parsed_prediction", "parse_status", "correct",
    "model", "prompt_version", "prompt_text", "prompt_sha256", "temperature",
    "timestamp", "media", "subtitle_used", "error", "latency_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate saved evaluation outputs (offline)."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        help="Outputs root (default: output_dir from configs/eval_config.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open(EVAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    outputs_dir = args.outputs_dir or Path(config["output_dir"])

    frozen = {}
    for split, path in SPLIT_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            for example in json.load(f):
                frozen[example["uid"]] = (split, example)

    captions = {
        x.video_id: x.video_caption
        for x in load_worldsense(JSON_PATH)
    }
    secret = os.environ.get(config["gateway"]["api_key_env_var"])

    problems = []
    counts = {}

    if not outputs_dir.exists():
        print(f"No outputs at {outputs_dir}; nothing to validate.")
        return 0

    for split_dir in sorted(p for p in outputs_dir.iterdir()):
        if split_dir.name == ANALYSIS_DIR and split_dir.is_dir():
            continue
        if not split_dir.is_dir() or split_dir.name not in SPLIT_FILES:
            problems.append(f"unexpected entry in outputs: {split_dir.name}")
            continue

        split = split_dir.name

        if split == "test" and not (split_dir / FORMAL_TEST_MARKER).exists():
            problems.append(
                "test outputs exist without a formal-run marker "
                f"({FORMAL_TEST_MARKER}); test must only be run formally "
                "with --allow-test"
            )

        for path in sorted(split_dir.iterdir()):
            if path.name == FORMAL_TEST_MARKER and split == "test":
                continue
            if path.suffix != ".jsonl" or path.stem not in CONDITIONS:
                problems.append(f"unexpected output file: {path}")
                continue

            condition = path.stem
            success_keys = set()

            for n, record in enumerate(load_jsonl(path), start=1):
                where = f"{path}:{n}"
                counts[(split, condition)] = counts.get((split, condition), 0) + 1

                missing = [k for k in REQUIRED_FIELDS if k not in record]
                if missing:
                    problems.append(f"{where}: missing fields {missing}")
                    continue

                uid = record["uid"]
                if uid not in frozen:
                    problems.append(f"{where}: {uid} is not a frozen UID")
                    continue

                frozen_split, example = frozen[uid]
                if record["split"] != split or frozen_split != split:
                    problems.append(
                        f"{where}: {uid} belongs to {frozen_split}, record "
                        f"says {record['split']}, stored under {split}"
                    )
                if record["condition"] != condition:
                    problems.append(f"{where}: condition {record['condition']} in {path.name}")
                if record["video_id"] != example["video_id"]:
                    problems.append(f"{where}: video_id does not match frozen example")
                if record["gold"] != example["answer"]:
                    problems.append(f"{where}: gold does not match frozen answer")

                if is_success(record):
                    key = result_key(record)
                    if key in success_keys:
                        problems.append(f"{where}: duplicate successful result for {key}")
                    success_keys.add(key)

                # Predictions are a valid letter or explicitly unparsed.
                letters = [c[0] for c in example["candidates"]]
                status = record["parse_status"]
                prediction = record["parsed_prediction"]
                if status in (EXACT, NORMALIZED):
                    if prediction not in letters:
                        problems.append(f"{where}: {status} prediction {prediction!r} not in {letters}")
                elif status in (AMBIGUOUS, INVALID):
                    if prediction is not None:
                        problems.append(f"{where}: {status} record has prediction {prediction!r}")
                else:
                    problems.append(f"{where}: unknown parse_status {status!r}")

                if record["raw_response"] is not None:
                    current = parse_answer(record["raw_response"], letters)
                    if (current.prediction, current.status) != (prediction, status):
                        problems.append(
                            f"{where}: stored parse {prediction} ({status}) differs "
                            f"from current parser {current.prediction} ({current.status}); "
                            "run python -m scripts.reparse_outputs --write"
                        )

                if is_success(record) and record["correct"] != (prediction == record["gold"]):
                    problems.append(f"{where}: 'correct' inconsistent with prediction/gold")

                # The saved prompt must be the frozen prompt.
                if record["prompt_version"] == PROMPT_VERSION:
                    expected = build_prompt(example, condition)
                    if record["prompt_text"] != expected.text:
                        problems.append(f"{where}: prompt_text differs from frozen {PROMPT_VERSION} prompt")
                    if [m["path"] for m in record["media"]] != [m.path for m in expected.media]:
                        problems.append(f"{where}: media differs from frozen condition inputs")

                serialized = json.dumps(record, ensure_ascii=False)
                caption = captions.get(record["video_id"], "")
                if "video_caption" in serialized or (caption and caption in serialized):
                    problems.append(f"{where}: video_caption appears in record")
                if secret and secret in serialized:
                    problems.append(f"{where}: API key appears in record")

    for (split, condition), n in sorted(counts.items()):
        print(f"  {split}/{condition}: {n} record(s)")

    print()
    if problems:
        print(f"EVALUATION VALIDATION FAILED: {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("EVALUATION VALIDATION PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
