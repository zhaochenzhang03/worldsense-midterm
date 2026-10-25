"""
Re-derive parsed_prediction, parse_status and correct for saved
evaluation records from their preserved raw_response, using the
current src/answer_parser.py. Offline: no API calls. raw_response and
every other field are left untouched.

Use after a parser change so stored parses match the parser that
scripts/validate_evaluation.py checks against. Files are rewritten
atomically (temp file + rename).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from scripts.run_evaluation import EVAL_CONFIG, SPLIT_FILES, load_jsonl
from src.answer_parser import parse_answer
from src.prompting import CONDITIONS, answer_letters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--split", choices=list(SPLIT_FILES), default="dev")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the files (default: only report changes).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open(EVAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    split_dir = Path(config["output_dir"]) / args.split
    changed_total = 0

    for condition in CONDITIONS:
        path = split_dir / f"{condition}.jsonl"
        records = load_jsonl(path)
        changed = 0

        for record in records:
            if record.get("raw_response") is None:
                continue

            parsed = parse_answer(
                record["raw_response"], answer_letters(record["candidates"])
            )
            correct = parsed.prediction == record["gold"]

            if (
                record["parsed_prediction"], record["parse_status"], record["correct"]
            ) != (parsed.prediction, parsed.status, correct):
                print(
                    f"{condition} {record['uid']}: "
                    f"{record['parsed_prediction']} ({record['parse_status']}) -> "
                    f"{parsed.prediction} ({parsed.status})"
                )
                record["parsed_prediction"] = parsed.prediction
                record["parse_status"] = parsed.status
                record["correct"] = correct
                changed += 1

        if changed and args.write:
            tmp = path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        changed_total += changed

    action = "rewritten" if args.write else "would change (use --write)"
    print(f"{changed_total} record(s) {action}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
