"""
Record or check the frozen final TEST protocol.

--write records SHA-256 hashes of the files that define the test
evaluation (config, split, prompt, parser, transport, runner) plus the
scientific settings in configs/final_test_protocol.json. Run it once,
before the first TEST request. Without --write, recompute the hashes
and fail if any file changed since the freeze. Contains no secrets.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_evaluation import EVAL_CONFIG, SPLIT_FILES
from src.prompting import CONDITIONS, PROMPT_VERSION


PROTOCOL_FILE = Path("configs/final_test_protocol.json")

HASHED_FILES = [
    "configs/eval_config.json",
    "configs/test_ids.json",
    "configs/dev_ids.json",
    "configs/media_manifest.json",
    "configs/variant_manifest.json",
    "src/prompting.py",
    "src/answer_parser.py",
    "src/model_client.py",
    "src/subtitle_utils.py",
    "scripts/run_evaluation.py",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Write {PROTOCOL_FILE} (refuses to overwrite an existing freeze).",
    )
    return parser.parse_args()


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_hashes() -> dict[str, str]:
    return {path: sha256_file(path) for path in HASHED_FILES}


def check() -> list[str]:
    """Files whose hash differs from the recorded freeze."""
    with open(PROTOCOL_FILE, "r", encoding="utf-8") as f:
        frozen = json.load(f)["sha256"]

    now = current_hashes()
    return [
        f"{path}: frozen {frozen[path][:12]}…, now {now.get(path, 'missing')[:12]}…"
        for path in frozen
        if now.get(path) != frozen[path]
    ]


def main() -> int:
    args = parse_args()

    if not args.write:
        if not PROTOCOL_FILE.exists():
            print(f"No {PROTOCOL_FILE}; nothing to check.")
            return 0
        changed = check()
        if changed:
            print("PROTOCOL HASH CHECK FAILED: files changed since the freeze:")
            for line in changed:
                print(f"  {line}")
            return 1
        print(f"PROTOCOL HASH CHECK PASSED ({len(HASHED_FILES)} files unchanged).")
        return 0

    if PROTOCOL_FILE.exists():
        print(f"ERROR: {PROTOCOL_FILE} already exists; the freeze is final.", file=sys.stderr)
        return 1

    with open(EVAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(SPLIT_FILES["test"], "r", encoding="utf-8") as f:
        test_size = len(json.load(f))

    protocol = {
        "description": "Frozen final TEST protocol, recorded before the first TEST request. "
                       "Check with: python -m scripts.freeze_test_protocol",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "frozen_before_first_test_request": True,
        "model": config["model"]["model_id"],
        "gateway_model_id": config["model"]["gateway_model_id"],
        "gateway_base_url": config["gateway"]["base_url"],
        "chat_completions_path": config["gateway"]["chat_completions_path"],
        "temperature": config["generation"]["temperature"],
        "max_tokens": config["generation"]["max_tokens"],
        "prompt_version": PROMPT_VERSION,
        "conditions": CONDITIONS,
        "parser": "src/answer_parser.py as hashed below; unparsed (ambiguous/invalid) answers count as incorrect",
        "test_split_size": test_size,
        "planned_pairs": test_size * len(CONDITIONS),
        "transient_retries": config["transient_retries"],
        "sha256": current_hashes(),
    }

    PROTOCOL_FILE.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {PROTOCOL_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
