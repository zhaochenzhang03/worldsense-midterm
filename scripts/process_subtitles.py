import json
import sys
from pathlib import Path

from scripts.create_split import DEV_OUTPUT, TEST_OUTPUT
from src.subtitle_utils import load_normalized_subtitle


SUBTITLE_DIR = Path("data/worldsense/subtitles")
PROCESSED_DIR = Path("data/worldsense/processed_subtitles")


def frozen_video_ids() -> list[str]:
    video_ids = []

    for path in (DEV_OUTPUT, TEST_OUTPUT):
        with open(path, "r", encoding="utf-8") as f:
            video_ids.extend(x["video_id"] for x in json.load(f))

    return video_ids


def main() -> int:
    """
    Write one normalized subtitle .txt per frozen video_id.
    Output is deterministic, so re-running overwrites files
    with identical content.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    written = 0

    for video_id in frozen_video_ids():
        srt_path = SUBTITLE_DIR / f"{video_id}.srt"

        if not srt_path.exists():
            missing.append(video_id)
            continue

        text = load_normalized_subtitle(srt_path)

        with open(
            PROCESSED_DIR / f"{video_id}.txt",
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:
            f.write(text)

        written += 1

    print(f"Wrote {written} normalized subtitles to {PROCESSED_DIR}")

    if missing:
        print(f"ERROR: no raw subtitle for: {missing}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
