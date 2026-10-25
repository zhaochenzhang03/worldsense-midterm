import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.create_split import DEV_OUTPUT, TEST_OUTPUT
from scripts.process_subtitles import PROCESSED_DIR, SUBTITLE_DIR
from scripts.validate_split import MANIFEST_PATH, sha256_of
from src.subtitle_utils import (
    is_non_speech_tags_only,
    load_normalized_subtitle,
)


SUBTITLE_ZIP = Path("data/worldsense/raw/worldsense_subtitles.zip")

STATUSES = ["usable", "tags_only", "empty", "missing", "unreadable"]


def processed_digest(texts: dict[str, str]) -> str:
    """
    One SHA-256 over all normalized subtitles: hash of the
    sorted lines "<video_id>\\t<sha256 of normalized text>".
    """
    lines = "".join(
        f"{video_id}\t"
        f"{hashlib.sha256(texts[video_id].encode('utf-8')).hexdigest()}\n"
        for video_id in sorted(texts)
    )

    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def main() -> int:
    errors = []

    # ---------------------------------------------------------
    # 1. Structural preconditions
    # ---------------------------------------------------------
    if not SUBTITLE_DIR.is_dir():
        print(
            f"ERROR: subtitle directory {SUBTITLE_DIR} not found. "
            "See README.md for how to obtain the official subtitles.",
            file=sys.stderr,
        )
        return 1

    with open(DEV_OUTPUT, "r", encoding="utf-8") as f:
        dev = json.load(f)

    with open(TEST_OUTPUT, "r", encoding="utf-8") as f:
        test = json.load(f)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        subtitle_manifest = json.load(f).get("subtitles", {})

    print("=" * 70)
    print("Subtitle archive")
    print("=" * 70)

    if SUBTITLE_ZIP.exists():
        actual_sha = sha256_of(SUBTITLE_ZIP)
        matches = actual_sha == subtitle_manifest.get("archive_sha256")
        print(
            f"  {SUBTITLE_ZIP} SHA-256: {actual_sha} "
            f"({'matches' if matches else 'DOES NOT MATCH'} manifest)"
        )
        if not matches:
            errors.append("subtitle archive SHA-256 does not match manifest")
    else:
        print(f"  {SUBTITLE_ZIP} not present; skipping archive hash check.")

    print(f"  .srt files in {SUBTITLE_DIR}: "
          f"{len(list(SUBTITLE_DIR.glob('*.srt')))}")

    # ---------------------------------------------------------
    # 2. Normalize the raw subtitle of each frozen example.
    #    Subtitle files are named <video_id>.srt.
    #    video_caption is never used as a fallback.
    # ---------------------------------------------------------
    rows = []
    normalized = {}

    for split_name, split in (("dev", dev), ("test", test)):
        for example in split:
            video_id = example["video_id"]
            has_speech = "Speech" in example["audio_class"]
            srt_path = SUBTITLE_DIR / f"{video_id}.srt"
            text = ""

            if not srt_path.exists():
                status = "missing"

                # Every WorldSense video ships a subtitle file, so a
                # missing file for a speech video means a broken
                # download/extraction.
                if has_speech:
                    errors.append(
                        f"missing subtitle for speech video {video_id}"
                    )
            else:
                try:
                    text = load_normalized_subtitle(srt_path)
                except (UnicodeDecodeError, ValueError) as error:
                    errors.append(f"{srt_path}: {error}")
                    status = "unreadable"
                else:
                    normalized[video_id] = text

                    if not text:
                        status = "empty"
                    elif is_non_speech_tags_only(text):
                        status = "tags_only"
                    else:
                        status = "usable"

            rows.append({
                "split": split_name,
                "video_id": video_id,
                "raw_found": srt_path.exists(),
                "has_speech": has_speech,
                "status": status,
                "chars": len(text),
            })

    # ---------------------------------------------------------
    # 3. Processed subtitle files: exactly one per frozen
    #    video_id, identical to a fresh normalization.
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Processed subtitles ({PROCESSED_DIR})")
    print("=" * 70)

    if not PROCESSED_DIR.is_dir():
        print("  not found; run: python -m scripts.process_subtitles")
        errors.append(f"{PROCESSED_DIR} not found")
    else:
        frozen_ids = {row["video_id"] for row in rows}
        processed_files = sorted(PROCESSED_DIR.iterdir())
        processed_ids = {p.stem for p in processed_files}

        unexpected = sorted(
            p.name
            for p in processed_files
            if p.suffix != ".txt" or p.stem not in frozen_ids
        )
        absent = sorted(frozen_ids - processed_ids)
        stale = sorted(
            video_id
            for video_id, text in normalized.items()
            if video_id in processed_ids
            and (PROCESSED_DIR / f"{video_id}.txt").read_text(
                encoding="utf-8"
            ) != text
        )

        print(f"  files: {len(processed_files)}")
        print(f"  files not mapping to a frozen video_id: {unexpected}")
        print(f"  frozen video_ids without a processed file: {absent}")
        print(f"  files differing from fresh normalization: {stale}")

        if unexpected:
            errors.append(f"unexpected processed files: {unexpected}")
        if absent:
            errors.append(f"missing processed files: {absent}")
        if stale:
            errors.append(f"stale processed files: {stale}")

        if len(normalized) == len(rows):
            digest = processed_digest(normalized)
            expected = subtitle_manifest.get("processed_digest_sha256")
            matches = digest == expected
            print(
                f"  normalized-subtitle digest: {digest} "
                f"({'matches' if matches else 'DOES NOT MATCH'} manifest)"
            )
            if not matches:
                errors.append(
                    "normalized-subtitle digest does not match manifest"
                )

    # ---------------------------------------------------------
    # 4. Report
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Per-example subtitles ({len(rows)} examples)")
    print("=" * 70)
    print(
        f"  {'split':<5} {'video_id':<10} {'raw':<4} "
        f"{'speech':<6} {'status':<10} {'chars':>6}"
    )

    for r in rows:
        print(
            f"  {r['split']:<5} {r['video_id']:<10} "
            f"{'yes' if r['raw_found'] else 'NO':<4} "
            f"{'yes' if r['has_speech'] else 'no':<6} "
            f"{r['status']:<10} {r['chars']:>6}"
        )

    print()
    print("=" * 70)
    print("Counts")
    print("=" * 70)
    print("  usable    = linguistic subtitle text present")
    print("  tags_only = only bracketed non-speech tags, e.g. [Music]")

    def report(label: str, subset: list[dict]) -> None:
        counts = Counter(row["status"] for row in subset)
        summary = ", ".join(f"{s}={counts.get(s, 0)}" for s in STATUSES)
        print(f"  {label:<16} n={len(subset):<3} {summary}")

    report("all", rows)
    report("dev", [r for r in rows if r["split"] == "dev"])
    report("test", [r for r in rows if r["split"] == "test"])
    report("with Speech", [r for r in rows if r["has_speech"]])
    report("without Speech", [r for r in rows if not r["has_speech"]])

    chars = sorted(r["chars"] for r in rows if r["status"] == "usable")
    if chars:
        print(
            f"  normalized chars (usable): min={chars[0]} "
            f"median={chars[len(chars) // 2]} max={chars[-1]}"
        )

    print()
    if errors:
        print(f"STRUCTURAL ERRORS ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
        return 1

    print("No structural errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
