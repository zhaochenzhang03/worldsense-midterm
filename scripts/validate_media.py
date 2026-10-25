import json
import shutil
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.download_frozen_videos import (
    MEDIA_MANIFEST,
    VIDEO_DIR,
    file_crc32_and_sha256,
    frozen_examples,
)
from scripts.create_split import parse_duration_seconds


# Report (but do not fail on) differences larger than this between the
# container duration and the dataset's integer-second video_duration.
DURATION_TOLERANCE_SECONDS = 2.0

# MP4 boxes that contain child boxes on the path moov/trak/mdia/hdlr.
CONTAINER_BOXES = {b"moov", b"trak", b"mdia"}


def iter_boxes(f, start: int, end: int):
    """
    Yield (box type, content start, box end) for ISO-BMFF boxes
    between byte offsets start and end of an open file.
    """
    pos = start

    while pos + 8 <= end:
        f.seek(pos)
        size, box_type = struct.unpack(">I4s", f.read(8))
        header = 8

        if size == 1:
            (size,) = struct.unpack(">Q", f.read(8))
            header = 16
        elif size == 0:
            size = end - pos

        if size < header or pos + size > end:
            raise ValueError(f"malformed MP4 box {box_type!r} at {pos}")

        yield box_type, pos + header, pos + size
        pos += size


def probe_mp4(path: Path) -> dict:
    """
    Minimal read-only MP4 parser (standard library only):
    container duration from moov/mvhd and track handler types
    from moov/trak/mdia/hdlr.
    """
    duration = None
    handlers = []

    def walk(f, start: int, end: int) -> None:
        nonlocal duration

        for box_type, content, box_end in iter_boxes(f, start, end):
            if box_type in CONTAINER_BOXES:
                walk(f, content, box_end)

            elif box_type == b"mvhd":
                f.seek(content)
                version = f.read(1)[0]
                if version == 1:
                    f.seek(content + 20)
                    timescale, length = struct.unpack(">IQ", f.read(12))
                else:
                    f.seek(content + 12)
                    timescale, length = struct.unpack(">II", f.read(8))
                duration = length / timescale if timescale else None

            elif box_type == b"hdlr":
                f.seek(content + 8)
                handlers.append(f.read(4).decode("latin-1"))

    with open(path, "rb") as f:
        size = path.stat().st_size
        walk(f, 0, size)

    if duration is None:
        raise ValueError("no moov/mvhd box found")

    return {
        "duration": duration,
        "has_video": "vide" in handlers,
        "has_audio": "soun" in handlers,
    }


def probe_ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    codec_types = [s.get("codec_type") for s in info.get("streams", [])]

    return {
        "duration": float(info["format"]["duration"]),
        "has_video": "video" in codec_types,
        "has_audio": "audio" in codec_types,
    }


def main() -> int:
    failures = []

    def check(condition: bool, message: str) -> None:
        print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
        if not condition:
            failures.append(message)

    examples = frozen_examples()
    expected = {x["video_id"]: (split, x) for split, x in examples}

    with open(MEDIA_MANIFEST, "r", encoding="utf-8") as f:
        manifest = {v["video_id"]: v for v in json.load(f)["videos"]}

    use_ffprobe = shutil.which("ffprobe") is not None
    probe = probe_ffprobe if use_ffprobe else probe_mp4

    # ---------------------------------------------------------
    # 1. File set
    # ---------------------------------------------------------
    print("=" * 70)
    print(f"Media files ({VIDEO_DIR})")
    print("=" * 70)

    present = sorted(p.name for p in VIDEO_DIR.iterdir()) if VIDEO_DIR.is_dir() else []
    expected_names = {f"{video_id}.mp4" for video_id in expected}
    missing = sorted(expected_names - set(present))
    unexpected = sorted(set(present) - expected_names)

    check(
        len(present) == len(expected) and not missing and not unexpected,
        f"exactly {len(expected)} expected video files "
        f"(found {len(present)})",
    )
    check(not missing, f"no frozen video missing (missing: {missing})")
    check(not unexpected, f"no unexpected files (unexpected: {unexpected})")
    check(
        set(manifest) == set(expected),
        f"media manifest covers exactly the frozen videos "
        f"({len(manifest)} entries)",
    )

    # ---------------------------------------------------------
    # 2. Per-file integrity and probing (read-only)
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Per-file checks (probe: {'ffprobe' if use_ffprobe else 'built-in MP4 parser; ffprobe not found'})")
    print("=" * 70)
    print(
        f"  {'split':<5} {'video_id':<10} {'MB':>6} {'sha':<4} "
        f"{'expected':>8} {'actual':>8} {'diff':>6} {'video':<5} {'audio':<5}"
    )

    rows = []

    for video_id, (split, example) in expected.items():
        path = VIDEO_DIR / f"{video_id}.mp4"
        if not path.exists():
            continue

        size = path.stat().st_size
        entry = manifest.get(video_id, {})
        _, sha256 = file_crc32_and_sha256(path)
        sha_ok = sha256 == entry.get("sha256")

        if size == 0:
            failures.append(f"{video_id}: empty file")
        if not sha_ok:
            failures.append(f"{video_id}: SHA-256 does not match manifest")
        if size != entry.get("file_size"):
            failures.append(f"{video_id}: size does not match manifest")

        try:
            info = probe(path)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            failures.append(f"{video_id}: cannot be opened/parsed ({error})")
            info = None

        expected_duration = parse_duration_seconds(example["video_duration"])
        row = {
            "video_id": video_id,
            "split": split,
            "expected": expected_duration,
            "actual": info["duration"] if info else None,
            "has_video": info["has_video"] if info else None,
            "has_audio": info["has_audio"] if info else None,
            "speech": "Speech" in example["audio_class"],
        }
        rows.append(row)

        diff = (
            row["actual"] - expected_duration
            if info and expected_duration is not None
            else None
        )
        row["diff"] = diff

        print(
            f"  {split:<5} {video_id:<10} {size / 1e6:>6.1f} "
            f"{'ok' if sha_ok else 'BAD':<4} "
            f"{expected_duration if expected_duration is not None else '?':>8} "
            f"{row['actual'] if info else float('nan'):>8.2f} "
            f"{diff if diff is not None else float('nan'):>+6.2f} "
            f"{'yes' if row['has_video'] else 'NO':<5} "
            f"{'yes' if row['has_audio'] else 'NO':<5}"
        )

    # ---------------------------------------------------------
    # 3. Summary
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    probed = [r for r in rows if r["actual"] is not None]
    total_mb = sum((VIDEO_DIR / f"{r['video_id']}.mp4").stat().st_size for r in rows) / 1e6
    print(f"  files checked: {len(rows)}  total size: {total_mb:.1f} MB")

    if probed:
        diffs = [abs(r["diff"]) for r in probed if r["diff"] is not None]
        print(
            f"  actual duration (s): min={min(r['actual'] for r in probed):.2f} "
            f"max={max(r['actual'] for r in probed):.2f}"
        )
        print(f"  |actual - dataset duration| (s): max={max(diffs):.2f}")

        off = [r for r in probed if r["diff"] is not None and abs(r["diff"]) > DURATION_TOLERANCE_SECONDS]
        print(f"  duration differs by > {DURATION_TOLERANCE_SECONDS}s: {len(off)}")
        for r in off:
            print(f"    {r['video_id']}: dataset {r['expected']}s, actual {r['actual']:.2f}s")

        over = [r["video_id"] for r in probed if r["actual"] > 120]
        print(f"  actual duration > 120s: {over}")

        print(f"  with video stream: {sum(bool(r['has_video']) for r in probed)}/{len(probed)}")
        print(f"  with audio stream: {sum(bool(r['has_audio']) for r in probed)}/{len(probed)}")

        no_audio = [r["video_id"] for r in probed if not r["has_audio"]]
        no_video = [r["video_id"] for r in probed if not r["has_video"]]
        print(f"  without audio stream: {no_audio}")
        print(f"  without video stream: {no_video}")

        audio_by_speech = Counter(
            (r["speech"], bool(r["has_audio"])) for r in probed
        )
        print(
            "  audio stream present | Speech in audio_class: "
            f"{audio_by_speech[(True, True)]}/{audio_by_speech[(True, True)] + audio_by_speech[(True, False)]}; "
            "| no Speech: "
            f"{audio_by_speech[(False, True)]}/{audio_by_speech[(False, True)] + audio_by_speech[(False, False)]}"
        )

    print()
    if failures:
        print(f"MEDIA VALIDATION FAILED: {len(failures)} problem(s):")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("MEDIA VALIDATION PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
