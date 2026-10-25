import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.download_frozen_videos import (
    MEDIA_MANIFEST,
    VIDEO_DIR,
    file_crc32_and_sha256,
    frozen_examples,
    write_json_atomic,
)
from src.media_tools import (
    ffmpeg_version,
    ffprobe,
    find_tool,
    summarize_streams,
)


VARIANT_DIR = Path("data/worldsense/variants")
VIDEO_ONLY_DIR = VARIANT_DIR / "video_only"
AUDIO_ONLY_DIR = VARIANT_DIR / "audio_only"
VARIANT_MANIFEST = Path("configs/variant_manifest.json")

# Maximum allowed |variant duration - original duration| in seconds,
# both container-level and for the copied stream. Original video and
# audio stream durations differ by at most 0.041 s in the frozen 60.
DURATION_TOLERANCE_SECONDS = 0.1

# Stream copy only (no re-encoding). Metadata, chapters, subtitle and
# data streams are dropped; bitexact flags keep the muxer from writing
# an encoder version string, so outputs are reproducible.
COMMON_ARGS = [
    "-map_metadata", "-1",
    "-map_chapters", "-1",
    "-sn",
    "-dn",
    "-fflags", "+bitexact",
]

VARIANTS = {
    "video_only": {
        "dir": VIDEO_ONLY_DIR,
        "suffix": ".mp4",
        "args": [
            "-map", "0:v:0", "-c:v", "copy", "-an",
            *COMMON_ARGS, "-flags:v", "+bitexact",
            "-f", "mp4",
        ],
        "keep": "video",
        "drop": "audio",
    },
    "audio_only": {
        "dir": AUDIO_ONLY_DIR,
        "suffix": ".m4a",
        "args": [
            "-map", "0:a:0", "-c:a", "copy", "-vn",
            *COMMON_ARGS, "-flags:a", "+bitexact",
            "-f", "ipod",
        ],
        "keep": "audio",
        "drop": "video",
    },
}


def ffmpeg_command(variant: str, src: Path, dst: Path) -> list[str]:
    return [
        find_tool("ffmpeg"),
        "-nostdin",
        "-v", "error",
        "-i", str(src),
        *VARIANTS[variant]["args"],
        str(dst),
    ]


def check_variant(variant: str, path: Path, original: dict) -> tuple[dict, list[str]]:
    """
    Probe a variant file and compare it with the original's stream
    summary. Returns (variant summary, list of problems).
    """
    spec = VARIANTS[variant]
    summary = summarize_streams(ffprobe(path))
    problems = []

    kept = summary[f"{spec['keep']}_streams"]
    dropped = summary[f"{spec['drop']}_streams"]
    original_kept = original[f"{spec['keep']}_streams"][0]

    if len(kept) != 1:
        problems.append(f"expected 1 {spec['keep']} stream, found {len(kept)}")
    if dropped:
        problems.append(f"contains a {spec['drop']} stream")
    if summary["other_streams"]:
        problems.append(f"extra streams: {summary['other_streams']}")

    if len(kept) == 1:
        for key in ("codec", "profile"):
            if kept[0][key] != original_kept[key]:
                problems.append(
                    f"{spec['keep']} {key} {kept[0][key]} != "
                    f"original {original_kept[key]}"
                )

        comparisons = [
            ("container", summary["duration"], original["duration"]),
            ("stream", kept[0]["duration"], original_kept["duration"]),
        ]
        for label, actual, reference in comparisons:
            if actual is None or reference is None:
                problems.append(f"{label} duration unavailable")
            elif abs(actual - reference) > DURATION_TOLERANCE_SECONDS:
                problems.append(
                    f"{label} duration {actual:.3f}s differs from "
                    f"original {reference:.3f}s"
                )

    return summary, problems


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create video_only (.mp4, audio removed) and audio_only "
            "(.m4a) variants of the frozen WorldSense videos by "
            "stream copy."
        )
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="all",
        help="Which frozen examples to process (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate variants even if they already match the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = ffmpeg_version()
    print(f"Using {version} ({find_tool('ffmpeg')})")

    with open(MEDIA_MANIFEST, "r", encoding="utf-8") as f:
        media = {v["video_id"]: v for v in json.load(f)["videos"]}

    entries = {}
    if VARIANT_MANIFEST.exists():
        with open(VARIANT_MANIFEST, "r", encoding="utf-8") as f:
            entries = {v["video_id"]: v for v in json.load(f)["videos"]}

    for spec in VARIANTS.values():
        spec["dir"].mkdir(parents=True, exist_ok=True)

        # Leftovers from interrupted runs are never complete.
        for stale in spec["dir"].glob("*.part"):
            print(f"Removing incomplete output {stale}")
            stale.unlink()

    examples = [
        (split, x)
        for split, x in frozen_examples()
        if args.split in ("all", split)
    ]
    frozen_order = [x["video_id"] for _, x in frozen_examples()]

    def save_manifest() -> None:
        write_json_atomic(VARIANT_MANIFEST, {
            "description": (
                "Deterministic modality variants of the frozen 60 "
                "WorldSense videos (scripts/create_modality_variants.py). "
                "video_subtitle reuses the video_only file."
            ),
            "ffmpeg_version": version,
            "duration_tolerance_seconds": DURATION_TOLERANCE_SECONDS,
            "commands": {
                variant: "ffmpeg -nostdin -v error -i INPUT "
                + " ".join(spec["args"]) + " OUTPUT"
                for variant, spec in VARIANTS.items()
            },
            "num_videos": len(entries),
            "videos": [
                entries[video_id]
                for video_id in frozen_order
                if video_id in entries
            ],
        })

    created = 0
    skipped = 0
    failed = []

    for index, (split_name, example) in enumerate(examples, start=1):
        video_id = example["video_id"]
        prefix = f"[{index:>2}/{len(examples)}] {video_id}"
        src = VIDEO_DIR / f"{video_id}.mp4"

        # The original must be the exact downloaded file.
        _, original_sha = file_crc32_and_sha256(src)
        if original_sha != media[video_id]["sha256"]:
            print(f"{prefix}: FAILED, original SHA-256 does not match media manifest")
            failed.append(video_id)
            continue

        original = summarize_streams(ffprobe(src))
        old = entries.get(video_id, {})
        entry = {
            "video_id": video_id,
            "split": split_name,
            "original_path": str(src),
            "original_sha256": original_sha,
            "original_duration": original["duration"],
            "original_codecs": {
                "video": original["video_streams"][0]["codec"],
                "audio": original["audio_streams"][0]["codec"],
            },
            "ffmpeg_version": version,
        }
        actions = []

        for variant, spec in VARIANTS.items():
            out_path = spec["dir"] / f"{video_id}{spec['suffix']}"
            recorded_sha = old.get(f"{variant}_sha256")

            is_validated = (
                out_path.exists()
                and recorded_sha is not None
                and file_crc32_and_sha256(out_path)[1] == recorded_sha
            )

            if is_validated and not args.force:
                for key in ("path", "sha256", "duration", "codec"):
                    entry[f"{variant}_{key}"] = old[f"{variant}_{key}"]
                actions.append(f"{variant} skipped")
                continue

            tmp_path = out_path.with_name(out_path.name + ".part")

            try:
                subprocess.run(
                    ffmpeg_command(variant, src, tmp_path),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                summary, problems = check_variant(variant, tmp_path, original)
                if problems:
                    raise RuntimeError("; ".join(problems))
            except (subprocess.CalledProcessError, RuntimeError) as error:
                tmp_path.unlink(missing_ok=True)
                detail = getattr(error, "stderr", None) or str(error)
                actions.append(f"{variant} FAILED ({detail.strip()})")
                break

            os.replace(tmp_path, out_path)
            _, sha256 = file_crc32_and_sha256(out_path)

            entry[f"{variant}_path"] = str(out_path)
            entry[f"{variant}_sha256"] = sha256
            entry[f"{variant}_duration"] = summary["duration"]
            entry[f"{variant}_codec"] = summary[f"{spec['keep']}_streams"][0]["codec"]
            actions.append(f"{variant} created")

        if any("FAILED" in action for action in actions):
            print(f"{prefix}: " + ", ".join(actions))
            failed.append(video_id)
            continue

        entries[video_id] = entry
        save_manifest()

        if all("skipped" in action for action in actions):
            skipped += 1
        else:
            created += 1
        print(f"{prefix}: " + ", ".join(actions))

    print()
    print(f"Created/updated: {created}  Skipped (already validated): {skipped}  "
          f"Failed: {len(failed)}")
    print(f"Variant manifest: {VARIANT_MANIFEST} ({len(entries)} videos)")

    if failed:
        print(f"Failed videos: {failed}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
