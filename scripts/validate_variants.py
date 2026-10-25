import json
import subprocess
import sys

from scripts.create_modality_variants import (
    VARIANT_MANIFEST,
    VARIANTS,
    check_variant,
)
from scripts.download_frozen_videos import (
    MEDIA_MANIFEST,
    VIDEO_DIR,
    file_crc32_and_sha256,
    frozen_examples,
)
from src.media_tools import (
    ffmpeg_version,
    ffprobe,
    stream_packet_hash,
    summarize_streams,
)


def main() -> int:
    failures = []

    def check(condition: bool, message: str) -> None:
        print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
        if not condition:
            failures.append(message)

    examples = frozen_examples()
    frozen_ids = [x["video_id"] for _, x in examples]

    with open(MEDIA_MANIFEST, "r", encoding="utf-8") as f:
        media = {v["video_id"]: v for v in json.load(f)["videos"]}

    with open(VARIANT_MANIFEST, "r", encoding="utf-8") as f:
        variant_manifest = json.load(f)
    entries = {v["video_id"]: v for v in variant_manifest["videos"]}

    print(f"Using {ffmpeg_version()}; manifest built with "
          f"{variant_manifest['ffmpeg_version']}")

    # ---------------------------------------------------------
    # 1. File sets
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("File sets")
    print("=" * 70)

    check(
        set(entries) == set(frozen_ids),
        f"variant manifest covers exactly the {len(frozen_ids)} "
        f"frozen videos ({len(entries)} entries)",
    )

    for variant, spec in VARIANTS.items():
        present = sorted(p.name for p in spec["dir"].iterdir()) if spec["dir"].is_dir() else []
        expected = {f"{video_id}{spec['suffix']}" for video_id in frozen_ids}
        missing = sorted(expected - set(present))
        extra = sorted(set(present) - expected)

        check(
            len(present) == len(frozen_ids) and not missing and not extra,
            f"exactly {len(frozen_ids)} {variant} files (found {len(present)}; "
            f"missing {missing}; extra {extra})",
        )

    # ---------------------------------------------------------
    # 2. Per-video checks
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("Per-video checks")
    print("=" * 70)

    max_diff = {variant: 0.0 for variant in VARIANTS}

    for video_id in frozen_ids:
        entry = entries.get(video_id)
        src = VIDEO_DIR / f"{video_id}.mp4"
        problems = []

        if entry is None:
            continue

        # Originals must be unchanged.
        _, original_sha = file_crc32_and_sha256(src)
        if original_sha != media[video_id]["sha256"]:
            problems.append("original SHA-256 differs from media manifest")
        if original_sha != entry["original_sha256"]:
            problems.append("original SHA-256 differs from variant manifest")

        original = summarize_streams(ffprobe(src))

        for variant, spec in VARIANTS.items():
            path = spec["dir"] / f"{video_id}{spec['suffix']}"
            if not path.exists():
                continue

            _, sha256 = file_crc32_and_sha256(path)
            if sha256 != entry.get(f"{variant}_sha256"):
                problems.append(f"{variant} SHA-256 differs from manifest")

            try:
                summary, variant_problems = check_variant(variant, path, original)
            except subprocess.CalledProcessError as error:
                problems.append(f"{variant}: ffprobe failed ({error.stderr.strip()})")
                continue

            problems.extend(f"{variant}: {p}" for p in variant_problems)

            if summary["duration"] is not None and original["duration"] is not None:
                max_diff[variant] = max(
                    max_diff[variant],
                    abs(summary["duration"] - original["duration"]),
                )

            # Stream copy: encoded packets must be identical.
            stream = f"0:{spec['keep'][0]}:0"
            if stream_packet_hash(path, stream) != stream_packet_hash(src, stream):
                problems.append(
                    f"{variant}: {spec['keep']} packets differ from original "
                    "(not a pure stream copy)"
                )

        status = "ok" if not problems else "FAIL: " + "; ".join(problems)
        print(f"  {entry['split']:<5} {video_id:<10} {status}")
        failures.extend(f"{video_id}: {p}" for p in problems)

    # ---------------------------------------------------------
    # 3. Summary
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    for variant in VARIANTS:
        print(
            f"  {variant}: max |container duration - original| = "
            f"{max_diff[variant]:.3f}s"
        )

    print()
    if failures:
        print(f"VARIANT VALIDATION FAILED: {len(failures)} problem(s):")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("VARIANT VALIDATION PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
