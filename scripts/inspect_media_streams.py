import json
import sys
from collections import Counter
from pathlib import Path

from scripts.download_frozen_videos import (
    VIDEO_DIR,
    frozen_examples,
    write_json_atomic,
)
from src.media_tools import ffmpeg_version, ffprobe, summarize_streams


STREAM_METADATA = Path("configs/media_stream_metadata.json")


def main() -> int:
    """
    Record ffprobe stream metadata for the 60 original frozen
    videos (read-only) and summarize codec consistency.
    """
    videos = []

    for split_name, example in frozen_examples():
        video_id = example["video_id"]
        path = VIDEO_DIR / f"{video_id}.mp4"

        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

        summary = summarize_streams(ffprobe(path))
        videos.append({
            "video_id": video_id,
            "split": split_name,
            "path": str(path),
            **summary,
        })

    def first(entry: dict, kind: str, key: str):
        streams = entry[f"{kind}_streams"]
        return streams[0][key] if streams else None

    stream_layouts = Counter(
        (
            len(v["video_streams"]),
            len(v["audio_streams"]),
            len(v["other_streams"]),
        )
        for v in videos
    )
    distributions = {
        "stream_layout (video, audio, other)": {
            f"{k[0]}v/{k[1]}a/{k[2]}o": n
            for k, n in stream_layouts.items()
        },
        "video_codec": Counter(
            f"{first(v, 'video', 'codec')} ({first(v, 'video', 'profile')})"
            for v in videos
        ),
        "audio_codec": Counter(
            f"{first(v, 'audio', 'codec')} ({first(v, 'audio', 'profile')})"
            for v in videos
        ),
        "resolution": Counter(
            f"{first(v, 'video', 'width')}x{first(v, 'video', 'height')}"
            for v in videos
        ),
        "frame_rate": Counter(
            str(first(v, "video", "r_frame_rate")) for v in videos
        ),
        "rotation": Counter(
            str(first(v, "video", "rotation")) for v in videos
        ),
        "audio_sample_rate": Counter(
            str(first(v, "audio", "sample_rate")) for v in videos
        ),
        "audio_channels": Counter(
            str(first(v, "audio", "channels")) for v in videos
        ),
    }
    distributions = {
        key: dict(sorted(value.items(), key=lambda kv: -kv[1]))
        for key, value in distributions.items()
    }

    consistent = (
        len(distributions["stream_layout (video, audio, other)"]) == 1
        and len(distributions["video_codec"]) == 1
        and len(distributions["audio_codec"]) == 1
    )

    write_json_atomic(STREAM_METADATA, {
        "description": (
            "ffprobe stream metadata of the 60 original frozen "
            "WorldSense videos (scripts/inspect_media_streams.py)."
        ),
        "ffmpeg_version": ffmpeg_version(),
        "codecs_consistent": consistent,
        "distributions": distributions,
        "videos": videos,
    })

    print(f"ffprobe from: {ffmpeg_version()}")
    for key, value in distributions.items():
        print(f"  {key}: {value}")
    print(f"\nSingle stream layout and codec pair across all 60: {consistent}")
    print(f"Saved {STREAM_METADATA}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
