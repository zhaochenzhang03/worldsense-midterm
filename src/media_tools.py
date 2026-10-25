import json
import os
import re
import shutil
import subprocess
from pathlib import Path


# Directory holding ffmpeg/ffprobe. Overrides PATH when set, e.g.
#   export WORLDSENSE_FFMPEG_DIR=~/miniforge3/envs/worldsense-ffmpeg/bin
FFMPEG_DIR_ENV = "WORLDSENSE_FFMPEG_DIR"


def find_tool(name: str) -> str:
    """
    Return the path to ffmpeg/ffprobe, preferring
    $WORLDSENSE_FFMPEG_DIR over PATH.
    """
    directory = os.environ.get(FFMPEG_DIR_ENV)

    if directory:
        path = Path(directory).expanduser() / name
        if path.exists():
            return str(path)

    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"{name} not found. Activate the worldsense-ffmpeg conda env "
            f"or set {FFMPEG_DIR_ENV} (see README.md)."
        )

    return path


def ffmpeg_version() -> str:
    """
    Return the first line of `ffmpeg -version`,
    e.g. "ffmpeg version 9.0.2 ...".
    """
    result = subprocess.run(
        [find_tool("ffmpeg"), "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    first_line = result.stdout.splitlines()[0]

    return re.sub(r"\s+Copyright.*$", "", first_line)


def ffprobe(path: str | Path) -> dict:
    """
    Run ffprobe on a media file and return its format and
    stream information as parsed JSON. Read-only.
    """
    result = subprocess.run(
        [
            find_tool("ffprobe"),
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def stream_packet_hash(path: str | Path, stream: str) -> str:
    """
    SHA-256 over the encoded packets of one stream (e.g. "0:v:0"),
    via ffmpeg's streamhash muxer with stream copy. Equal hashes
    mean the bitstream was copied without re-encoding. Read-only.
    """
    result = subprocess.run(
        [
            find_tool("ffmpeg"),
            "-nostdin",
            "-v", "error",
            "-i", str(path),
            "-map", stream,
            "-c", "copy",
            "-f", "streamhash",
            "-hash", "sha256",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip().rsplit("=", 1)[-1]


def streams_of_type(info: dict, codec_type: str) -> list[dict]:
    return [
        stream
        for stream in info.get("streams", [])
        if stream.get("codec_type") == codec_type
    ]


def as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_streams(info: dict) -> dict:
    """
    Compact description of a probed file: container duration and
    the codec parameters of each video/audio stream.
    """
    video = []
    for stream in streams_of_type(info, "video"):
        rotation = None
        for side_data in stream.get("side_data_list", []):
            if "rotation" in side_data:
                rotation = side_data["rotation"]

        video.append({
            "codec": stream.get("codec_name"),
            "profile": stream.get("profile"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "pix_fmt": stream.get("pix_fmt"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "rotation": rotation,
            "duration": as_float(stream.get("duration")),
        })

    audio = []
    for stream in streams_of_type(info, "audio"):
        audio.append({
            "codec": stream.get("codec_name"),
            "profile": stream.get("profile"),
            "sample_rate": int(stream["sample_rate"])
            if stream.get("sample_rate") else None,
            "channels": stream.get("channels"),
            "channel_layout": stream.get("channel_layout"),
            "duration": as_float(stream.get("duration")),
        })

    other = sorted(
        stream.get("codec_type", "unknown")
        for stream in info.get("streams", [])
        if stream.get("codec_type") not in ("video", "audio")
    )

    return {
        "format": info.get("format", {}).get("format_name"),
        "duration": as_float(info.get("format", {}).get("duration")),
        "video_streams": video,
        "audio_streams": audio,
        "other_streams": other,
    }
