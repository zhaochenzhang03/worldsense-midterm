import hashlib
import http.client
import json
import os
import struct
import sys
import zlib
from pathlib import Path

from scripts.create_split import (
    DEV_OUTPUT,
    TEST_OUTPUT,
    parse_duration_seconds,
)
from scripts.map_video_archives import (
    MAP_OUTPUT,
    RESOLVE_URL,
    fetch_range,
    list_video_archives,
    open_range,
    read_central_directory_entries,
    with_retries,
)


VIDEO_DIR = Path("data/worldsense/videos")
MEDIA_MANIFEST = Path("configs/media_manifest.json")

LOCAL_HEADER_SIG = b"PK\x03\x04"
CHUNK_SIZE = 1 << 20

METHOD_STORED = 0
METHOD_DEFLATED = 8


def frozen_examples() -> list[tuple[str, dict]]:
    """
    Return [(split name, example)] for the frozen split, in order.
    """
    examples = []

    for split_name, path in (("dev", DEV_OUTPUT), ("test", TEST_OUTPUT)):
        with open(path, "r", encoding="utf-8") as f:
            examples.extend((split_name, x) for x in json.load(f))

    return examples


def file_crc32_and_sha256(path: Path) -> tuple[int, str]:
    crc = 0
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)

    return crc, digest.hexdigest()


def write_json_atomic(path: Path, data) -> None:
    tmp_path = path.with_name(path.name + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    os.replace(tmp_path, path)


def download_member(url: str, entry: dict, out_path: Path) -> int:
    """
    Download one zip member with HTTP range requests, decompress
    it into a temporary file, verify size and CRC-32 against the
    central directory, then atomically rename to out_path.

    Returns the number of compressed bytes downloaded.
    """
    if entry["flags"] & 0x1:
        raise RuntimeError(f"{entry['name']} is encrypted")

    if entry["method"] not in (METHOD_STORED, METHOD_DEFLATED):
        raise RuntimeError(
            f"{entry['name']}: unsupported compression method "
            f"{entry['method']}"
        )

    # The local header's extra field may differ from the central
    # directory's, so read it to find where the data starts.
    offset = entry["header_offset"]
    header = fetch_range(url, offset, offset + 29)

    if header[:4] != LOCAL_HEADER_SIG:
        raise RuntimeError(f"{entry['name']}: bad local file header")

    name_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = offset + 30 + name_len + extra_len
    data_end = data_start + entry["compressed_size"] - 1

    tmp_path = out_path.with_name(out_path.name + ".part")

    def attempt() -> None:
        decompressor = (
            zlib.decompressobj(-15)
            if entry["method"] == METHOD_DEFLATED
            else None
        )
        received = 0
        crc = 0
        written = 0

        with (
            open_range(url, data_start, data_end) as response,
            open(tmp_path, "wb") as out,
        ):
            while received < entry["compressed_size"]:
                chunk = response.read(
                    min(CHUNK_SIZE, entry["compressed_size"] - received)
                )
                if not chunk:
                    raise http.client.IncompleteRead(
                        b"", entry["compressed_size"] - received
                    )
                received += len(chunk)

                data = (
                    decompressor.decompress(chunk)
                    if decompressor
                    else chunk
                )
                crc = zlib.crc32(data, crc)
                written += len(data)
                out.write(data)

            if decompressor:
                data = decompressor.flush()
                crc = zlib.crc32(data, crc)
                written += len(data)
                out.write(data)

                if not decompressor.eof:
                    raise RuntimeError(
                        f"{entry['name']}: deflate stream incomplete"
                    )

            out.flush()
            os.fsync(out.fileno())

        if written != entry["file_size"]:
            raise RuntimeError(
                f"{entry['name']}: size {written} != "
                f"expected {entry['file_size']}"
            )

        if crc != entry["crc32"]:
            raise RuntimeError(
                f"{entry['name']}: CRC-32 {crc:08x} != "
                f"expected {entry['crc32']:08x}"
            )

    try:
        with_retries(f"download {entry['name']}", attempt)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    os.replace(tmp_path, out_path)

    return entry["compressed_size"]


def main() -> int:
    with open(MAP_OUTPUT, "r", encoding="utf-8") as f:
        archive_map = json.load(f)

    examples = frozen_examples()

    missing_from_map = [
        x["video_id"]
        for _, x in examples
        if x["video_id"] not in archive_map
    ]
    if missing_from_map:
        print(
            f"ERROR: not in {MAP_OUTPUT}: {missing_from_map}",
            file=sys.stderr,
        )
        return 1

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Remove leftovers from interrupted runs; they are never complete.
    for stale in VIDEO_DIR.glob("*.part"):
        print(f"Removing incomplete download {stale}")
        stale.unlink()

    # ---------------------------------------------------------
    # 1. Read the central directory of each needed archive.
    # ---------------------------------------------------------
    archive_sizes = list_video_archives()
    entries_by_archive = {}

    for archive in sorted(set(archive_map.values())):
        entries = read_central_directory_entries(
            f"{RESOLVE_URL}/{archive}",
            archive_sizes[archive],
        )
        entries_by_archive[archive] = {
            entry["name"]: entry
            for entry in entries
        }

    # ---------------------------------------------------------
    # 2. Download (or verify and skip) each frozen video.
    #    The manifest is rewritten after every video.
    # ---------------------------------------------------------
    manifest_entries = []
    downloaded = 0
    skipped = 0
    downloaded_bytes = 0
    failed = []

    def save_manifest() -> None:
        write_json_atomic(MEDIA_MANIFEST, {
            "description": (
                "Frozen 60 WorldSense videos, extracted member-by-member "
                "from the official zip archives via HTTP range requests "
                "by scripts/download_frozen_videos.py."
            ),
            "source_repository": RESOLVE_URL.rsplit("/resolve/", 1)[0],
            "video_dir": str(VIDEO_DIR),
            "num_videos": len(manifest_entries),
            "videos": manifest_entries,
        })

    for index, (split_name, example) in enumerate(examples, start=1):
        video_id = example["video_id"]
        archive = archive_map[video_id]
        member = f"{video_id}.mp4"
        url = f"{RESOLVE_URL}/{archive}"
        out_path = VIDEO_DIR / member
        prefix = f"[{index:>2}/{len(examples)}] {video_id}"

        entry = entries_by_archive[archive].get(member)
        if entry is None:
            print(f"{prefix}: FAILED, {member} not in {archive}")
            failed.append(video_id)
            continue

        is_complete = False
        if out_path.exists() and out_path.stat().st_size == entry["file_size"]:
            crc, sha256 = file_crc32_and_sha256(out_path)
            is_complete = crc == entry["crc32"]

        if is_complete:
            skipped += 1
            print(f"{prefix}: already complete, skipped")
        else:
            try:
                downloaded_bytes += download_member(url, entry, out_path)
            except Exception as error:
                print(f"{prefix}: FAILED, {error!r}")
                failed.append(video_id)
                continue

            downloaded += 1
            _, sha256 = file_crc32_and_sha256(out_path)
            print(
                f"{prefix}: downloaded "
                f"{entry['compressed_size'] / 1e6:.1f} MB from {archive} "
                f"(total {downloaded_bytes / 1e6:.1f} MB)"
            )

        manifest_entries.append({
            "video_id": video_id,
            "split": split_name,
            "archive": archive,
            "source_url": url,
            "zip_member": member,
            "local_path": str(out_path),
            "file_size": entry["file_size"],
            "zip_crc32": f"{entry['crc32']:08x}",
            "sha256": sha256,
            "expected_duration_seconds": parse_duration_seconds(
                example["video_duration"]
            ),
        })
        save_manifest()

    print()
    print(f"Downloaded: {downloaded}  Skipped (already complete): {skipped}  "
          f"Failed: {len(failed)}")
    print(f"Compressed bytes downloaded this run: {downloaded_bytes / 1e6:.1f} MB")
    print(f"Media manifest: {MEDIA_MANIFEST} ({len(manifest_entries)} videos)")

    if failed:
        print(f"Failed videos (re-run to resume): {failed}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
