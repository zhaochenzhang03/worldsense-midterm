import http.client
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.process_subtitles import frozen_video_ids


REPO_URL = "https://huggingface.co/datasets/honglyhly/WorldSense"
TREE_API_URL = (
    "https://huggingface.co/api/datasets/honglyhly/WorldSense/tree/main"
)
RESOLVE_URL = f"{REPO_URL}/resolve/main"

MAP_OUTPUT = Path("configs/video_archive_map.json")

EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CENTRAL_HEADER_SIG = b"PK\x01\x02"

# EOCD record (22 bytes) + maximum zip comment (65535 bytes).
EOCD_SEARCH_BYTES = 22 + 65535


MAX_ATTEMPTS = 5


def with_retries(description: str, func):
    """
    Call func(), retrying transient network failures with
    exponential backoff (2, 4, 8, 16 s).
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return func()
        except (OSError, http.client.HTTPException) as error:
            # Client errors other than rate limiting are not transient.
            if (
                isinstance(error, urllib.error.HTTPError)
                and 400 <= error.code < 500
                and error.code != 429
            ):
                raise

            if attempt == MAX_ATTEMPTS:
                raise

            delay = 2 ** attempt
            print(
                f"    {description}: {error!r}; "
                f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)


def open_range(url: str, start: int, end: int):
    """
    Open an HTTP Range request for bytes [start, end] (inclusive)
    and return the response, checking for HTTP 206.
    """
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}"},
    )
    response = urllib.request.urlopen(request, timeout=60)

    if response.status != 206:
        response.close()
        raise RuntimeError(
            f"{url}: expected HTTP 206 for range request, "
            f"got {response.status}"
        )

    return response


def fetch_range(url: str, start: int, end: int) -> bytes:
    """
    Fetch bytes [start, end] (inclusive) with an HTTP Range request.
    """
    def attempt() -> bytes:
        with open_range(url, start, end) as response:
            data = response.read()

        if len(data) != end - start + 1:
            raise http.client.IncompleteRead(data, end - start + 1)

        return data

    return with_retries(f"range {start}-{end} of {url}", attempt)


def list_video_archives() -> dict[str, int]:
    """
    Return {archive filename: size in bytes} for every
    worldsense_videos_*.zip in the official repository.
    """
    def attempt():
        with urllib.request.urlopen(TREE_API_URL, timeout=60) as response:
            return json.load(response)

    tree = with_retries("list repository files", attempt)

    return {
        entry["path"]: entry["size"]
        for entry in tree
        if entry["path"].startswith("worldsense_videos_")
        and entry["path"].endswith(".zip")
    }


def read_central_directory(url: str, size: int) -> list[str]:
    """
    Read only the zip central directory of a remote archive
    and return the member filenames.
    """
    return [
        entry["name"]
        for entry in read_central_directory_entries(url, size)
    ]


def read_central_directory_entries(url: str, size: int) -> list[dict]:
    """
    Read only the zip central directory of a remote archive and
    return one record per member: name, flags, compression method,
    CRC-32, compressed/uncompressed size and local header offset.
    """
    tail_start = max(0, size - EOCD_SEARCH_BYTES)
    tail = fetch_range(url, tail_start, size - 1)

    eocd_pos = tail.rfind(EOCD_SIG)
    if eocd_pos < 0:
        raise RuntimeError(f"{url}: end of central directory not found")

    (
        _, _, _, _,
        num_entries,
        cd_size,
        cd_offset,
        _,
    ) = struct.unpack("<4sHHHHIIH", tail[eocd_pos:eocd_pos + 22])

    # Zip64 archives store the real values in the zip64 EOCD record.
    if 0xFFFFFFFF in (cd_size, cd_offset) or num_entries == 0xFFFF:
        locator_pos = eocd_pos - 20
        if tail[locator_pos:locator_pos + 4] != ZIP64_LOCATOR_SIG:
            raise RuntimeError(f"{url}: zip64 locator not found")

        (zip64_eocd_offset,) = struct.unpack(
            "<Q", tail[locator_pos + 8:locator_pos + 16]
        )
        record = fetch_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
        if record[:4] != ZIP64_EOCD_SIG:
            raise RuntimeError(f"{url}: zip64 EOCD record not found")

        num_entries, cd_size, cd_offset = struct.unpack(
            "<QQQ", record[32:56]
        )

    directory = fetch_range(url, cd_offset, cd_offset + cd_size - 1)

    entries = []
    pos = 0

    for _ in range(num_entries):
        if directory[pos:pos + 4] != CENTRAL_HEADER_SIG:
            raise RuntimeError(
                f"{url}: bad central directory header at offset {pos}"
            )

        flags, method = struct.unpack("<HH", directory[pos + 8:pos + 12])
        crc32, compressed_size, file_size = struct.unpack(
            "<III", directory[pos + 16:pos + 28]
        )
        name_len, extra_len, comment_len = struct.unpack(
            "<HHH", directory[pos + 28:pos + 34]
        )
        (header_offset,) = struct.unpack(
            "<I", directory[pos + 42:pos + 46]
        )
        name = directory[pos + 46:pos + 46 + name_len].decode("utf-8")

        # Zip64 extra field (id 0x0001) holds 64-bit values for any
        # field stored as 0xFFFFFFFF, in this fixed order.
        if 0xFFFFFFFF in (file_size, compressed_size, header_offset):
            extra = directory[
                pos + 46 + name_len:pos + 46 + name_len + extra_len
            ]
            values = {}
            i = 0
            while i + 4 <= len(extra):
                tag, length = struct.unpack("<HH", extra[i:i + 4])
                if tag == 0x0001:
                    data = extra[i + 4:i + 4 + length]
                    fields = [
                        key
                        for key, value in (
                            ("file_size", file_size),
                            ("compressed_size", compressed_size),
                            ("header_offset", header_offset),
                        )
                        if value == 0xFFFFFFFF
                    ]
                    for j, key in enumerate(fields):
                        (values[key],) = struct.unpack(
                            "<Q", data[8 * j:8 * j + 8]
                        )
                i += 4 + length

            file_size = values.get("file_size", file_size)
            compressed_size = values.get("compressed_size", compressed_size)
            header_offset = values.get("header_offset", header_offset)

        entries.append({
            "name": name,
            "flags": flags,
            "method": method,
            "crc32": crc32,
            "compressed_size": compressed_size,
            "file_size": file_size,
            "header_offset": header_offset,
        })

        pos += 46 + name_len + extra_len + comment_len

    return entries


def main() -> int:
    archives = list_video_archives()
    print(f"Found {len(archives)} video archives in {REPO_URL}")

    # video_id -> list of (archive, member name)
    locations: dict[str, list[tuple[str, str]]] = {}

    for archive in sorted(
        archives,
        key=lambda name: int(name.split("_")[-1].split(".")[0]),
    ):
        names = read_central_directory(
            f"{RESOLVE_URL}/{archive}",
            archives[archive],
        )
        videos = [n for n in names if not n.endswith("/")]
        print(f"  {archive}: {len(videos)} files")

        for name in videos:
            video_id = Path(name).stem
            locations.setdefault(video_id, []).append((archive, name))

    frozen = frozen_video_ids()
    archive_map = {}
    unresolved = []

    for video_id in frozen:
        found = locations.get(video_id, [])

        # Only record a mapping when membership is unambiguous.
        if len(found) == 1:
            archive_map[video_id] = found[0][0]
        else:
            unresolved.append((video_id, found))

    MAP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MAP_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(archive_map, f, indent=2)
        f.write("\n")

    print()
    print(f"Verified archive locations: {len(archive_map)}/{len(frozen)}")

    needed = sorted(set(archive_map.values()))
    needed_bytes = sum(archives[a] for a in needed)
    print(
        f"Archives needed: {len(needed)}/{len(archives)} "
        f"({needed_bytes / 1e9:.2f} GB of "
        f"{sum(archives.values()) / 1e9:.2f} GB)"
    )
    for archive in needed:
        count = sum(a == archive for a in archive_map.values())
        print(f"  {archive}: {count} frozen videos")

    if unresolved:
        print()
        print("Unresolved (not found or found in more than one archive):")
        for video_id, found in unresolved:
            print(f"  {video_id}: {found}")

    print(f"\nSaved mapping to {MAP_OUTPUT}")

    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
