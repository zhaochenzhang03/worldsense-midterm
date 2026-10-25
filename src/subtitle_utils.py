import re
from pathlib import Path


TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
)

# A line consisting only of bracketed annotations, e.g. "[Music]"
# or "[Applause] [Music]".
NON_SPEECH_TAG_LINE_RE = re.compile(r"^(?:\[[^\]]*\]\s*)+$")


def extract_subtitle_lines(srt_text: str) -> list[str]:
    """
    Return the cue text lines of an SRT document in order,
    dropping sequence numbers, timestamps and blank lines.

    A digit-only line is treated as a sequence number only when
    the next line is a timestamp, so numeric subtitle text such
    as "2001" is preserved.

    Raises ValueError if the document is non-empty but contains
    no SRT timestamp lines.
    """
    lines = [line.strip() for line in srt_text.splitlines()]

    if any(lines) and not any(TIMESTAMP_RE.match(x) for x in lines):
        raise ValueError("no SRT timestamp lines found")

    text_lines = []

    for i, line in enumerate(lines):
        if not line or TIMESTAMP_RE.match(line):
            continue

        is_sequence_number = (
            line.isdigit()
            and i + 1 < len(lines)
            and TIMESTAMP_RE.match(lines[i + 1])
        )
        if is_sequence_number:
            continue

        text_lines.append(line)

    return text_lines


def dedupe_consecutive_lines(lines: list[str]) -> list[str]:
    """
    Remove a line only when it exactly equals the line directly
    before it (rolling-caption repeats). Identical text that
    recurs later, after a different line, is kept.
    """
    return [
        line
        for i, line in enumerate(lines)
        if i == 0 or line != lines[i - 1]
    ]


def normalize_subtitle_text(srt_text: str) -> str:
    """
    Frozen subtitle normalization used for model input:
      - remove SRT sequence numbers and timestamps
      - remove consecutive exact duplicate lines only
        (rolling captions); recurring text is kept
      - keep all text, including bracketed tags like [Music],
        exactly as written
    Lines are joined with "\\n". Returns "" for an empty file.
    """
    return "\n".join(
        dedupe_consecutive_lines(extract_subtitle_lines(srt_text))
    )


def load_normalized_subtitle(path: str | Path) -> str:
    """
    Read one official WorldSense .srt file (UTF-8) and return
    its normalized plain text.
    """
    raw = Path(path).read_bytes()

    return normalize_subtitle_text(raw.decode("utf-8-sig"))


def is_non_speech_tags_only(normalized_text: str) -> bool:
    """
    True if the text is non-empty and every line consists only
    of bracketed annotations such as [Music].
    """
    lines = normalized_text.splitlines()

    return bool(lines) and all(
        NON_SPEECH_TAG_LINE_RE.match(line)
        for line in lines
    )
