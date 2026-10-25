import hashlib
from dataclasses import dataclass, field
from pathlib import Path


PROMPT_VERSION = "v1"

ORIGINAL_VIDEO_DIR = Path("data/worldsense/videos")
VIDEO_ONLY_DIR = Path("data/worldsense/variants/video_only")
AUDIO_ONLY_DIR = Path("data/worldsense/variants/audio_only")
PROCESSED_SUBTITLE_DIR = Path("data/worldsense/processed_subtitles")

CONDITIONS = [
    "full",
    "video_only",
    "audio_only",
    "video_subtitle",
    "subtitle_only",
    "question_only",
]

# Media attached per condition: (directory, file suffix, MIME type).
# Media parts are sent before the text prompt.
CONDITION_MEDIA = {
    "full": [(ORIGINAL_VIDEO_DIR, ".mp4", "video/mp4")],
    "video_only": [(VIDEO_ONLY_DIR, ".mp4", "video/mp4")],
    "audio_only": [(AUDIO_ONLY_DIR, ".m4a", "audio/m4a")],
    "video_subtitle": [(VIDEO_ONLY_DIR, ".mp4", "video/mp4")],
    "subtitle_only": [],
    "question_only": [],
}

CONDITIONS_WITH_SUBTITLE = {"video_subtitle", "subtitle_only"}

# One instruction for every condition. It never names the condition
# or says which modality is absent.
INSTRUCTION = (
    "Carefully analyze the provided information and answer the following "
    "multiple-choice question."
)


@dataclass
class MediaInput:
    path: str
    mime_type: str


@dataclass
class Prompt:
    condition: str
    text: str
    media: list[MediaInput] = field(default_factory=list)
    subtitle_used: bool = False
    valid_letters: list[str] = field(default_factory=list)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def answer_letters(candidates: list[str]) -> list[str]:
    """
    Letters of the answer choices, e.g. ["A", "B", "C", "D"].
    Candidates are stored as "A. ...", "B. ...", in order.
    """
    letters = []

    for i, candidate in enumerate(candidates):
        expected = "ABCDEFGH"[i]
        if not candidate.startswith(f"{expected}. "):
            raise ValueError(
                f"candidate {i} does not start with '{expected}. ': "
                f"{candidate!r}"
            )
        letters.append(expected)

    return letters


def format_letter_list(letters: list[str]) -> str:
    """["A", "B", "C", "D"] -> "A, B, C, or D"."""
    if len(letters) == 1:
        return letters[0]

    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def load_subtitle(video_id: str) -> str:
    """
    Normalized official WorldSense subtitle for a video
    (see scripts/process_subtitles.py).
    """
    path = PROCESSED_SUBTITLE_DIR / f"{video_id}.txt"

    return path.read_text(encoding="utf-8")


def build_prompt(example: dict, condition: str) -> Prompt:
    """
    Build the model input for one frozen example under one condition.

    Only these example fields are used: video_id, question,
    candidates. video_caption is never read.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")

    video_id = example["video_id"]
    letters = answer_letters(example["candidates"])

    sections = [INSTRUCTION]

    subtitle_used = condition in CONDITIONS_WITH_SUBTITLE
    if subtitle_used:
        sections.append("Subtitles:\n" + load_subtitle(video_id))

    sections.append(
        f"Return only the letter of the correct answer: "
        f"{format_letter_list(letters)}."
    )
    sections.append(
        f"Question: {example['question']}\n"
        + "\n".join(example["candidates"])
    )

    media = [
        MediaInput(
            path=str(directory / f"{video_id}{suffix}"),
            mime_type=mime_type,
        )
        for directory, suffix, mime_type in CONDITION_MEDIA[condition]
    ]

    return Prompt(
        condition=condition,
        text="\n\n".join(sections),
        media=media,
        subtitle_used=subtitle_used,
        valid_letters=letters,
    )
