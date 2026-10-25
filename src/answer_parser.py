import re
from dataclasses import dataclass


# parse_status values
EXACT = "exact"            # response is exactly one valid letter
NORMALIZED = "normalized"  # one valid letter in a recognized answer format
AMBIGUOUS = "ambiguous"    # conflicting answer letters; no guess made
INVALID = "invalid"        # no recognizable answer letter


@dataclass
class ParsedAnswer:
    prediction: str | None
    status: str


# Whole-response formats, e.g. "A", "A.", "(A)", "[A]", "Answer: A",
# "The answer is (B).", "Option C", "A. One." (letter + its choice text).
# A bare letter followed directly by a word ("A man ...") is NOT matched,
# so the English article is never read as an answer.
_WHOLE_RESPONSE_RE = re.compile(
    r"""^
    (?:(?:the\s+)?(?:correct\s+|final\s+)?(?:answer|option|choice)
       (?:\s+is)?\s*[:\-]?\s*)?
    [\(\[]?(?P<letter>[A-Z])[\)\]]?
    (?:\s*$|[.:)]\s*.*$)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Explicit answer statements inside longer text,
# e.g. "... so the answer is B." or "Answer: (C)".
_ANSWER_STATEMENT_RE = re.compile(
    r"(?:answer|option|choice)(?:\s+is)?\s*[:\-]?\s*[\(\[]?([A-Z])\b",
    re.IGNORECASE,
)

# Strip markdown emphasis / code markers around a response.
_MARKUP_RE = re.compile(r"[*_`]+")

# A standalone uppercase letter, e.g. the "B" in "A or B".
_LETTER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])")


def _letter_tokens(text: str, valid_letters: list[str]) -> set[str]:
    """
    Valid answer letters appearing as standalone tokens. An "A"
    followed by a lowercase word ("A man ...") is the English
    article, not an answer letter, unless that word is a connector
    as in "A or B".
    """
    letters = set()

    for match in _LETTER_TOKEN_RE.finditer(text):
        letter = match.group(1)
        if letter not in valid_letters:
            continue

        next_word = re.match(r"\s+([a-z]+)", text[match.end():])
        if (
            letter == "A"
            and next_word
            and next_word.group(1) not in {"or", "and", "nor", "vs"}
        ):
            continue
        letters.add(letter)

    return letters


def parse_answer(raw: str | None, valid_letters: list[str]) -> ParsedAnswer:
    """
    Parse a model response into one of valid_letters.
    Never guesses between conflicting letters.
    """
    if raw is None:
        return ParsedAnswer(None, INVALID)

    text = raw.strip()

    if text in valid_letters:
        return ParsedAnswer(text, EXACT)

    cleaned = _MARKUP_RE.sub("", text).strip()

    if not cleaned:
        return ParsedAnswer(None, INVALID)

    # The answer letter itself must be uppercase; "answer"/"option"
    # keywords may be any case.
    match = _WHOLE_RESPONSE_RE.match(cleaned)
    if match and match.group("letter").isupper():
        letter = match.group("letter")
        if letter not in valid_letters:
            return ParsedAnswer(None, INVALID)

        # Text after the letter (e.g. its choice text) must not
        # name a different answer letter, as in "A. B." or "(A) or (C)".
        rest = cleaned[match.end("letter"):]
        if _letter_tokens(rest, valid_letters) - {letter}:
            return ParsedAnswer(None, AMBIGUOUS)

        return ParsedAnswer(letter, NORMALIZED)

    stated = {
        letter
        for letter in _ANSWER_STATEMENT_RE.findall(cleaned)
        if letter.isupper()
    }

    if len(stated) > 1:
        return ParsedAnswer(None, AMBIGUOUS)

    if len(stated) == 1:
        letter = stated.pop()
        if letter in valid_letters:
            return ParsedAnswer(letter, NORMALIZED)
        return ParsedAnswer(None, INVALID)

    # Several standalone letters ("A or B", "A, B") with no explicit
    # answer statement: do not guess.
    if len(_letter_tokens(cleaned, valid_letters)) > 1:
        return ParsedAnswer(None, AMBIGUOUS)

    return ParsedAnswer(None, INVALID)
