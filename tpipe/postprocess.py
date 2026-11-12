"""Stage 3: turn raw STT output into text that is safe and pleasant to consume.

Every step is a small pure function, so each can be unit-tested and toggled
independently. These are heuristics, not NLP magic: they are cheap, predictable
and easy to explain, and the original text is kept in `raw_text` (unless PII
redaction is on, in which case keeping it would defeat the purpose).
"""
from __future__ import annotations

import re
from dataclasses import replace

from .models import Segment

# ", uh," / "um" / "hmm" along with any commas hugging them.
_FILLER_RE = re.compile(r",?\s*\b(?:um+|uh+|erm|hmm+)\b\s*,?", re.IGNORECASE)
# "the the" -> "the". Case-insensitive, handles apostrophes ("don't don't").
_REPEAT_RE = re.compile(r"\b([\w']+)(?:\s+\1\b)+", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_LONE_I_RE = re.compile(r"\bi\b(?!\.)")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Loose North-American style 10-digit numbers; international formats vary too much
# for a regex, so treat this as best-effort and document the limitation.
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def _sentence_case(text: str) -> str:
    text = _LONE_I_RE.sub("I", text)
    return text[:1].upper() + text[1:] if text else text


def redact_pii(text: str) -> tuple[str, int]:
    """Replace e-mail addresses and phone numbers. Returns (text, n_replacements)."""
    text, n_email = _EMAIL_RE.subn("[EMAIL]", text)
    text, n_phone = _PHONE_RE.subn("[PHONE]", text)
    return text, n_email + n_phone


def clean_segments(
    segments: list[Segment],
    *,
    remove_fillers: bool = True,
    collapse_repeats: bool = True,
    redact: bool = False,
    low_conf_threshold: float = 0.5,
) -> list[Segment]:
    cleaned: list[Segment] = []
    for seg in segments:
        text = seg.text
        flags = list(seg.flags)

        # Order matters: removing "uh" from "the uh the" creates the repeat we then collapse.
        if remove_fillers:
            text = _FILLER_RE.sub(" ", text)
        if collapse_repeats:
            text = _REPEAT_RE.sub(r"\1", text)
        text = _sentence_case(_normalize_whitespace(text))

        if redact:
            text, n = redact_pii(text)
            if n:
                flags.append("pii_redacted")

        # A segment that was only fillers ("Um.") has nothing left worth keeping.
        if not re.search(r"\w", text):
            continue

        if seg.confidence is not None and seg.confidence < low_conf_threshold:
            flags.append("low_confidence")

        cleaned.append(replace(seg, text=text, raw_text=None if redact else seg.text, flags=flags))
    return cleaned
