"""Stage 4: downstream products built from the cleaned Transcript.

* JSON: canonical, machine-readable output (transcript + insights), for APIs,
  search indexing or a data warehouse.
* SRT:  subtitles / captions.
* TXT:  human-readable, one segment per line.
* Insights: cheap rule-based extras (stats, keywords, action items) to show how
  the structured transcript feeds further processing. An LLM could replace the
  rules later without changing the interface.

All files are written atomically (temp file + os.replace) so a crash mid-write
can never leave a half-written transcript that looks valid.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Segment, Transcript

_STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been but by can could did do does
    for from get had has have he her here him his how i if in into is it its just like me
    more most my no not of on one or our out she so some than that the their them then there
    these they this to up us was we were what when which who will with would you your
    need talk thanks everyone going also anything everything""".split()
)

_ACTION_RE = re.compile(
    r"\b(i'll|i will|we'll|we will|we need to|we should|let's|please|can you|could you|"
    r"action item|follow up|make sure)\b",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\b(?:by|before|until|on)\s+("
    r"(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)"
    r"|tomorrow|(?:the\s+)?end of (?:the\s+)?(?:day|week|month)|eod|eow)\b",
    re.IGNORECASE,
)


# ---------- insights ----------

def speech_stats(t: Transcript) -> dict[str, Any]:
    words = sum(len(s.text.split()) for s in t.segments)
    speech_s = sum(s.duration for s in t.segments)
    return {
        "word_count": words,
        "speech_s": round(speech_s, 2),
        "speech_ratio": round(speech_s / t.duration_s, 3) if t.duration_s else 0.0,
        "words_per_minute": round(words / (speech_s / 60), 1) if speech_s else 0.0,
    }


def extract_keywords(text: str, top_n: int = 8) -> list[dict[str, Any]]:
    words = re.findall(r"[a-z][a-z'-]{2,}", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return [{"term": w, "count": c} for w, c in counts.most_common(top_n)]


def find_action_items(segments: list[Segment]) -> list[dict[str, Any]]:
    items = []
    for s in segments:
        if _ACTION_RE.search(s.text):
            m = _DEADLINE_RE.search(s.text)
            items.append(
                {
                    "text": s.text,
                    "at_s": round(s.start, 2),
                    "deadline_hint": m.group(1) if m else None,
                }
            )
    return items


def build_insights(t: Transcript) -> dict[str, Any]:
    return {
        "stats": speech_stats(t),
        "keywords": extract_keywords(t.text),
        "action_items": find_action_items(t.segments),
        "low_confidence_segments": sum("low_confidence" in s.flags for s in t.segments),
    }


# ---------- formatters ----------

def format_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(t: Transcript) -> str:
    blocks = [
        f"{i}\n{format_timestamp(s.start)} --> {format_timestamp(s.end)}\n{s.text}\n"
        for i, s in enumerate(t.segments, 1)
    ]
    return "\n".join(blocks)


def to_txt(t: Transcript) -> str:
    return "\n".join(s.text for s in t.segments) + ("\n" if t.segments else "")


def to_json(t: Transcript, insights: dict[str, Any]) -> str:
    payload = t.to_dict()
    payload["insights"] = insights
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


# ---------- writing ----------

FORMAT_FILENAMES = {"json": "transcript.json", "srt": "transcript.srt", "txt": "transcript.txt"}


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_outputs(
    t: Transcript, insights: dict[str, Any], out_dir: Path, formats: tuple[str, ...]
) -> dict[str, Path]:
    renderers = {
        "json": lambda: to_json(t, insights),
        "srt": lambda: to_srt(t),
        "txt": lambda: to_txt(t),
    }
    written: dict[str, Path] = {}
    # JSON last: it carries the fingerprint the pipeline uses to decide "already done",
    # so its presence signals every other requested file was written first.
    for fmt in sorted(formats, key=lambda f: f == "json"):
        if fmt not in renderers:
            raise ValueError(f"unknown format '{fmt}'")
        path = out_dir / FORMAT_FILENAMES[fmt]
        atomic_write_text(path, renderers[fmt]())
        written[fmt] = path
    return written
