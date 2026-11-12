"""Data models shared by every pipeline stage.

The Transcript is the pipeline's canonical output. Every exporter (JSON, SRT, TXT)
and every analysis step reads from it, so backends can be swapped without
touching anything downstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class Segment:
    start: float  # seconds from start of audio
    end: float
    text: str
    confidence: float | None = None  # 0..1 when the backend can provide it
    raw_text: str | None = None  # text before cleaning, kept for auditability
    flags: list[str] = field(default_factory=list)  # e.g. ["low_confidence"]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    source: str
    audio_sha256: str
    duration_s: float
    language: str | None
    backend: str
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # e.g. ["silent_audio"]
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["text"] = self.text
        return d
