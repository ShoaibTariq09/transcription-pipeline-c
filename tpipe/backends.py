"""Stage 2: speech-to-text backends behind one small interface.

The pipeline only depends on `STTBackend`. Adding another engine (a hosted API,
whisper.cpp, Vosk, ...) means writing one class that returns `BackendResult`;
no other stage changes.

Two implementations ship here:
* FasterWhisperBackend: real, local, open-source (CTranslate2 build of Whisper).
* MockBackend: deterministic fake output, so the whole pipeline can be run and
  tested with no model download, no GPU and no network.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import Segment


@dataclass
class BackendResult:
    segments: list[Segment]
    language: str | None


class STTBackend(Protocol):
    name: str

    def transcribe(self, wav_path: Path, language: str | None = None) -> BackendResult:
        """Transcribe a normalized 16 kHz mono WAV into timestamped segments."""
        ...


class FasterWhisperBackend:
    """Local Whisper via faster-whisper (pip install faster-whisper)."""

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "int8"):
        # Lazy import: the package is heavy and only needed for this backend.
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "faster-whisper is not installed. Run `pip install faster-whisper` "
                "or use --backend mock."
            ) from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.name = f"faster-whisper:{model_size}"

    def transcribe(self, wav_path: Path, language: str | None = None) -> BackendResult:  # pragma: no cover
        segments_iter, info = self._model.transcribe(
            str(wav_path),
            language=language,  # None means auto-detect
            beam_size=5,
            vad_filter=True,  # skip non-speech; also reduces hallucination
            condition_on_previous_text=False,  # stops one bad window from poisoning the rest
        )
        out: list[Segment] = []
        for s in segments_iter:  # generator: decoding happens lazily, memory stays flat
            # Whisper's own heuristic for "this window was probably not speech".
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            text = s.text.strip()
            if not text:
                continue
            out.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=text,
                    confidence=round(math.exp(s.avg_logprob), 4),  # mean token prob, 0..1
                )
            )
        return BackendResult(segments=out, language=info.language)


class MockBackend:
    """Deterministic fake STT. Output is scaled to the audio's duration.

    The script deliberately contains filler words, a stutter, an e-mail address,
    a phone number, a mis-cased "i", action items and one low-confidence
    segment, so the downstream stages have something real to chew on.
    """

    name = "mock"

    _SCRIPT: list[tuple[str, float]] = [
        ("Um, good morning everyone, thanks for joining the the weekly sync.", 0.94),
        ("So uh the main topic today is the Q3 launch.", 0.91),
        ("Please email the launch report to priya.sharma@example.com by Friday.", 0.88),
        ("We also need to talk about the, uh, kubernetes migration.", 0.38),
        ("If anything breaks you can reach me on 555-867-5309.", 0.9),
        ("So i think the demo went well and the customers liked it.", 0.82),
        ("We need to update the roadmap before next week.", 0.93),
        ("Okay, that's everything, thanks all.", 0.96),
    ]

    def transcribe(self, wav_path: Path, language: str | None = None) -> BackendResult:
        with wave.open(str(wav_path), "rb") as w:
            duration = w.getnframes() / w.getframerate()
        n = max(1, min(len(self._SCRIPT), int(duration // 2)))
        step = duration / n
        segments = []
        for i in range(n):
            text, conf = self._SCRIPT[i]
            start = i * step
            end = max(start, (i + 1) * step - 0.05)
            segments.append(
                Segment(start=round(start, 3), end=round(end, 3), text=text, confidence=conf)
            )
        return BackendResult(segments=segments, language=language or "en")


def build_backend(name: str, *, model_size: str = "base") -> STTBackend:
    if name == "mock":
        return MockBackend()
    if name == "whisper":
        return FasterWhisperBackend(model_size=model_size)
    raise ValueError(f"unknown backend '{name}' (choose 'mock' or 'whisper')")
