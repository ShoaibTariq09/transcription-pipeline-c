"""Long-audio handling: split at pauses, transcribe chunk by chunk, stitch timestamps.

Why chunk at all? Without it a 3-hour recording is one giant job: memory and
runtime grow with length, there is no progress signal, and a failure near the
end throws away everything. With it, work happens in bounded pieces.

Why cut at the quietest moment near each boundary instead of every N seconds?
A hard cut lands mid-word about as often as not, and that word is usually
mistranscribed on both sides. Searching +/- a few seconds around the target for
the lowest-energy 100 ms window puts the cut in a pause, so no overlap-and-dedupe
logic is needed. Chunks are written to disk one at a time and deleted straight
after use, so peak extra disk is one chunk (10 minutes of 16 kHz mono ~ 19 MB).
"""
from __future__ import annotations

import array
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

FRAME_S = 0.1  # energy-measurement window for finding a pause


@dataclass
class Chunk:
    index: int
    start_s: float
    end_s: float
    path: Path


def _quietest_point(w: wave.Wave_read, center_s: float, search_s: float) -> float:
    """Time (s) of the lowest-energy FRAME_S window within +/- search_s of center_s."""
    sr = w.getframerate()
    start_f = max(0, int((center_s - search_s) * sr))
    end_f = min(w.getnframes(), int((center_s + search_s) * sr))
    w.setpos(start_f)
    samples = array.array("h")
    samples.frombytes(w.readframes(end_f - start_f))
    if sys.byteorder == "big":
        samples.byteswap()

    frame_n = max(1, int(FRAME_S * sr))
    center_f = int(center_s * sr) - start_f
    best_key, best_i = None, 0
    for i in range(0, max(1, len(samples) - frame_n + 1), frame_n):
        energy = sum(s * s for s in samples[i : i + frame_n : 4])  # stride 4: plenty for energy
        # Tie-break toward the target so a long silent stretch cuts near the ideal point.
        key = (energy, abs(i + frame_n // 2 - center_f))
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    return (start_f + best_i + frame_n // 2) / sr


def plan_chunks(wav_path: Path, duration_s: float, *, chunk_s: float) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...] covering the file. One span if it is short enough."""
    if chunk_s <= 0 or duration_s <= chunk_s * 1.5:
        return [(0.0, duration_s)]

    search_s = min(15.0, chunk_s / 4)  # never wander far enough to overlap the neighbour
    cuts: list[float] = []
    target = chunk_s
    with wave.open(str(wav_path), "rb") as w:
        # Stop once what's left is under half a chunk: it just rides along with the last chunk.
        while duration_s - target > chunk_s * 0.5:
            cut = _quietest_point(w, target, search_s)
            cuts.append(cut)
            target = cut + chunk_s

    bounds = [0.0, *cuts, duration_s]
    return list(zip(bounds[:-1], bounds[1:]))


def iter_chunks(wav_path: Path, spans: list[tuple[float, float]], workdir: Path) -> Iterator[Chunk]:
    """Yield each span as its own WAV file, deleting it once the consumer moves on."""
    with wave.open(str(wav_path), "rb") as src:
        sr = src.getframerate()
        params = (src.getnchannels(), src.getsampwidth(), sr)
        for i, (start_s, end_s) in enumerate(spans):
            start_f = int(round(start_s * sr))
            end_f = min(src.getnframes(), int(round(end_s * sr)))
            src.setpos(start_f)
            path = workdir / f"chunk_{i:04d}.wav"
            with wave.open(str(path), "wb") as dst:
                dst.setnchannels(params[0])
                dst.setsampwidth(params[1])
                dst.setframerate(params[2])
                dst.writeframes(src.readframes(end_f - start_f))
            try:
                yield Chunk(index=i, start_s=start_s, end_s=end_s, path=path)
            finally:
                path.unlink(missing_ok=True)
