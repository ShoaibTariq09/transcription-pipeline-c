"""Generate small synthetic WAV files so the pipeline can be demoed without real audio.

    python -m tpipe.samples samples/

The audio is just tone bursts (or silence); the MockBackend supplies the "speech".
"""
from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path


def write_wav(path: Path, seconds: float, *, sample_rate: int = 16_000, channels: int = 1, silent: bool = False) -> Path:
    """Write a PCM16 WAV: a quiet 220 Hz tone pulsing on/off, or pure silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    frames = bytearray()
    for i in range(n):
        if silent:
            v = 0
        else:
            on = (i // (sample_rate // 2)) % 2 == 0  # half-second pulses
            v = int(6000 * math.sin(2 * math.pi * 220 * i / sample_rate)) if on else 0
        frames += struct.pack("<h", v) * channels
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    out = ap.parse_args().out_dir
    write_wav(out / "weekly_sync.wav", 18.0)
    write_wav(out / "stereo_44k.wav", 6.0, sample_rate=44_100, channels=2)  # exercises ffmpeg path
    write_wav(out / "silence.wav", 5.0, silent=True)
    (out / "corrupt.wav").write_bytes(b"this is not audio")
    print(f"wrote sample files to {out}/")


if __name__ == "__main__":
    main()
