"""Stage 1: validate the input and normalize it to 16 kHz mono 16-bit WAV.

Speech models expect one canonical format. Doing the conversion once, up front,
means every backend sees identical input and the rest of the pipeline never has
to care what the user uploaded (mp3, m4a, stereo 44.1 kHz, ...).

Design notes:
* Files that are already 16 kHz mono PCM16 WAV skip ffmpeg entirely, so the
  happy path and the tests need no external binaries.
* We hash the original file. The hash drives idempotency (skip work we've
  already done) and gives each output folder a stable name.
* A cheap loudness check lets us skip silent files. Whisper-family models are
  known to hallucinate text on silence, so we gate on this before decoding.
"""
from __future__ import annotations

import array
import hashlib
import math
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

TARGET_SR = 16_000
SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4"}
SILENCE_FLOOR_DBFS = -120.0


class AudioError(Exception):
    """Input audio is missing, unsupported, corrupt, or out of allowed bounds."""


@dataclass
class PreparedAudio:
    source: Path
    wav_path: Path  # normalized 16 kHz mono PCM16 WAV
    sha256: str  # hash of the ORIGINAL file
    duration_s: float
    rms_dbfs: float


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def _wav_info(path: Path) -> tuple[int, int, int, int]:
    """Return (sample_rate, channels, sample_width_bytes, n_frames)."""
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()


def _is_target_format(path: Path) -> bool:
    try:
        sr, ch, sw, _ = _wav_info(path)
    except (wave.Error, EOFError):
        return False  # not PCM WAV (e.g. float WAV, or corrupt): let ffmpeg decide
    return sr == TARGET_SR and ch == 1 and sw == 2


def _convert_with_ffmpeg(src: Path, dst: Path, timeout_s: int = 900) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioError(
            f"{src.name} needs conversion to 16 kHz mono WAV but ffmpeg is not "
            "installed. Install ffmpeg or supply a 16 kHz mono PCM16 WAV."
        )
    cmd = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src), "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-sample_fmt", "s16", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffmpeg timed out converting {src.name}") from exc
    if proc.returncode != 0:
        # ffmpeg prints several lines; the last non-empty one is the actual reason.
        lines = [ln.strip() for ln in proc.stderr.splitlines() if ln.strip()]
        reason = lines[-1] if lines else f"exit code {proc.returncode}"
        raise AudioError(f"ffmpeg could not decode {src.name}: {reason[:200]}")


def rms_dbfs(wav_path: Path, stride: int = 16) -> float:
    """Approximate RMS loudness in dBFS of a 16-bit mono WAV.

    Subsamples every `stride`-th sample: plenty accurate for a silence check and
    keeps pure-Python cost low on long files.
    """
    total = 0
    count = 0
    with wave.open(str(wav_path), "rb") as w:
        while True:
            frames = w.readframes(TARGET_SR * 10)
            if not frames:
                break
            samples = array.array("h")
            samples.frombytes(frames)
            if sys.byteorder == "big":
                samples.byteswap()
            sub = samples[::stride]
            total += sum(s * s for s in sub)
            count += len(sub)
    if count == 0 or total == 0:
        return SILENCE_FLOOR_DBFS
    rms = math.sqrt(total / count)
    return max(SILENCE_FLOOR_DBFS, 20 * math.log10(rms / 32768))


def prepare_audio(
    path: str | Path,
    workdir: Path,
    *,
    min_duration_s: float = 0.3,
    max_duration_s: float = 4 * 3600,
) -> PreparedAudio:
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"not a file: {path}")
    if path.stat().st_size == 0:
        raise AudioError(f"empty file: {path.name}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise AudioError(
            f"unsupported extension '{path.suffix}' (supported: {', '.join(sorted(SUPPORTED_EXTS))})"
        )

    digest = sha256_file(path)

    if path.suffix.lower() == ".wav" and _is_target_format(path):
        wav_path = path
    else:
        wav_path = workdir / f"{digest[:12]}.wav"
        _convert_with_ffmpeg(path, wav_path)

    try:
        sr, _, _, n_frames = _wav_info(wav_path)
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"could not read audio from {path.name}: {exc}") from exc

    duration = n_frames / sr
    if duration < min_duration_s:
        raise AudioError(f"audio too short ({duration:.2f}s < {min_duration_s}s)")
    if duration > max_duration_s:
        raise AudioError(f"audio too long ({duration:.0f}s > {max_duration_s:.0f}s)")

    return PreparedAudio(
        source=path,
        wav_path=wav_path,
        sha256=digest,
        duration_s=duration,
        rms_dbfs=rms_dbfs(wav_path),
    )
