"""Orchestrator: prepare -> transcribe -> postprocess -> export.

Engineering decisions worth knowing about:

* Failure isolation: one bad file never kills a batch. Each file yields a
  FileResult (ok / skipped / failed) and the batch carries on.
* Idempotency: a fingerprint of (audio hash, backend, settings that change the
  text) is stored inside transcript.json. Re-running on unchanged input is a
  no-op; changing a setting (say, turning on PII redaction) reprocesses.
* Silence gating: near-silent audio never reaches the model, because Whisper-style
  models invent text on silence.
* Observability: per-stage timings are recorded and a run_summary.json is written.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import __version__
from .audio import prepare_audio
from .backends import BackendResult, STTBackend
from .chunking import iter_chunks, plan_chunks
from .downstream import FORMAT_FILENAMES, atomic_write_text, build_insights, write_outputs
from .models import Transcript
from .postprocess import clean_segments

log = logging.getLogger("tpipe")


@dataclass
class PipelineConfig:
    language: str | None = None  # None = let the backend auto-detect
    min_duration_s: float = 0.3
    max_duration_s: float = 4 * 3600
    silence_dbfs: float = -55.0  # quieter than this is treated as "no audio"
    chunk_s: float = 600.0  # files longer than 1.5x this are split at pauses; <=0 disables
    remove_fillers: bool = True
    collapse_repeats: bool = True
    redact_pii: bool = False
    low_conf_threshold: float = 0.5
    formats: tuple[str, ...] = ("json", "srt", "txt")
    force: bool = False  # ignore the "already done" check

    def text_affecting(self) -> dict:
        """Only settings that change the transcript's content belong in the fingerprint."""
        return {
            "language": self.language,
            "silence_dbfs": self.silence_dbfs,
            "chunk_s": self.chunk_s,
            "remove_fillers": self.remove_fillers,
            "collapse_repeats": self.collapse_repeats,
            "redact_pii": self.redact_pii,
            "low_conf_threshold": self.low_conf_threshold,
        }


@dataclass
class FileResult:
    source: str
    status: str = "failed"  # ok | skipped | failed
    out_dir: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)


class TranscriptionPipeline:
    def __init__(self, backend: STTBackend, out_dir: str | Path, config: PipelineConfig | None = None):
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.config = config or PipelineConfig()
        # JSON is always written: it holds the fingerprint used for idempotency.
        self._formats = tuple(dict.fromkeys((*self.config.formats, "json")))

    # ----- public API -----

    def run_file(self, path: str | Path) -> FileResult:
        path = Path(path)
        res = FileResult(source=str(path))
        try:
            with tempfile.TemporaryDirectory(prefix="tpipe-") as tmp:
                with self._stage(res, "prepare"):
                    audio = prepare_audio(
                        path,
                        Path(tmp),
                        min_duration_s=self.config.min_duration_s,
                        max_duration_s=self.config.max_duration_s,
                    )

                fingerprint = self._fingerprint(audio.sha256)
                out_dir = self.out_dir / f"{path.stem}-{audio.sha256[:8]}"
                res.out_dir = str(out_dir)

                if not self.config.force and self._is_up_to_date(out_dir, fingerprint):
                    log.info("skip %s (already processed)", path.name)
                    res.status = "skipped"
                    return res

                transcript = Transcript(
                    source=str(path),
                    audio_sha256=audio.sha256,
                    duration_s=round(audio.duration_s, 3),
                    language=self.config.language,
                    backend=self.backend.name,
                )

                if audio.rms_dbfs < self.config.silence_dbfs:
                    transcript.warnings.append("silent_audio")
                else:
                    with self._stage(res, "transcribe"):
                        result, n_chunks = self._transcribe(audio.wav_path, audio.duration_s, Path(tmp))
                    transcript.meta["chunks"] = n_chunks
                    transcript.language = result.language or transcript.language
                    with self._stage(res, "postprocess"):
                        transcript.segments = clean_segments(
                            result.segments,
                            remove_fillers=self.config.remove_fillers,
                            collapse_repeats=self.config.collapse_repeats,
                            redact=self.config.redact_pii,
                            low_conf_threshold=self.config.low_conf_threshold,
                        )
                    if not transcript.segments:
                        transcript.warnings.append("no_speech_detected")

                transcript.meta |= {
                    "fingerprint": fingerprint,
                    "pipeline_version": __version__,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "config": self.config.text_affecting(),
                }

                with self._stage(res, "export"):
                    write_outputs(transcript, build_insights(transcript), out_dir, self._formats)

                res.warnings = list(transcript.warnings)
                res.status = "ok"
                log.info("done %s in %.2fs", path.name, sum(res.timings_s.values()))
        except Exception as exc:  # noqa: BLE001 - isolate per-file failures on purpose
            res.status = "failed"
            res.error = f"{type(exc).__name__}: {exc}"
            log.error("failed %s: %s", path.name, res.error)
            log.debug("traceback for %s", path.name, exc_info=True)
        return res

    def run_batch(self, paths: list[str | Path]) -> list[FileResult]:
        results = [self.run_file(p) for p in paths]
        self._write_summary(results)
        return results

    # ----- internals -----

    def _transcribe(self, wav_path: Path, duration_s: float, workdir: Path) -> tuple[BackendResult, int]:
        """Transcribe whole, or chunk by chunk with timestamps shifted back to file time."""
        spans = plan_chunks(wav_path, duration_s, chunk_s=self.config.chunk_s)
        if len(spans) == 1:
            return self.backend.transcribe(wav_path, self.config.language), 1

        segments = []
        language = self.config.language
        for chunk in iter_chunks(wav_path, spans, workdir):
            log.info("chunk %d/%d  [%.0fs - %.0fs]", chunk.index + 1, len(spans), chunk.start_s, chunk.end_s)
            result = self.backend.transcribe(chunk.path, language)
            # Detect the language once on the first chunk, then reuse it: faster, and it
            # stops the detector flip-flopping on a quiet or music-heavy chunk later on.
            language = language or result.language
            segments += [
                replace(s, start=round(s.start + chunk.start_s, 3), end=round(s.end + chunk.start_s, 3))
                for s in result.segments
            ]
        return BackendResult(segments=segments, language=language), len(spans)

    @contextmanager
    def _stage(self, res: FileResult, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            res.timings_s[name] = round(time.perf_counter() - t0, 4)

    def _fingerprint(self, audio_sha256: str) -> str:
        blob = json.dumps(
            {"audio": audio_sha256, "backend": self.backend.name, "cfg": self.config.text_affecting()},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _is_up_to_date(self, out_dir: Path, fingerprint: str) -> bool:
        json_path = out_dir / FORMAT_FILENAMES["json"]
        if not json_path.is_file():
            return False
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False  # corrupt output: redo it
        if existing.get("meta", {}).get("fingerprint") != fingerprint:
            return False
        return all((out_dir / FORMAT_FILENAMES[f]).is_file() for f in self._formats)

    def _write_summary(self, results: list[FileResult]) -> None:
        counts = {s: sum(r.status == s for r in results) for s in ("ok", "skipped", "failed")}
        summary = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": self.backend.name,
            "counts": counts,
            "files": [asdict(r) for r in results],
        }
        atomic_write_text(self.out_dir / "run_summary.json", json.dumps(summary, indent=2) + "\n")
