"""Command line entry point.

    python -m tpipe.cli samples/ -o out/ --backend mock
    python -m tpipe.cli talk.mp3 -o out/ --backend whisper --model small --redact-pii
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .audio import SUPPORTED_EXTS
from .backends import build_backend
from .pipeline import PipelineConfig, TranscriptionPipeline


def collect_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS)
        else:
            files.append(p)  # non-existent paths are reported per-file, not silently dropped
    return list(dict.fromkeys(files))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="tpipe", description="Audio -> cleaned transcript -> JSON/SRT/TXT")
    ap.add_argument("inputs", nargs="+", help="audio files and/or directories")
    ap.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    ap.add_argument("--backend", choices=["mock", "whisper"], default="mock")
    ap.add_argument("--model", default="base", help="whisper model size: tiny/base/small/medium/large-v3")
    ap.add_argument("--language", default=None, help="ISO code such as 'en'; omit to auto-detect")
    ap.add_argument("--chunk-seconds", type=float, default=600.0,
                    help="split audio longer than 1.5x this at pauses (default 600; 0 disables)")
    ap.add_argument("--formats", nargs="+", default=["json", "srt", "txt"], choices=["json", "srt", "txt"])
    ap.add_argument("--redact-pii", action="store_true", help="mask e-mails and phone numbers")
    ap.add_argument("--keep-fillers", action="store_true", help="do not strip um/uh")
    ap.add_argument("--force", action="store_true", help="reprocess even if output is up to date")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    files = collect_inputs(args.inputs)
    if not files:
        print("no input files found", file=sys.stderr)
        return 2

    config = PipelineConfig(
        language=args.language,
        chunk_s=args.chunk_seconds,
        remove_fillers=not args.keep_fillers,
        redact_pii=args.redact_pii,
        formats=tuple(args.formats),
        force=args.force,
    )
    pipeline = TranscriptionPipeline(build_backend(args.backend, model_size=args.model), args.out, config)
    results = pipeline.run_batch(files)

    print()
    for r in results:
        detail = r.error or (", ".join(r.warnings) if r.warnings else "")
        print(f"{r.status:<8} {Path(r.source).name:<24} {detail}")
    failed = sum(r.status == "failed" for r in results)
    print(f"\n{len(results) - failed}/{len(results)} succeeded -> {Path(args.out) / 'run_summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
