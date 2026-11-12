import array
import math
import tempfile
import unittest
import wave
from pathlib import Path

from tpipe.backends import MockBackend
from tpipe.chunking import iter_chunks, plan_chunks
from tpipe.pipeline import PipelineConfig, TranscriptionPipeline

SR = 16_000


def tone_with_gaps(path: Path, seconds: int, gaps: list[tuple[float, float]]) -> Path:
    """Continuous tone, with silence during each (start, end) gap."""
    samples = array.array("h")
    for i in range(seconds * SR):
        t = i / SR
        quiet = any(a <= t < b for a, b in gaps)
        samples.append(0 if quiet else int(8000 * math.sin(2 * math.pi * 220 * t)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())
    return path


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_short_audio_is_a_single_span(self):
        wav = tone_with_gaps(self.tmp / "a.wav", 12, [])
        self.assertEqual(plan_chunks(wav, 12.0, chunk_s=10), [(0.0, 12.0)])  # 12 <= 1.5 * 10

    def test_disabled_when_chunk_s_is_zero(self):
        wav = tone_with_gaps(self.tmp / "a.wav", 30, [])
        self.assertEqual(plan_chunks(wav, 30.0, chunk_s=0), [(0.0, 30.0)])

    def test_cuts_land_inside_the_pauses(self):
        # Targets are ~10 s and ~20 s; pauses sit slightly off-target, so a fixed-interval
        # splitter would cut mid-tone, while the silence-aware one must find them.
        gaps = [(8.5, 9.5), (19.2, 20.2)]
        wav = tone_with_gaps(self.tmp / "a.wav", 30, gaps)
        spans = plan_chunks(wav, 30.0, chunk_s=10)
        self.assertEqual(len(spans), 3)
        cuts = [end for _, end in spans[:-1]]
        for cut, (a, b) in zip(cuts, gaps):
            self.assertTrue(a <= cut <= b, f"cut {cut:.2f}s not inside gap {a}-{b}")

    def test_spans_are_contiguous_and_cover_everything(self):
        wav = tone_with_gaps(self.tmp / "a.wav", 47, [(9.0, 10.0), (19.0, 20.0), (29.0, 30.0), (39.0, 40.0)])
        spans = plan_chunks(wav, 47.0, chunk_s=10)
        self.assertEqual(spans[0][0], 0.0)
        self.assertEqual(spans[-1][1], 47.0)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertEqual(prev_end, next_start)
        self.assertTrue(all(end - start > 4 for start, end in spans))  # no sliver chunks

    def test_iter_chunks_writes_valid_wavs_and_cleans_up(self):
        wav = tone_with_gaps(self.tmp / "a.wav", 30, [(9.5, 10.5)])
        spans = [(0.0, 10.0), (10.0, 30.0)]
        seen = []
        for chunk in iter_chunks(wav, spans, self.tmp):
            with wave.open(str(chunk.path)) as w:
                seen.append(round(w.getnframes() / w.getframerate(), 2))
            self.assertTrue(chunk.path.exists())
        self.assertEqual(seen, [10.0, 20.0])
        self.assertEqual(list(self.tmp.glob("chunk_*.wav")), [])  # temp chunks removed

    def test_pipeline_stitches_timestamps_across_chunks(self):
        wav = tone_with_gaps(self.tmp / "long.wav", 40, [(9.5, 10.5), (19.5, 20.5), (29.5, 30.5)])
        pipeline = TranscriptionPipeline(MockBackend(), self.tmp / "out", PipelineConfig(chunk_s=10))
        res = pipeline.run_file(wav)
        self.assertEqual(res.status, "ok", res.error)

        import json
        data = json.loads((Path(res.out_dir) / "transcript.json").read_text())
        self.assertEqual(data["meta"]["chunks"], 4)
        segs = data["segments"]
        starts = [s["start"] for s in segs]
        self.assertEqual(starts, sorted(starts))                     # monotonic across chunk seams
        self.assertGreater(max(starts), 30)                          # last chunk shifted to file time
        self.assertTrue(all(0 <= s["start"] <= s["end"] <= 40.01 for s in segs))
        self.assertEqual(data["language"], "en")

    def test_language_detected_once_then_reused(self):
        class Spy(MockBackend):
            seen = []
            def transcribe(self, wav_path, language=None):
                self.seen.append(language)
                r = super().transcribe(wav_path, language)
                r.language = "de"  # pretend auto-detection said German
                return r
        wav = tone_with_gaps(self.tmp / "long.wav", 40, [(9.5, 10.5), (19.5, 20.5), (29.5, 30.5)])
        TranscriptionPipeline(Spy(), self.tmp / "out", PipelineConfig(chunk_s=10)).run_file(wav)
        self.assertEqual(Spy.seen, [None, "de", "de", "de"])


if __name__ == "__main__":
    unittest.main()
