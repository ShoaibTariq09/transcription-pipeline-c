import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tpipe.audio import AudioError, prepare_audio
from tpipe.backends import BackendResult, MockBackend
from tpipe.models import Segment
from tpipe.pipeline import PipelineConfig, TranscriptionPipeline
from tpipe.samples import write_wav


class CountingBackend(MockBackend):
    """Mock that records how often the (expensive) model is actually invoked."""
    def __init__(self):
        self.calls = 0

    def transcribe(self, wav_path, language=None):
        self.calls += 1
        return super().transcribe(wav_path, language)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.audio = write_wav(self.tmp / "in" / "talk.wav", 12.0)
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, backend=None, **cfg):
        return TranscriptionPipeline(backend or CountingBackend(), self.out, PipelineConfig(**cfg))

    def test_happy_path_writes_all_formats(self):
        res = self.make().run_file(self.audio)
        self.assertEqual(res.status, "ok", res.error)
        out_dir = Path(res.out_dir)
        for name in ("transcript.json", "transcript.srt", "transcript.txt"):
            self.assertTrue((out_dir / name).is_file(), name)
        data = json.loads((out_dir / "transcript.json").read_text())
        self.assertTrue(data["segments"])
        self.assertNotIn("Um,", data["text"])  # fillers stripped
        self.assertIn("insights", data)
        self.assertEqual(set(res.timings_s), {"prepare", "transcribe", "postprocess", "export"})

    def test_second_run_is_skipped_and_does_not_call_the_model(self):
        backend = CountingBackend()
        p = self.make(backend)
        self.assertEqual(p.run_file(self.audio).status, "ok")
        self.assertEqual(p.run_file(self.audio).status, "skipped")
        self.assertEqual(backend.calls, 1)

    def test_force_reprocesses(self):
        backend = CountingBackend()
        self.make(backend).run_file(self.audio)
        self.assertEqual(self.make(backend, force=True).run_file(self.audio).status, "ok")
        self.assertEqual(backend.calls, 2)

    def test_changing_text_affecting_setting_reprocesses(self):
        backend = CountingBackend()
        self.make(backend).run_file(self.audio)
        res = self.make(backend, redact_pii=True).run_file(self.audio)
        self.assertEqual(res.status, "ok")
        data = json.loads((Path(res.out_dir) / "transcript.json").read_text())
        self.assertIn("[EMAIL]", data["text"])
        self.assertNotIn("example.com", json.dumps(data))  # PII must not survive in raw_text either

    def test_missing_output_file_triggers_rerun(self):
        p = self.make()
        res = p.run_file(self.audio)
        (Path(res.out_dir) / "transcript.srt").unlink()
        self.assertEqual(p.run_file(self.audio).status, "ok")

    def test_silent_audio_skips_the_model(self):
        silent = write_wav(self.tmp / "in" / "quiet.wav", 5.0, silent=True)
        backend = CountingBackend()
        res = self.make(backend).run_file(silent)
        self.assertEqual(res.status, "ok")
        self.assertEqual(backend.calls, 0)
        self.assertIn("silent_audio", res.warnings)

    def test_bad_inputs_fail_without_crashing_the_batch(self):
        corrupt = self.tmp / "in" / "corrupt.wav"
        corrupt.write_bytes(b"not audio at all")
        wrong_ext = self.tmp / "in" / "notes.txt"
        wrong_ext.write_text("hi")
        empty = self.tmp / "in" / "empty.wav"
        empty.write_bytes(b"")
        results = self.make().run_batch([corrupt, wrong_ext, empty, self.tmp / "in" / "nope.wav", self.audio])
        self.assertEqual([r.status for r in results], ["failed", "failed", "failed", "failed", "ok"])
        self.assertTrue(all(r.error for r in results[:4]))
        summary = json.loads((self.out / "run_summary.json").read_text())
        self.assertEqual(summary["counts"], {"ok": 1, "skipped": 0, "failed": 4})

    def test_backend_exception_is_captured(self):
        class Boom(MockBackend):
            def transcribe(self, wav_path, language=None):
                raise RuntimeError("model exploded")
        res = self.make(Boom()).run_file(self.audio)
        self.assertEqual(res.status, "failed")
        self.assertIn("model exploded", res.error)

    def test_only_fillers_yields_no_speech_warning(self):
        class Fillers(MockBackend):
            def transcribe(self, wav_path, language=None):
                return BackendResult([Segment(0, 1, "Um."), Segment(1, 2, "uh, hmm")], "en")
        res = self.make(Fillers()).run_file(self.audio)
        self.assertEqual(res.status, "ok")
        self.assertIn("no_speech_detected", res.warnings)


class AudioTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_stereo_44k_is_normalized_to_16k_mono(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = write_wav(d / "s.wav", 3.0, sample_rate=44_100, channels=2)
            prepared = prepare_audio(src, d)
            self.assertNotEqual(prepared.wav_path, src)
            import wave
            with wave.open(str(prepared.wav_path)) as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16_000, 1, 2))
            self.assertAlmostEqual(prepared.duration_s, 3.0, delta=0.1)

    def test_too_short_audio_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src = write_wav(Path(d) / "tiny.wav", 0.1)
            with self.assertRaises(AudioError):
                prepare_audio(src, Path(d), min_duration_s=0.3)


if __name__ == "__main__":
    unittest.main()
