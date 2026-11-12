import json
import tempfile
import unittest
from pathlib import Path

from tpipe.downstream import (
    extract_keywords, find_action_items, format_timestamp, to_srt, to_txt, write_outputs, build_insights,
)
from tpipe.models import Segment, Transcript


def make_transcript(*segs):
    return Transcript(source="x.wav", audio_sha256="abc", duration_s=10.0, language="en",
                      backend="mock", segments=list(segs))


class DownstreamTests(unittest.TestCase):
    def test_timestamp_format(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(3725.5), "01:02:05,500")
        self.assertEqual(format_timestamp(1.9996), "00:00:02,000")  # rounds, no "1,1000"

    def test_srt_structure(self):
        t = make_transcript(Segment(0, 1.5, "Hello."), Segment(2, 3, "World."))
        self.assertEqual(
            to_srt(t),
            "1\n00:00:00,000 --> 00:00:01,500\nHello.\n\n2\n00:00:02,000 --> 00:00:03,000\nWorld.\n",
        )

    def test_txt_empty_transcript_is_empty(self):
        self.assertEqual(to_txt(make_transcript()), "")

    def test_action_items_with_deadlines(self):
        segs = [
            Segment(1, 2, "Please email the report by Friday."),
            Segment(3, 4, "We need to update the roadmap before next week."),
            Segment(5, 6, "The weather was nice."),
        ]
        items = find_action_items(segs)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["deadline_hint"], "Friday")
        self.assertEqual(items[1]["deadline_hint"], "next week")

    def test_keywords_skip_stopwords(self):
        kws = extract_keywords("The roadmap and the roadmap review. A review of the launch.")
        self.assertEqual(kws[0], {"term": "roadmap", "count": 2})
        self.assertNotIn("the", [k["term"] for k in kws])

    def test_write_outputs_always_valid_json(self):
        t = make_transcript(Segment(0, 1, "Hello.", confidence=0.9))
        with tempfile.TemporaryDirectory() as d:
            written = write_outputs(t, build_insights(t), Path(d), ("txt", "json"))
            data = json.loads(written["json"].read_text())
            self.assertEqual(data["text"], "Hello.")
            self.assertEqual(data["schema_version"], "1.0")
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["transcript.json", "transcript.txt"])


if __name__ == "__main__":
    unittest.main()
