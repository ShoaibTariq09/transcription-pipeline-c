import unittest

from tpipe.models import Segment
from tpipe.postprocess import clean_segments, redact_pii


def clean(text, conf=0.9, **kw):
    out = clean_segments([Segment(0, 1, text, confidence=conf)], **kw)
    return out[0] if out else None


class PostprocessTests(unittest.TestCase):
    def test_removes_fillers_and_stray_commas(self):
        self.assertEqual(clean("Um, good morning everyone.").text, "Good morning everyone.")
        self.assertEqual(clean("We talk about the, uh, migration.").text, "We talk about the migration.")
        self.assertEqual(clean("So uh the topic is X.").text, "So the topic is X.")

    def test_collapses_repeats_created_by_filler_removal(self):
        self.assertEqual(clean("the uh the plan").text, "The plan")
        self.assertEqual(clean("thanks for joining the the sync.").text, "Thanks for joining the sync.")

    def test_sentence_case_and_lone_i(self):
        self.assertEqual(clean("so i think i'll go.").text, "So I think I'll go.")

    def test_drops_segments_that_are_only_fillers(self):
        self.assertIsNone(clean("Um."))
        self.assertIsNone(clean("uh, hmm"))

    def test_low_confidence_flag(self):
        self.assertIn("low_confidence", clean("Hello there", conf=0.3).flags)
        self.assertNotIn("low_confidence", clean("Hello there", conf=0.9).flags)
        self.assertEqual(clean("Hello there", conf=None).flags, [])

    def test_raw_text_kept_unless_redacting(self):
        self.assertEqual(clean("Um, hi there").raw_text, "Um, hi there")
        s = clean("mail me at a.b@example.com", redact=True)
        self.assertIsNone(s.raw_text)
        self.assertEqual(s.text, "Mail me at [EMAIL]")
        self.assertIn("pii_redacted", s.flags)

    def test_redact_pii(self):
        text, n = redact_pii("Call 555-867-5309 or (555) 123 4567, or mail x@y.org")
        self.assertEqual(n, 3)
        self.assertNotIn("867", text)
        self.assertIn("[PHONE]", text)
        self.assertIn("[EMAIL]", text)

    def test_filler_words_inside_real_words_are_untouched(self):
        self.assertEqual(clean("The humming umbrella is here.").text, "The humming umbrella is here.")

    def test_toggles(self):
        self.assertEqual(clean("Um, hi", remove_fillers=False).text, "Um, hi")
        self.assertEqual(clean("the the cat", collapse_repeats=False).text, "The the cat")


if __name__ == "__main__":
    unittest.main()
