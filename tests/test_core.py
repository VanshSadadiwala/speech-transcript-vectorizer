"""Run:  python -m unittest discover -s tests -v   (from the project root)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import AugmentedBoWVectorizer, CollocationExtractor  # noqa: E402
from src.preprocessing import TranscriptPreprocessor  # noqa: E402


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.p = TranscriptPreprocessor()

    def test_fillers_and_case(self):
        self.assertEqual(self.p("UM play uh MUSIC you know"), ["play", "music"])

    def test_contractions(self):
        self.assertIn("want", self.p.clean_text("I wanna go"))

    def test_keywords_protected(self):
        self.assertEqual(self.p("turn off the heater"), ["turn", "off", "heater"])

    def test_noise_tags(self):
        self.assertEqual(self.p("[noise] call (cough) mom"), ["call", "mom"])


class TestModels(unittest.TestCase):
    DOCS = [["credit", "card", "lost"], ["credit", "card", "bill"], ["credit", "card", "limit"],
            ["play", "music"], ["play", "jazz"], ["pay", "bill"]]

    def test_pmi_selects_credit_card(self):
        ce = CollocationExtractor(min_freq=3, pmi_threshold=0).fit(self.DOCS)
        self.assertIn(("credit", "card"), ce.selected_)

    def test_min_freq_filters_rare_pairs(self):
        ce = CollocationExtractor(min_freq=4, pmi_threshold=-10).fit(self.DOCS)
        self.assertEqual(ce.selected_, [])

    def test_matrix_shape_and_values(self):
        v = AugmentedBoWVectorizer(min_freq=3, pmi_threshold=0)
        X = v.fit_transform(self.DOCS)
        self.assertEqual(X.shape, (6, len(v.bow_feature_names_) + len(v.collocation_feature_names_)))
        j = v.feature_names_.index("credit_card")
        self.assertEqual(X[0, j], 1)
        self.assertEqual(X[3, j], 0)

    def test_transform_unseen_docs(self):
        v = AugmentedBoWVectorizer(min_freq=3, pmi_threshold=0).fit(self.DOCS)
        X = v.transform([["credit", "card", "unknownword"]])
        self.assertEqual(X.shape[1], len(v.feature_names_))


if __name__ == "__main__":
    unittest.main()
