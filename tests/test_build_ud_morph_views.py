import unittest
from types import SimpleNamespace

from scripts.build_ud_morph_views import (
    document_to_views,
    make_randomized_morph_texts,
    parse_feats,
)


class MorphologyViewTests(unittest.TestCase):
    def test_feature_parsing_is_stable(self):
        self.assertEqual(parse_feats("Number=Sing|Case=Nom"), ["Case=Nom", "Number=Sing"])
        self.assertEqual(parse_feats(None), [])

    def test_document_serialization(self):
        words = [
            SimpleNamespace(text="evlerde", lemma="ev", upos="NOUN", feats="Case=Loc|Number=Plur"),
            SimpleNamespace(text="değil", lemma="değil", upos="AUX", feats="Polarity=Neg"),
        ]
        document = SimpleNamespace(sentences=[SimpleNamespace(words=words)])
        views = document_to_views(document)

        self.assertEqual(views["lemma_text"], "ev değil")
        self.assertIn("<M:UPOS=NOUN>", views["morph_text"])
        self.assertIn("<M:Case=Loc>", views["morph_text"])
        self.assertEqual(views["ud_word_count"], 2)
        self.assertEqual(views["ud_morph_feature_count"], 5)

    def test_randomized_placebo_preserves_bundle_inventory(self):
        rows = [
            {"_lemmas": ["a", "b"], "_bundles": [["UPOS=NOUN"], ["UPOS=VERB", "Tense=Past"]]},
            {"_lemmas": ["c"], "_bundles": [["UPOS=ADJ"]]},
        ]
        randomized = make_randomized_morph_texts(rows, seed=42)
        self.assertEqual(len(randomized), 2)
        combined = " ".join(randomized)
        for tag in ("UPOS=NOUN", "UPOS=VERB", "Tense=Past", "UPOS=ADJ"):
            self.assertEqual(combined.count(tag), 1)


if __name__ == "__main__":
    unittest.main()
