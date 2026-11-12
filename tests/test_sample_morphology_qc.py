import unittest

import pandas as pd

from scripts.sample_morphology_qc import QC_COLUMNS, create_qc_sample


class MorphologyQCTests(unittest.TestCase):
    def test_language_balanced_sample(self):
        rows = []
        for language in ("TR", "KO"):
            for index in range(5):
                rows.append(
                    {
                        "example_id": f"{language}-{index}",
                        "language": language,
                        "split": "test",
                        "text": f"surface {index}",
                        "lemma_text": f"lemma {index}",
                        "morph_text": f"lemma {index} <M:UPOS=NOUN>",
                        "randomized_morph_text": f"lemma {index} <M:UPOS=VERB>",
                    }
                )
        sample = create_qc_sample(pd.DataFrame(rows), per_language=3, seed=42)
        self.assertEqual(len(sample), 6)
        self.assertEqual(sample.groupby("language").size().to_dict(), {"KO": 3, "TR": 3})
        self.assertTrue(set(QC_COLUMNS).issubset(sample.columns))
        self.assertTrue(sample["qc_id"].is_unique)


if __name__ == "__main__":
    unittest.main()
