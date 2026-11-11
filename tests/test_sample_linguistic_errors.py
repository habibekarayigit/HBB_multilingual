import unittest

import pandas as pd

from scripts.sample_linguistic_errors import PHENOMENON_COLUMNS, create_sample


class ErrorSamplingTests(unittest.TestCase):
    def test_balanced_sampling_and_blinding(self):
        rows = []
        index = 0
        for language in ("TR", "KO"):
            for label, prediction in ((0, 0), (0, 1), (1, 0), (1, 1)):
                for _ in range(4):
                    rows.append(
                        {
                            "example_id": index,
                            "language": language,
                            "text": f"text-{index}",
                            "label": label,
                            "prediction": prediction,
                            "probability_1": 0.5,
                        }
                    )
                    index += 1
        blind, key, availability = create_sample(
            pd.DataFrame(rows), per_cell=3, seed=42
        )

        self.assertEqual(len(blind), 24)
        self.assertEqual(len(key), 24)
        self.assertEqual(set(key["outcome"]), {"TN", "FP", "FN", "TP"})
        self.assertNotIn("label", blind.columns)
        self.assertNotIn("prediction", blind.columns)
        self.assertTrue(set(PHENOMENON_COLUMNS).issubset(blind.columns))
        self.assertEqual(set(availability["available"]), {4})
        self.assertEqual(set(availability["sampled"]), {3})


if __name__ == "__main__":
    unittest.main()
