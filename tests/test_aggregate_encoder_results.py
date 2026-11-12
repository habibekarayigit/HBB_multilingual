import unittest

import pandas as pd

from scripts.aggregate_encoder_results import aggregate


class AggregateEncoderResultsTests(unittest.TestCase):
    def make_frame(self, seeds=(42, 123, 456)):
        rows = []
        for seed, score in zip(seeds, (0.4, 0.5, 0.6)):
            rows.append(
                {
                    "model": "model",
                    "model_revision": "a" * 40,
                    "view": "surface",
                    "max_length": 128,
                    "frozen_encoder": False,
                    "language": "KO",
                    "source_dataset": "BEEP",
                    "split": "test",
                    "seed": seed,
                    "scientifically_valid": True,
                    "accuracy": score,
                    "macro_f1": score,
                    "f1_abusive": score,
                    "macro_precision": score,
                    "macro_recall": score,
                    "mcc": score,
                    "false_positive_rate": 1 - score,
                    "false_negative_rate": 1 - score,
                }
            )
        return pd.DataFrame(rows)

    def test_complete_three_seed_aggregation(self):
        summary, completeness = aggregate(
            self.make_frame(), expected_seeds={42, 123, 456}, allow_incomplete=False
        )
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.iloc[0]["macro_f1_mean"], 0.5)
        self.assertEqual(summary.iloc[0]["seed_count"], 3)
        self.assertTrue(completeness.iloc[0]["complete"])

    def test_incomplete_seed_set_fails(self):
        with self.assertRaisesRegex(ValueError, "Incomplete seed"):
            aggregate(
                self.make_frame(seeds=(42, 123)),
                expected_seeds={42, 123, 456},
                allow_incomplete=False,
            )


if __name__ == "__main__":
    unittest.main()
