import unittest

import pandas as pd

from scripts.bootstrap_encoder_comparisons import comparison_rows


class EncoderBootstrapTests(unittest.TestCase):
    def make_predictions(self, seeds=(42, 123, 456)):
        rows = []
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        for seed in seeds:
            for model, predictions in (
                ("model-a", labels),
                ("model-b", [0, 1, 0, 1, 1, 0, 1, 0]),
            ):
                for index, (label, prediction) in enumerate(zip(labels, predictions)):
                    rows.append(
                        {
                            "run_id": f"{model}-{seed}",
                            "example_id": f"example-{index}",
                            "language": "FI",
                            "source_dataset": "fixture",
                            "label": label,
                            "prediction": prediction,
                            "seed": seed,
                            "model": model,
                            "model_revision": "a" * 40,
                            "view": "surface",
                            "max_length": 128,
                            "frozen_encoder": False,
                            "scientifically_valid": True,
                        }
                    )
        return pd.DataFrame(rows)

    def config(self):
        return {
            "expected_seeds": [42, 123, 456],
            "replicates": 100,
            "bootstrap_seed": 7,
            "comparisons": [
                {
                    "name": "a_minus_b",
                    "a": {
                        "model": "model-a",
                        "view": "surface",
                        "max_length": 128,
                        "frozen_encoder": False,
                    },
                    "b": {
                        "model": "model-b",
                        "view": "surface",
                        "max_length": 128,
                        "frozen_encoder": False,
                    },
                }
            ],
        }

    def test_hierarchical_paired_bootstrap(self):
        result = comparison_rows(self.make_predictions(), self.config())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["n_seeds"], 3)
        self.assertGreater(result.iloc[0]["difference_a_minus_b"], 0)
        self.assertIn("p_holm_all_prespecified_tests", result.columns)

    def test_incomplete_seeds_fail(self):
        with self.assertRaisesRegex(ValueError, "Incomplete seeds"):
            comparison_rows(self.make_predictions(seeds=(42, 123)), self.config())


if __name__ == "__main__":
    unittest.main()
