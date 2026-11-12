import unittest

import pandas as pd

from scripts.ensemble_encoder_predictions import build_ensemble


class EnsemblePredictionTests(unittest.TestCase):
    def test_probability_ensemble(self):
        rows = []
        for seed, probability in ((42, 0.2), (123, 0.8), (456, 0.7)):
            rows.append(
                {
                    "example_id": "E1",
                    "language": "FI",
                    "source_dataset": "fixture",
                    "split": "test",
                    "label": 1,
                    "text": "example",
                    "text_sha256": "hash",
                    "probability_1": probability,
                    "prediction": int(probability >= 0.5),
                    "seed": seed,
                    "model": "model",
                    "model_revision": "a" * 40,
                    "view": "surface",
                    "max_length": 128,
                    "frozen_encoder": False,
                    "scientifically_valid": True,
                    "run_id": f"run-{seed}",
                }
            )
        result = build_ensemble(
            pd.DataFrame(rows),
            model="model",
            view="surface",
            max_length=128,
            frozen_encoder=False,
            expected_seeds={42, 123, 456},
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["probability_1"], (0.2 + 0.8 + 0.7) / 3)
        self.assertEqual(result.iloc[0]["prediction"], 1)


if __name__ == "__main__":
    unittest.main()
