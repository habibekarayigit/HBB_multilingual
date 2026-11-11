import unittest

import numpy as np
import torch

from scripts.run_encoder_experiment import binary_metrics, freeze_base_encoder, parameter_counts


class DummyClassifier(torch.nn.Module):
    base_model_prefix = "encoder"

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 2)


class EncoderRunnerTests(unittest.TestCase):
    def test_binary_metrics_confusion_matrix(self):
        metrics = binary_metrics(
            np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1])
        )
        self.assertEqual(
            {key: metrics[key] for key in ("tn", "fp", "fn", "tp")},
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)

    def test_frozen_encoder_leaves_head_trainable(self):
        model = DummyClassifier()
        freeze_base_encoder(model)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        counts = parameter_counts(model)
        self.assertGreater(counts["frozen"], 0)
        self.assertGreater(counts["trainable"], 0)


if __name__ == "__main__":
    unittest.main()
