import json
import unittest
from pathlib import Path

from scripts.pin_model_revisions import pin_config
from scripts.run_experiment_matrix import (
    command_for,
    expand_matrix,
    is_immutable_revision,
    run_id,
)


class ExperimentMatrixTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            Path("configs/experiment_matrix.json").read_text(encoding="utf-8")
        )

    def test_matrix_has_expected_runs_and_block_specific_data(self):
        runs = expand_matrix(self.config)
        self.assertEqual(len(runs), 30)
        morphology = [run for run in runs if run["view"] != "surface"]
        surface = [run for run in runs if run["view"] == "surface"]
        self.assertEqual(len(morphology), 9)
        self.assertEqual(len(surface), 21)
        self.assertTrue(
            all(run["data"].endswith("standardized_with_morphology.parquet") for run in morphology)
        )
        self.assertTrue(all(run["data"].endswith("standardized_all.csv") for run in surface))
        effective_batches = {
            run["train_batch_size"] * run["gradient_accumulation_steps"] for run in runs
        }
        self.assertEqual(effective_batches, {16})

    def test_run_id_includes_revision(self):
        run = expand_matrix(self.config)[0]
        self.assertIn("__revmain__", run_id(run))
        self.assertEqual(command_for(run)[0], "python")
        self.assertFalse(is_immutable_revision("main"))
        self.assertTrue(is_immutable_revision("a" * 40))

    def test_revision_pinning_is_deduplicated(self):
        calls = []

        def resolver(name, revision):
            calls.append((name, revision))
            return {"bert-base-multilingual-cased": "a" * 40}.get(name, "b" * 40)

        pinned, records = pin_config(self.config, resolver)
        runs = expand_matrix(pinned)
        self.assertTrue(all(is_immutable_revision(run["model_revision"]) for run in runs))
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(records), 4)


if __name__ == "__main__":
    unittest.main()
