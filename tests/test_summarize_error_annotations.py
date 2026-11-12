import unittest

import pandas as pd

from scripts.sample_linguistic_errors import PHENOMENON_COLUMNS
from scripts.summarize_error_annotations import summarize


class ErrorAnnotationSummaryTests(unittest.TestCase):
    def test_prevalence_summary(self):
        annotations = []
        key = []
        for index, outcome in enumerate(("FP", "FP", "FN", "FN")):
            row = {
                "annotation_id": f"ANN-{index}",
                "primary_phenomenon": "negation",
            }
            for phenomenon in PHENOMENON_COLUMNS:
                row[phenomenon] = "1" if index % 2 else "0"
            annotations.append(row)
            key.append(
                {
                    "annotation_id": f"ANN-{index}",
                    "language": "TR",
                    "outcome": outcome,
                    "label": 0 if outcome == "FP" else 1,
                    "prediction": 1 if outcome == "FP" else 0,
                }
            )
        prevalence, primary = summarize(pd.DataFrame(annotations), pd.DataFrame(key))
        negation = prevalence.loc[prevalence["phenomenon"].eq("negation")]
        self.assertEqual(set(negation["prevalence"]), {0.5})
        self.assertEqual(primary["n"].sum(), 4)


if __name__ == "__main__":
    unittest.main()
