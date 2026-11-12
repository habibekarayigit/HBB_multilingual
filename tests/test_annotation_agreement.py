import unittest

import pandas as pd

from scripts.calculate_annotation_agreement import calculate
from scripts.sample_linguistic_errors import PHENOMENON_COLUMNS


class AnnotationAgreementTests(unittest.TestCase):
    def make_frame(self, disagreement=False):
        rows = []
        for index in range(6):
            row = {
                "annotation_id": f"ANN-{index}",
                "language": "TR",
                "text": f"text {index}",
                "primary_phenomenon": "negation" if index % 2 else "obfuscation",
            }
            for column in PHENOMENON_COLUMNS:
                row[column] = str(index % 2)
            rows.append(row)
        if disagreement:
            rows[0]["negation"] = "1" if rows[0]["negation"] == "0" else "0"
        return pd.DataFrame(rows)

    def test_agreement_and_disagreement_outputs(self):
        agreement, disagreements = calculate(
            [self.make_frame(), self.make_frame(disagreement=True)], ["A", "B"]
        )
        self.assertEqual(set(agreement["annotator_a"]), {"A"})
        self.assertEqual(set(agreement["annotator_b"]), {"B"})
        self.assertEqual(len(disagreements), 1)
        negation = agreement.loc[agreement["column"].eq("negation")].iloc[0]
        self.assertLess(negation["raw_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
