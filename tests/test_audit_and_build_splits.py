import unittest

import pandas as pd

from scripts.audit_and_build_splits import (
    build_stratified_manifest,
    normalize_text,
)


class NormalizeTextTests(unittest.TestCase):
    def test_normalizes_unicode_case_and_whitespace(self):
        self.assertEqual(normalize_text("  İYİ\u200b   Günler  "), "i̇yi̇ günler")

    def test_does_not_collapse_distinct_urls_or_users(self):
        self.assertNotEqual(normalize_text("@a https://a.example"), normalize_text("@b https://b.example"))


class ManifestTests(unittest.TestCase):
    def test_manifest_is_disjoint_and_has_requested_sizes(self):
        frame = pd.DataFrame(
            {
                "row_id": range(100),
                "label": [0] * 60 + [1] * 40,
                "text_sha256": [f"hash-{index}" for index in range(100)],
            }
        )
        manifest = build_stratified_manifest(
            frame,
            row_id_col="row_id",
            label_col="label",
            split_sizes={"train": 50, "dev": 20, "test": 20},
            seed=42,
            hash_col="text_sha256",
        )

        self.assertEqual(manifest.groupby("split").size().to_dict(), {"dev": 20, "test": 20, "train": 50})
        self.assertFalse(manifest["row_id"].duplicated().any())
        self.assertFalse(manifest["text_sha256"].duplicated().any())
        self.assertEqual(manifest.groupby("split")["label"].mean().to_dict(), {"dev": 0.4, "test": 0.4, "train": 0.4})

    def test_rejects_nonbinary_labels(self):
        frame = pd.DataFrame({"row_id": [0, 1, 2], "label": [0, 1, 2]})
        with self.assertRaisesRegex(ValueError, "binary labels"):
            build_stratified_manifest(
                frame,
                row_id_col="row_id",
                label_col="label",
                split_sizes={"test": 2},
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
