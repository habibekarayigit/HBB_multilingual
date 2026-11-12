import ast
import json
import unittest
from pathlib import Path


class RevisedNotebookTests(unittest.TestCase):
    def test_notebook_is_clean_and_code_cells_parse(self):
        notebook = json.loads(Path("Cross_lingual_REVISED.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))
        text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("pin_model_revisions.py", text)
        self.assertIn("bootstrap_encoder_comparisons.py", text)
        self.assertIn("MORPH_QC_APPROVED", text)
        self.assertIn("Expected 30 runs", text)


if __name__ == "__main__":
    unittest.main()
