"""Excluding training diseases (--exclude_diseases): the metadata filter."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline as pl


def meta():
    rows = []
    for g, ds in {"autoimmunity": ["MS", "T1D"], "cancer": ["Melanoma", "Ovarian_cancer", "NSCLC"],
                  "infection": ["CMV"]}.items():
        for d in ds:
            rows += [{"sample_id": f"{d}_{i}", "disease": d, "disease_group": g} for i in range(3)]
    return pd.DataFrame(rows)


class DropDiseases(unittest.TestCase):
    def test_none_or_empty_keeps_everything(self):
        for ex in (None, []):
            kept, names = pl.drop_diseases(meta(), ex)
            self.assertEqual((len(kept), names), (18, []))

    def test_case_insensitive_and_canonical_names(self):
        kept, names = pl.drop_diseases(meta(), ["melanoma", "OVARIAN_CANCER", "Melanoma"])
        self.assertEqual(names, ["Melanoma", "Ovarian_cancer"])
        self.assertEqual(len(kept), 12)
        self.assertFalse(kept["disease"].isin(names).any())
        self.assertEqual(kept.index.tolist(), list(range(12)))     # index reset

    def test_unknown_disease_lists_the_available_ones(self):
        with self.assertRaises(SystemExit) as cm:
            pl.drop_diseases(meta(), ["Flu"])
        self.assertIn("Flu", str(cm.exception))
        self.assertIn("Melanoma", str(cm.exception))

    def test_removing_a_whole_class_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            pl.drop_diseases(meta(), ["CMV"])
        self.assertIn("infection", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
