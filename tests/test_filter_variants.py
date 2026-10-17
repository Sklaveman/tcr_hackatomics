"""filter_variants: no grid, the default grid (bare --grid_fisher), explicit values."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline as pl


class FilterVariants(unittest.TestCase):
    def filters(self, mode):
        return {**pl.DEFAULT_FILTERS, "seq_filter": "kmer", "fisher_select": mode}

    def test_off_by_default(self):
        v = pl.filter_variants(self.filters("percentile"), None)
        self.assertEqual([x["fisher_top_percent"] for x in v], [5.0])

    def test_bare_flag_uses_the_default_percent_grid(self):
        v = pl.filter_variants(self.filters("percentile"), [])
        self.assertEqual([x["fisher_top_percent"] for x in v], [1.0, 5.0, 20.0])

    def test_bare_flag_needs_values_for_other_modes(self):
        for mode in ("fdr", "topk"):
            with self.assertRaises(SystemExit) as cm:
                pl.filter_variants(self.filters(mode), [])
            self.assertIn("percentile", str(cm.exception))

    def test_explicit_values_and_types(self):
        v = pl.filter_variants(self.filters("topk"), [200, 2000])
        self.assertEqual([x["fisher_top_k"] for x in v], [200, 2000])
        self.assertTrue(all(isinstance(x["fisher_top_k"], int) for x in v))
        self.assertTrue(all(x["fisher_select"] == "topk" for x in v))


if __name__ == "__main__":
    unittest.main()
