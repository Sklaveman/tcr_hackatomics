"""Unit tests for src/preprocessing.py (pure pandas, runs in seconds).

Run all tests with:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocessing as prep

OK = "CASSLGTDTQYF"


def frame(rows):
    return pd.DataFrame(rows, columns=["cdr3_aa", "v_gene", "j_gene", "clone_count"])


class PreprocessRepertoire(unittest.TestCase):
    def run_one(self, rows, **kw):
        return prep.preprocess_repertoire(frame(rows), **kw)

    def test_noncanonical_cdr3_dropped(self):
        rows = [(OK, "TRBV5-1", "TRBJ2-3", 1),
                ("ASSLGTDTQYF", "TRBV5-1", "TRBJ2-3", 1),     # no leading C
                ("CASSLGTDTQY", "TRBV5-1", "TRBJ2-3", 1),     # not ending F/W
                ("CASS*GTDTQYF", "TRBV5-1", "TRBJ2-3", 1),    # stop codon
                ("CASS_GTDTQYF", "TRBV5-1", "TRBJ2-3", 1),    # frameshift
                ("CASSXGTDTQYF", "TRBV5-1", "TRBJ2-3", 1),    # non-standard aa
                (None, "TRBV5-1", "TRBJ2-3", 1)]
        df, log = self.run_one(rows)
        self.assertEqual(df["cdr3_aa"].tolist(), [OK])
        self.assertEqual(log["n_noncanonical"], 6)

    def test_max_length_only_when_requested(self):
        long_cdr3 = "C" + "A" * 40 + "F"
        rows = [(OK, "TRBV5-1", "TRBJ2-3", 1), (long_cdr3, "TRBV5-1", "TRBJ2-3", 1)]
        df, log = self.run_one(rows, max_cdr3_len=40)
        self.assertEqual((len(df), log["n_too_long"]), (1, 1))
        df, log = self.run_one(rows)  # no maximum length
        self.assertEqual((len(df), log["n_too_long"]), (2, 0))

    def test_junk_ambiguous_and_missing_vj_dropped(self):
        rows = [(OK, "TRBV5-1", "TRBJ2-3", 1),
                ("CASSAAAF", "TRBVA-OR09_02", "TRBJ2-3", 1),        # '_'
                ("CASSHHHF", "TRBV5+TRBV6", "TRBJ2-3", 1),                  # '+'
                ("CASSCCCF", "TRBV6-1,TRBV6-4", "TRBJ2-3", 1),      # ambiguous V
                ("CASSEEEF", "TRBV5-1", "TRBJ2-1,TRBJ2-5", 1),      # ambiguous J
                ("CASSFFFF", None, "TRBJ2-3", 1),                   # missing V
                ("CASSGGGF", "TRBV5-1", None, 1)]                   # missing J
        df, log = self.run_one(rows)
        self.assertEqual(df["cdr3_aa"].tolist(), [OK])
        self.assertEqual((log["n_vgene_junk"], log["n_ambiguous_vj"], log["n_missing_vj"]),
                         (2, 2, 2))

    def test_zero_clone_count_dropped(self):
        df, log = self.run_one([(OK, "TRBV5-1", "TRBJ2-3", 0), (OK, "TRBV5-1", "TRBJ2-1", 4)])
        self.assertEqual((len(df), log["n_zero_clonecount"]), (1, 1))

    def test_single_member_genes_canonicalized_and_collisions_collapsed(self):
        rows = [(OK, "TRBV2-1", "TRBJ2-3", 3),   # same clonotype under both names:
                (OK, "TRBV2", "TRBJ2-3", 4),     # must collapse AFTER renaming
                ("CASSAAAF", "TRBV12-1", "TRBJ2-3", 1),   # real family member: untouched
                ("CASSDDDF", "TRBV30-1", "TRBJ2-3", 2)]
        df, log = self.run_one(rows)
        got = dict(zip(df["key"], df["clone_count"]))
        self.assertEqual(got, {f"{OK}+TRBV2_TRBJ2-3": 7,
                               "CASSAAAF+TRBV12-1_TRBJ2-3": 1,
                               "CASSDDDF+TRBV30_TRBJ2-3": 2})
        self.assertEqual((log["n_v_renamed"], log["n_dup_rows_collapsed"]), (2, 1))

    def test_nucleotide_duplicates_collapse(self):
        df, log = self.run_one([(OK, "TRBV5-1", "TRBJ2-3", 1)] * 3)
        self.assertEqual((len(df), df["clone_count"].iloc[0], log["n_dup_rows_collapsed"]), (1, 3, 2))

    def test_log_accounts_for_every_row(self):
        rows = [(OK, "TRBV5-1", "TRBJ2-3", 1), (OK, "TRBV5-1", "TRBJ2-3", 2),
                ("ASS", "TRBV5-1", "TRBJ2-3", 1), ("CASSAAAF", None, "TRBJ2-3", 1),
                ("CASSDDDF", "TRBV5-1", "TRBJ2-3", 0)]
        _, log = self.run_one(rows)
        removed = sum(log[k] for k in ("n_noncanonical", "n_too_long", "n_vgene_junk",
                                       "n_ambiguous_vj", "n_missing_vj", "n_zero_clonecount",
                                       "n_dup_rows_collapsed"))
        self.assertEqual(log["n_raw_rows"], removed + log["n_clonotypes"])

    def test_empty_result_is_well_formed(self):
        df, log = self.run_one([("bad", "TRBV5-1", "TRBJ2-3", 1)])
        self.assertEqual((len(df), log["n_clonotypes"]), (0, 0))
        self.assertIn("key", df.columns)


class SelectSamples(unittest.TestCase):
    def test_balanced_over_classes_and_seeded(self):
        meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(12)],
                             "disease_group": ["a"] * 6 + ["b"] * 4 + ["c"] * 2})
        picked = prep.select_samples(meta, 6, seed=1)
        classes = meta.set_index("sample_id").loc[picked, "disease_group"].value_counts()
        self.assertEqual(classes.to_dict(), {"a": 2, "b": 2, "c": 2})
        self.assertEqual(picked, prep.select_samples(meta, 6, seed=1))
        self.assertEqual(len(prep.select_samples(meta, None)), 12)


if __name__ == "__main__":
    unittest.main()
