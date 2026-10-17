"""Unit tests for src/feature_selection.py (synthetic, runs in seconds)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import feature_selection as fs


def keys(*items):
    return np.array([i.encode() for i in items], dtype="S120")


class Presence(unittest.TestCase):
    def setUp(self):
        self.s = [keys("CASSAF+TRBV1_TRBJ1-1", "CASSBF+TRBV2_TRBJ2-1"),
                  keys("CASSAF+TRBV1_TRBJ1-1"),
                  keys("CASSCF+TRBV2_TRBJ1-1")]

    def test_clonotype(self):
        sp_ = fs.build_presence_matrix(self.s, "clonotype")
        X = sp_["X"].toarray()
        self.assertEqual(X.shape, (3, 3))
        self.assertEqual(X.sum(), 4)
        i = list(sp_["names"]).index("CASSAF+TRBV1_TRBJ1-1")
        self.assertEqual(X[:, i].tolist(), [1, 1, 0])

    def test_vj(self):
        sp_ = fs.build_presence_matrix(self.s, "vj")
        names = list(sp_["names"])
        X = sp_["X"].toarray()
        self.assertEqual(X[:, names.index("TRBV1")].tolist(), [1, 1, 0])
        self.assertEqual(X[:, names.index("TRBJ1-1")].tolist(), [1, 1, 1])

    def test_kmer_and_key_mapping(self):
        sp_ = fs.build_presence_matrix(self.s, "kmer", k=3)
        names = list(sp_["names"])
        X = sp_["X"].toarray()
        self.assertEqual(X[:, names.index("SSA")].tolist(), [1, 1, 0])  # CASSAF = CAS ASS SSA SAF
        self.assertEqual(X[:, names.index("SSC")].tolist(), [0, 0, 1])
        mask = sp_["key_features"](np.array([names.index("SSB")]))
        self.assertEqual([k for k, m in zip(sp_["unique_keys"], mask) if m], ["CASSBF+TRBV2_TRBJ2-1"])

    def test_kmer_with_v_separates_genes(self):
        s = [keys("CASSAF+TRBV1_TRBJ1-1"), keys("CASSAF+TRBV2_TRBJ1-1")]
        sp_ = fs.build_presence_matrix(s, "kmer", k=3, with_v=True)
        names = list(sp_["names"])
        X = sp_["X"].toarray()
        self.assertEqual(X[:, names.index("TRBV1|SSA")].tolist(), [1, 0])
        self.assertEqual(X[:, names.index("TRBV2|SSA")].tolist(), [0, 1])

    def test_trimming_cuts_the_ends_before_taking_kmers(self):
        s = [keys("CASAAAFGF+TRBV1_TRBJ1-1"), keys("CASGG+TRBV1_TRBJ1-1")]
        plain = fs.build_presence_matrix(s, "kmer", k=3)
        self.assertIn("CAS", list(plain["names"]))
        cut = fs.build_presence_matrix(s, "kmer", k=3, trim_start=3, trim_end=3)
        self.assertEqual(list(cut["names"]), ["AAA"])                    # only the middle of the first CDR3
        self.assertEqual(cut["X"].toarray().tolist(), [[1], [0]])       # the second one is too short to carry anything
        mask = cut["key_features"](np.array([0]))
        self.assertEqual(mask.tolist(), [True, False])
        # trimming only one side
        end_only = fs.build_presence_matrix(s, "kmer", k=3, trim_end=3)
        self.assertIn("CAS", list(end_only["names"]))
        self.assertNotIn("FGF", list(end_only["names"]))

    def test_trimming_everything_is_a_clear_error(self):
        with self.assertRaises(ValueError) as cm:
            fs.build_presence_matrix([keys("CASSAF+TRBV1_TRBJ1-1")], "kmer", k=3, trim_start=3, trim_end=3)
        self.assertIn("long enough", str(cm.exception))

    def test_with_v_names_are_lazy_but_behave_like_a_list(self):
        s = [keys("CASSAF+TRBV1_TRBJ1-1", "CASSBF+TRBV2_TRBJ1-1"), keys("CASSAF+TRBV2_TRBJ1-1")]
        sp_ = fs.build_presence_matrix(s, "kmer", k=3, with_v=True, trim_start=1)
        names = sp_["names"]
        self.assertEqual(len(names), sp_["X"].shape[1])
        self.assertEqual(sorted(names), sorted(set(names)))
        self.assertIn("TRBV1|SSA", list(names))
        self.assertNotIn("TRBV1|CAS", list(names))                       # cut by trim_start=1
        j = list(names).index("TRBV2|SSB")
        self.assertEqual(names[j], "TRBV2|SSB")
        self.assertEqual(fs.take_names(names, np.array([j])).tolist(), ["TRBV2|SSB"])
        out = fs.fit_sequence_filter(s * 4, np.array([0, 1] * 4), feature_type="kmer", with_v=True,
                                     min_samples=2, select="topk", top_k=2)
        self.assertGreaterEqual(out["stats"]["n_survivors"], 1)


class Fisher(unittest.TestCase):
    def test_matches_scipy_fisher_exact(self):
        rng = np.random.default_rng(0)
        N, F = 60, 200
        y = rng.integers(0, 3, N)
        X = (rng.random((N, F)) < rng.random(F) * 0.4).astype(np.int8)
        import scipy.sparse as sp
        res = fs.fisher_ovr(sp.csr_matrix(X), y, min_samples=1)
        for c in range(3):
            for f in rng.choice(F, 30, replace=False):
                k = X[:, f].sum()
                if k < 1:
                    continue
                a = X[y == c, f].sum()
                n_c = (y == c).sum()
                table = [[a, n_c - a], [k - a, (N - n_c) - (k - a)]]
                expected = fisher_exact(table, alternative="greater")[1]
                self.assertAlmostEqual(res[c]["p"][f], expected, places=10)

    def test_min_samples_untested(self):
        import scipy.sparse as sp
        X = sp.csr_matrix(np.array([[1, 1], [0, 1], [0, 1], [0, 0]], dtype=np.int8))
        res = fs.fisher_ovr(X, np.array([0, 0, 1, 1]), min_samples=2)
        self.assertTrue(np.isnan(res[0]["p"][0]))
        self.assertFalse(np.isnan(res[0]["p"][1]))

    def test_bh_known_values(self):
        q = fs.bh_adjust([0.01, 0.04, 0.03, 0.005, np.nan])
        np.testing.assert_allclose(q[:4], [0.02, 0.04, 0.04, 0.02])
        self.assertTrue(np.isnan(q[4]))

    def test_topk_ties_are_deterministic(self):
        res = {0: {"p": np.array([0.01, 0.01, 0.01, 0.5]), "a": np.array([3, 5, 5, 1]), "n_c": 10}}
        names = np.array(["c", "b", "a", "d"])
        sel = fs.select_features(res, names, select="topk", top_k=2)[0]
        self.assertEqual(sel.tolist(), [2, 1])  # equal p and fraction -> name order a, b; higher fraction first

    def test_percentile_counts_and_order_independence(self):
        p = np.array([0.5, 0.01, np.nan, 0.2, 0.04, 0.3, 0.001, 0.9, 0.6, 0.7, np.nan])   # 9 tested
        res = {0: {"p": p, "a": np.ones(11, int), "n_c": 10}}
        names = np.array([f"f{i:02d}" for i in range(11)])
        sel = fs.select_features(res, names, select="percentile", top_percent=25)[0]
        self.assertEqual(sel.tolist(), [6, 1, 4])               # ceil(0.25*9)=3 smallest p, NaN never chosen
        self.assertEqual(len(fs.select_features(res, names, select="percentile", top_percent=0.01)[0]), 1)  # at least 1
        self.assertEqual(len(fs.select_features(res, names, select="percentile", top_percent=100)[0]), 9)
        # same set after permuting the features
        perm = np.random.default_rng(0).permutation(11)
        res2 = {0: {"p": p[perm], "a": np.ones(11, int), "n_c": 10}}
        sel2 = fs.select_features(res2, names[perm], select="percentile", top_percent=25)[0]
        self.assertEqual(set(names[perm][sel2]), set(names[sel]))

    def test_percentile_is_per_class(self):
        res = {0: {"p": np.array([0.001, 0.5, 0.6, 0.7]), "a": np.ones(4, int), "n_c": 5},
               1: {"p": np.array([0.9, 0.8, 0.002, 0.7]), "a": np.ones(4, int), "n_c": 5}}
        sel = fs.select_features(res, np.array(list("abcd")), select="percentile", top_percent=25)
        self.assertEqual((sel[0].tolist(), sel[1].tolist()), ([0], [2]))

    def test_percentile_rejects_bad_values(self):
        res = {0: {"p": np.array([0.1, 0.2]), "a": np.ones(2, int), "n_c": 5}}
        for bad in (0, -1, 101):
            with self.assertRaises(ValueError):
                fs.select_features(res, np.array(["a", "b"]), select="percentile", top_percent=bad)


class SequenceFilter(unittest.TestCase):
    def make(self, n_per_class=12, seed=0):
        """3 classes; each class has a public clonotype carried by most of its
        samples; plus per-sample private noise and one clonotype shared by all."""
        rng = np.random.default_rng(seed)
        keys_, y, dis = [], [], []
        for c in range(3):
            for i in range(n_per_class):
                s = [f"CASS{c}PUBLICF+TRBV1_TRBJ1-1"] if i < int(0.8 * n_per_class) else []
                s += ["CASSEVERYONEF+TRBV2_TRBJ1-1"]
                s += [f"CASSP{rng.integers(10**9)}F+TRBV3_TRBJ2-1" for _ in range(20)]
                keys_.append(np.array([x.encode() for x in s], dtype="S120"))
                y.append(c)
                dis.append(f"d{c}_{i % 2}")
        return keys_, np.array(y), np.array(dis)

    def test_recovers_class_specific_public_clonotypes(self):
        keys_, y, dis = self.make()
        out = fs.fit_sequence_filter(keys_, y, disease=dis, feature_type="clonotype",
                                     min_samples=3, select="fdr", alpha=0.05)
        survivors = {k.decode() for k in out["survivor_keys"]}
        self.assertEqual(survivors, {f"CASS{c}PUBLICF+TRBV1_TRBJ1-1" for c in range(3)})
        self.assertEqual(out["stats"]["n_survivors"], 3)
        self.assertEqual(sorted(out["report"]["class"].tolist()), [0, 1, 2])
        self.assertTrue((out["report"]["n_diseases_in_class"] == 2).all())

    def test_shared_tests_match_single_fits(self):
        keys_, y, dis = self.make()
        tests = fs.fisher_tests(keys_, y, feature_type="clonotype", min_samples=3)
        for kw in ({"select": "percentile", "top_percent": 5}, {"select": "topk", "top_k": 4}, {"select": "fdr"}):
            a = fs.finish_filter(tests, y, dis, **kw)
            b = fs.fit_sequence_filter(keys_, y, disease=dis, feature_type="clonotype", min_samples=3, **kw)
            np.testing.assert_array_equal(a["survivor_keys"], b["survivor_keys"])
        small = fs.finish_filter(tests, y, dis, select="percentile", top_percent=1)["stats"]["n_survivors"]
        large = fs.finish_filter(tests, y, dis, select="percentile", top_percent=50)["stats"]["n_survivors"]
        self.assertLessEqual(small, large)

    def test_vj_and_kmer_run(self):
        keys_, y, dis = self.make()
        for ft in ("vj", "kmer"):
            out = fs.fit_sequence_filter(keys_, y, disease=dis, feature_type=ft,
                                         min_samples=3, select="topk", top_k=5)
            self.assertGreaterEqual(out["stats"]["n_survivors"], 3)


class Prevalence(unittest.TestCase):
    def setUp(self):
        n = 60
        self.y = np.repeat([0, 1, 2], 20)
        occ = np.zeros((n, 5))
        occ[:, 0] = 1                       # everywhere
        occ[:6, 1] = 1                      # 30% of class 0, 10% overall
        occ[:2, 2] = 1                      # 2 samples of class 0: count guard
        occ[20:23, 3] = 1                   # 15% of class 1, 5% overall
        occ[:, 4] = 0                       # never
        self.f = pd.DataFrame(occ, columns=[f"c{i}" for i in range(5)])
        for e in fs.EXTRA_FEATURES:
            self.f[e] = 0.5

    def test_class_aware_keeps_class_concentrated_clusters(self):
        glob = fs.select_clusters_by_prevalence(self.f, self.y, "global", 0.1)
        ca = fs.select_clusters_by_prevalence(self.f, self.y, "class_aware", 0.1, min_count=3)
        self.assertEqual(glob, ["c0", "c1"])           # c1: 6/60 = 10% overall
        self.assertEqual(ca, ["c0", "c1", "c3"])       # c3: 15% of class 1, 3 samples
        self.assertNotIn("c2", ca)                      # only 2 samples
        self.assertTrue(set(glob) <= set(ca))
        self.assertNotIn("unassigned", ca)

    def test_stricter_threshold_drops_global_but_not_class_aware(self):
        f = self.f.copy()
        f["c1"] = 0.0
        f.loc[:5, "c1"] = 1                            # 30% of class 0, 10% overall
        glob = fs.select_clusters_by_prevalence(f, self.y, "global", 0.15)
        ca = fs.select_clusters_by_prevalence(f, self.y, "class_aware", 0.15)
        self.assertNotIn("c1", glob)
        self.assertIn("c1", ca)

    def test_report_flags_only_class_aware(self):
        sel = fs.select_clusters_by_prevalence(self.f, self.y, "class_aware", 0.1)
        dis = np.array([f"d{i // 10}" for i in range(60)])
        rep = fs.prevalence_report(self.f, self.y, dis, sel, 0.1)
        self.assertEqual(rep.set_index("cluster").loc["c3", "only_class_aware"], True)
        self.assertEqual(rep.set_index("cluster").loc["c3", "best_class"], 1)

    def test_signature_cannot_see_validation_labels(self):
        import inspect
        self.assertEqual(list(inspect.signature(fs.select_clusters_by_prevalence).parameters)[:2],
                         ["features", "y"])


if __name__ == "__main__":
    unittest.main()
