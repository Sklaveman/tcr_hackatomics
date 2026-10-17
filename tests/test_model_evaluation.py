"""Unit tests for src/model.py and src/evaluation.py (synthetic features, seconds)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import embeddings as emb
from src import evaluation as ev
from src import model as mdl


def planted(n=90, seed=0):
    """3 classes; feature 0 marks class 1, feature 1 marks class 2, rest noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n // 3)
    X = rng.random((n, 6))
    X[y == 1, 0] += 1.5
    X[y == 2, 1] += 1.5
    return X, y, [f"f{i}" for i in range(6)]


class Model(unittest.TestCase):
    def test_macro_auc_needs_all_classes(self):
        self.assertTrue(np.isnan(mdl.macro_auc(np.array([0, 1]), np.array([[.5, .5, 0], [.5, .5, 0]]))))

    def test_inner_split_is_stratified_or_none(self):
        y = np.repeat([0, 1, 2], 20)
        tr, es = mdl.inner_split(y)
        self.assertEqual(sorted(np.bincount(y[es])), [5, 5, 5])
        self.assertEqual(len(set(tr) & set(es)), 0)
        self.assertIsNone(mdl.inner_split(np.array([0, 1, 2, 2])))     # a class has 1 sample

    def test_fit_and_score_learns_planted_signal(self):
        X, y, names = planted()
        idx = np.arange(len(y))
        va = idx % 3 == 0
        proba, n_rounds, booster = mdl.fit_and_score(
            X[~va], y[~va], X[va], y[va], mdl.default_xgb_params(n_train=int((~va).sum())), names)
        self.assertEqual(proba.shape, (int(va.sum()), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        self.assertGreater(mdl.macro_auc(y[va], proba), 0.9)
        self.assertGreaterEqual(n_rounds, 1)
        self.assertEqual(booster.attr("best_iteration"), str(n_rounds - 1))

    def test_multiclass_params(self):
        p = mdl.default_xgb_params()
        self.assertEqual((p["objective"], p["num_class"], p["eval_metric"]),
                         ("multi:softprob", 3, "mlogloss"))
        self.assertNotIn("scale_pos_weight", p)

    def test_tiny_fold_falls_back_to_fixed_rounds(self):
        X, y, names = planted(9)
        proba, n_rounds, _ = mdl.fit_and_score(X[:6], np.array([0, 0, 0, 1, 1, 2]), X[6:], y[6:],
                                               mdl.default_xgb_params(n_train=6), names)   # class 2 has 1 sample
        self.assertEqual(n_rounds, 100)
        self.assertEqual(proba.shape[0], 3)

    def test_optuna_returns_num_boost_round_and_respects_small_folds(self):
        X, y, names = planted(60)
        folds = []
        for k in range(2):
            va = (np.arange(60) % 2) == k
            folds.append({"X_train": X[~va], "y_train": y[~va], "X_val": X[va], "y_val": y[va],
                          "names": names})
        best, study, trials = mdl.optimize_xgb_params(folds, n_trials=3, random_state=0)
        self.assertIn("num_boost_round", best)
        self.assertLessEqual(best["min_child_weight"], max(1, int(0.05 * 30)))
        self.assertEqual(len(trials), 3)


class Metrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = np.repeat([0, 1, 2], 20)
        P = rng.dirichlet([1, 1, 1], 60) * 0.5
        P[np.arange(60), self.y] += 0.5
        self.P = P / P.sum(1, keepdims=True)
        self.fold = np.tile([0, 1, 2], 20)
        self.dis = np.array(["a", "b", "c"] * 20)

    def test_compute_metrics_contents(self):
        m = ev.compute_metrics(self.y, self.P, self.fold, self.dis)
        self.assertEqual(len(m["macro_roc_auc_per_fold"]), 3)
        self.assertAlmostEqual(m["macro_roc_auc_mean"], np.mean(m["macro_roc_auc_per_fold"]))
        self.assertEqual(set(m["roc_auc_ovr"]), set(emb.CLASS_NAMES))
        cm = np.array(m["confusion_matrix_counts"])
        self.assertEqual(cm.sum(), 60)
        np.testing.assert_allclose(np.array(m["confusion_matrix_row_normalized"]).sum(axis=1), 1.0)
        self.assertEqual(m["per_disease_accuracy"]["a"]["n"], 20)
        self.assertIn("balanced_accuracy", m)

    def test_write_metrics_flat_csv(self):
        m = ev.compute_metrics(self.y, self.P, self.fold, self.dis)
        with tempfile.TemporaryDirectory() as d:
            ev.write_metrics(d, m)
            flat = pd.read_csv(Path(d) / "metrics.csv", index_col=0)["value"]
            self.assertIn("macro_roc_auc_pooled", flat.index)
            self.assertIn("confusion_matrix_counts[0][0]", flat.index)
            self.assertTrue((Path(d) / "metrics.json").exists())

    def test_roc_plot_and_curve_points(self):
        with tempfile.TemporaryDirectory() as d:
            ev.plot_roc_ovr(self.y, self.P, Path(d) / "roc.png", Path(d) / "roc.csv")
            self.assertTrue((Path(d) / "roc.png").stat().st_size > 1000)
            self.assertEqual(set(pd.read_csv(Path(d) / "roc.csv")["class"]), set(emb.CLASS_NAMES))

    def test_lodo_metrics_flags_small_diseases(self):
        dis = np.array(["x"] * 5 + ["y"] * 25 + ["z"] * 30)
        m, per, oof = ev.lodo_metrics([f"s{i}" for i in range(60)], self.y, self.P, dis)
        self.assertEqual(per.set_index("disease")["flag_small_n"].to_dict(), {"x": True, "y": False, "z": False})
        self.assertIn("macro_roc_auc_pooled", m)
        self.assertEqual(list(oof.columns), ["sample_id", "disease", "true_class"] + emb.CLASS_NAMES)


class Predictions(unittest.TestCase):
    ids = ["sample_001", "sample_002", "sample_003"]

    def test_valid_file_matches_expected_format(self):
        P = np.array([[.85, .10, .05], [.15, .75, .10], [.05, .20, .75]])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.tsv"
            ev.write_predictions(path, self.ids, P, self.ids)
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0], "sample_id\tautoimmunity\tcancer\tinfection")
            self.assertEqual(len(lines), 4)

    def test_reorders_to_metadata_order(self):
        P = np.array([[.2, .3, .5], [.3, .3, .4], [.1, .1, .8]])
        with tempfile.TemporaryDirectory() as d:
            df = ev.write_predictions(Path(d) / "p.tsv", ["sample_003", "sample_001", "sample_002"], P, self.ids)
            self.assertEqual(df["sample_id"].tolist(), self.ids)
            self.assertAlmostEqual(df.iloc[0]["infection"], 0.4)

    def test_rejects_bad_predictions(self):
        good = np.full((3, 3), 1 / 3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.tsv"
            with self.assertRaises(ValueError):                    # rows do not sum to 1
                ev.write_predictions(p, self.ids, good * 0.9, self.ids)
            with self.assertRaises(ValueError):                    # missing sample
                ev.write_predictions(p, self.ids[:2], good[:2], self.ids)
            with self.assertRaises(ValueError):                    # NaN
                bad = good.copy()
                bad[0, 0] = np.nan
                ev.write_predictions(p, self.ids, bad, self.ids)


class Shap(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.names = planted()
        self.booster = mdl.fit_final(self.X, self.y, self.names, mdl.default_xgb_params(), 60)

    def test_additivity_with_output_margin(self):
        self.assertLess(ev.check_additivity(self.booster, self.X, self.names), 1e-4)
        phi, bias = ev.shap_contributions(self.booster, self.X, self.names)
        self.assertEqual(phi.shape, (90, 3, 6))
        self.assertEqual(bias.shape, (90, 3))

    def test_importance_table(self):
        imp = ev.shap_importance(self.booster, self.X, self.y, self.names, cluster_names=["f0", "f1"])
        self.assertEqual(imp["importance_overall"].tolist(), sorted(imp["importance_overall"], reverse=True))
        np.testing.assert_allclose(imp["importance_overall"],
                                   imp[[f"importance_{c}" for c in emb.CLASS_NAMES]].sum(axis=1))
        top = imp.set_index("feature")
        self.assertEqual(top.loc["f0", "top_class"], "cancer")
        self.assertEqual(top.loc["f1", "top_class"], "infection")
        self.assertGreater(top.loc["f0", "direction_cancer"], 0.5)     # higher value -> more cancer
        self.assertEqual(top.loc["f0", "feature_type"], "cluster")
        self.assertEqual(top.loc["f3", "feature_type"], "extra")
        self.assertIn(imp.loc[0, "feature"], ("f0", "f1"))


if __name__ == "__main__":
    unittest.main()
