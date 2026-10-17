"""End-to-end CPU micro-run of scripts/tune_hyperparameters.py and run_pipeline.py
on synthetic embedded data with planted signal (tests/synthetic.py).

Covers: both tuning stages and their outputs, reuse of persisted fold features,
the Fisher sequence filter + sequence list, CV reproduction, fit_predict
(validated prediction files, SHAP outputs, saved models that regenerate the
predictions without reclustering), LODO, and the CLI guards.
About 5 minutes on CPU. Run with:  python -m unittest tests.test_pipeline_micro -v
"""
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

if importlib.util.find_spec("faiss") is None:
    raise unittest.SkipTest("faiss is not installed (use sceptr_env)")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import synthetic
from src import clustering as cl
from src import embeddings as emb
from src import evaluation as ev
from src import model as mdl
from src import pipeline as pl

TUNE = ROOT / "scripts" / "tune_hyperparameters.py"
RUN = ROOT / "run_pipeline.py"
SMALL_GRID = ["--n_splits", "3", "--grid_kmeans_k", "20", "--grid_kmeans_radius_q", "1.0",
              "--min_cluster_size", "10", "--n_trials", "3", "--n_jobs", "2"]


def run(script, *args, check=True):
    res = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise AssertionError(f"{script.name} failed ({res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    return res


class PipelineMicro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.data = cls.d / "data"
        cls.info = synthetic.make_synthetic_data(cls.data, n_per_disease=6, n_seq=150, n_bulk=9,
                                                 n_sc=6, n_public=100, p_public=0.75)
        common = ["--data_dir", cls.data]
        # tuning without a sequence filter, then CV with its hyperparameters (reuses persisted features)
        cls.tune_none = run(TUNE, *common, "--seq_filter", "none", "--output_dir", cls.d / "r_none", *SMALL_GRID)
        cls.hp_none = cls.d / "r_none" / "tuning" / "sceptr_none_kmeans" / "hyperparameters.json"
        cls.cv_none = run(RUN, *common, "--hyperparams_path", cls.hp_none, "--stage", "cv",
                          "--output_dir", cls.d / "cv_none", "--n_jobs", "2")
        # tuning with the Fisher clonotype filter, then CV with it
        # only ~300 sequences survive the filter, so clusters must be allowed to be small
        grid = [("5" if a == "10" and SMALL_GRID[i - 1] == "--min_cluster_size" else a)
                for i, a in enumerate(SMALL_GRID)]
        cls.tune_fisher = run(TUNE, *common, "--seq_filter", "clonotype", "--output_dir", cls.d / "r_fisher",
                              *grid)
        cls.hp_fisher = cls.d / "r_fisher" / "tuning" / "sceptr_clonotype_kmeans" / "hyperparameters.json"
        run(RUN, *common, "--hyperparams_path", cls.hp_fisher, "--stage", "cv",
            "--output_dir", cls.d / "cv_fisher", "--n_jobs", "2")
        # final fit + predictions + LODO with the unfiltered configuration
        cls.fp = run(RUN, *common, "--hyperparams_path", cls.hp_none, "--stage", "fit_predict", "--lodo",
                     "--sc_without_log_depth", "--output_dir", cls.d / "fp",
                     "--models_dir", cls.d / "models", "--n_jobs", "2")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tuning_outputs(self):
        t = self.d / "r_none" / "tuning" / "sceptr_none_kmeans"
        hp = json.loads(self.hp_none.read_text())
        self.assertEqual((hp["embedding_method"], hp["seq_filter"], hp["n_splits"]), ("sceptr", "none", 3))
        self.assertEqual(set(hp["clustering"]), {"cluster_method", "kmeans_k", "kmeans_radius_q", "min_cluster_size",
                                                 "kmeans_niter", "kmeans_nredo", "kmeans_max_points_per_centroid",
                                                 "kmeans_spherical"})
        self.assertIn("num_boost_round", hp["xgboost"])
        self.assertEqual(hp["filters"]["prevalence_mode"], "class_aware")
        m = json.loads((t / "cv_metrics.json").read_text())
        self.assertGreater(m["macro_roc_auc_mean"], 0.85)          # planted signal is learnable
        self.assertEqual(len(m["stage1_leaderboard"]), 1)
        self.assertEqual(len(m["stage2_trials"]), 3)
        self.assertTrue((t / "cv_metrics.csv").exists())
        # per-fold features persisted with their configuration
        cv = self.data / "features" / "sceptr" / "cv"
        self.assertTrue((cv / "config.json").exists())
        self.assertTrue(emb.table_exists(str(cv / "fold_0_train")) and emb.table_exists(str(cv / "fold_2_val")))

    def test_cv_stage_reuses_features_and_reproduces_tuning_metrics(self):
        self.assertEqual(self.cv_none.stdout.count("loaded persisted features"), 3)
        a = json.loads((self.d / "r_none" / "tuning" / "sceptr_none_kmeans" / "cv_metrics.json").read_text())
        b = json.loads((self.d / "cv_none" / "metrics.json").read_text())
        self.assertAlmostEqual(a["macro_roc_auc_pooled"], b["macro_roc_auc_pooled"], places=9)
        for f in ("roc_ovr.png", "roc_curves.csv", "oof_predictions.tsv", "metrics.csv"):
            self.assertTrue((self.d / "cv_none" / f).exists(), f)
        self.assertIn("limitation", b)
        oof = pd.read_csv(self.d / "cv_none" / "oof_predictions.tsv", sep="\t")
        self.assertEqual(len(oof), 36)

    def test_fisher_filter_writes_sequence_list_with_the_planted_clonotypes(self):
        t = self.d / "r_fisher" / "tuning" / "sceptr_clonotype_kmeans"
        listed = set((t / "sequence_list.txt").read_text().split())
        planted = {k for ks in self.info["public"].values() for k in ks}
        self.assertGreater(len(listed & planted) / len(planted), 0.9)
        self.assertLess(len(listed), 1500)                          # far fewer than all sequences
        rep = pd.read_csv(t / "sequence_filter_report.csv")
        self.assertTrue((rep["n_diseases_in_class"] == 2).all())    # planted in both diseases of a class
        hp = json.loads(self.hp_fisher.read_text())
        self.assertEqual(hp["seq_filter"], "clonotype")
        self.assertTrue((self.d / "cv_fisher" / "metrics.json").exists())

    def test_fit_predict_outputs_are_valid(self):
        fp = self.d / "fp"
        for ds, n in (("test_bulk", 9), ("test_sc", 6)):
            meta = emb.load_dataset_meta(emb.embedding_dir(self.data, "sceptr", ds))
            ev.validate_predictions_file(fp / f"predictions_{ds}.tsv", meta["sample_id"].tolist())
            self.assertEqual(len(pd.read_csv(fp / f"predictions_{ds}.tsv", sep="\t")), n)
        imp = pd.read_csv(fp / "cluster_importance.csv")
        self.assertEqual(set(imp.loc[imp["feature_type"] == "extra", "feature"]),
                         {"unassigned", "evenness", "log_depth"})
        self.assertEqual(imp["importance_overall"].tolist(), sorted(imp["importance_overall"], reverse=True))
        self.assertFalse((fp / "sequence_importance.tsv").exists())
        self.assertTrue((fp / "prevalence_report.csv").exists())
        self.assertIn("unassigned share (test_sc)", self.fp.stdout)
        fin = self.data / "features" / "sceptr" / "final"
        for name in ("train", "test_bulk", "test_sc"):
            self.assertTrue(emb.table_exists(str(fin / name)), name)

    def test_saved_models_regenerate_predictions_without_reclustering(self):
        models = self.d / "models"
        cfg = json.loads((models / "model_config.json").read_text())
        model = pickle.loads((models / "cluster_model.pkl").read_bytes())
        booster = xgb.Booster()
        booster.load_model(str(models / "booster.json"))
        h5 = emb.h5_path(emb.embedding_dir(self.data, "sceptr", "test_bulk"))
        ids = emb.load_dataset_meta(emb.embedding_dir(self.data, "sceptr", "test_bulk"))["sample_id"].tolist()
        feats = cl.extract_features(h5, ids, model, verbose=False)
        P = mdl.predict_proba(booster, cl.align_features(feats, cfg["feature_names"]), cfg["feature_names"])
        saved = pd.read_csv(self.d / "fp" / "predictions_test_bulk.tsv", sep="\t").set_index("sample_id")
        np.testing.assert_allclose(P, saved.loc[ids, emb.CLASS_NAMES].to_numpy(), atol=2e-6)

    def test_sc_prediction_without_log_depth(self):
        meta = emb.load_dataset_meta(emb.embedding_dir(self.data, "sceptr", "test_sc"))
        ev.validate_predictions_file(self.d / "fp" / "predictions_test_sc_without_log_depth.tsv",
                                     meta["sample_id"].tolist())

    def test_downsampled_cv_matches_single_cell_size(self):
        out = self.d / "cv_down"
        res = run(RUN, "--data_dir", self.data, "--hyperparams_path",
                  self.hp_none, "--stage", "cv", "--downsample_to", "sc_median", "--output_dir", out,
                  "--n_jobs", "2")
        self.assertIn("Downsampling every repertoire to 50 sequences", res.stdout)   # test_sc size here
        self.assertEqual(json.loads((out / "metrics.json").read_text())["downsample_to"], 50)
        feats = emb.load_table(str(self.data / "features" / "sceptr" / "cv" / "fold_0_train"))
        np.testing.assert_allclose(feats["log_depth"], np.log(50))

    def test_lodo_outputs(self):
        lodo = self.d / "fp" / "lodo"
        per = pd.read_csv(lodo / "per_disease.csv")
        self.assertEqual(sorted(per["disease"]), sorted(synthetic.DISEASES["autoimmunity"]
                                                        + synthetic.DISEASES["cancer"]
                                                        + synthetic.DISEASES["infection"]))
        self.assertEqual(len(pd.read_csv(lodo / "oof_predictions.tsv", sep="\t")), 36)
        m = json.loads((lodo / "metrics.json").read_text())
        self.assertIn("macro_roc_auc_pooled", m)
        self.assertIn("Pooled LODO macro AUC", self.fp.stdout)

    def test_embedding_variant_mismatch_is_rejected(self):
        res = run(RUN, "--sceptr_variant", "large", "--data_dir", self.data, "--hyperparams_path",
                  self.hp_none, "--stage", "cv", "--output_dir", self.d / "bad", check=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("does not match", res.stdout + res.stderr)

    def test_old_leiden_hyperparameters_are_rejected(self):
        hp = json.loads(self.hp_none.read_text())
        hp["clustering"] = {"n_neighbors": 10, "trb_weight": 0.25, "target_n_clusters": 20, "min_cluster_size": 10}
        path = self.d / "hp_leiden.json"
        path.write_text(json.dumps(hp))
        res = run(RUN, "--data_dir", self.data, "--hyperparams_path", path, "--stage", "cv",
                  "--output_dir", self.d / "bad3", check=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("not a FAISS k-means configuration", res.stdout + res.stderr)

    def test_cuda_requested_but_unavailable_fails_clearly(self):
        if xgb_has_cuda():
            self.skipTest("this XGBoost build has CUDA")
        res = run(RUN, "--data_dir", self.data, "--hyperparams_path",
                  self.hp_none, "--stage", "cv", "--xgboost_device", "cuda", "--output_dir", self.d / "bad2",
                  check=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("no CUDA support", res.stdout + res.stderr)


def xgb_has_cuda():
    return bool(xgb.build_info().get("USE_CUDA", False))


class FisherGrid(unittest.TestCase):
    """Fisher strictness tuned in stage 1 (--grid_fisher): one screen per value, best one wins."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.data = cls.d / "data"
        synthetic.make_synthetic_data(cls.data, n_per_disease=6, n_seq=150, n_bulk=9, n_sc=6,
                                      n_public=100, p_public=0.75)
        cls.common = ["--data_dir", cls.data]
        cls.res = run(TUNE, *cls.common, "--seq_filter", "clonotype", "--output_dir", cls.d / "r",
                      "--fisher_select", "percentile", "--grid_fisher", "0.05", "10", "--min_samples", "3",
                      "--n_splits", "3", "--grid_kmeans_k", "20", "--grid_kmeans_radius_q", "1.0", "--min_cluster_size", "5", "--n_trials", "2",
                      "--n_jobs", "2")
        cls.t = cls.d / "r" / "tuning" / "sceptr_clonotype_kmeans"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_leaderboard_covers_every_value_and_winner_is_stored(self):
        board = pd.read_csv(self.t / "stage1_leaderboard.csv")
        self.assertEqual(sorted(board["fisher_top_percent"].unique().tolist()), [0.05, 10.0])
        self.assertTrue(board["auc_mean"].is_monotonic_decreasing or board["auc_mean"].isna().any())
        hp = json.loads((self.t / "hyperparameters.json").read_text())
        self.assertEqual(hp["filters"]["fisher_select"], "percentile")
        self.assertEqual(hp["filters"]["fisher_top_percent"], float(board.iloc[0]["fisher_top_percent"]))
        self.assertIn("Best Fisher strictness", self.res.stdout)

    def test_setting_that_keeps_nothing_is_skipped_or_stops_cleanly(self):
        common = ["--seq_filter", "clonotype", "--fisher_select", "fdr", "--n_splits", "3",
                  "--grid_kmeans_k", "20", "--grid_kmeans_radius_q", "1.0",
                  "--min_cluster_size", "5", "--n_trials", "2", "--n_jobs", "2"]
        # every setting keeps nothing -> clear message, no traceback
        bad = run(TUNE, *self.common, *common, "--output_dir", self.d / "z1", "--fisher_alpha", "1e-12",
                  check=False)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("kept no sequence", bad.stderr + bad.stdout)
        self.assertNotIn("Traceback", bad.stderr)
        # a grid with one empty setting: that one is skipped, the other wins
        ok = run(TUNE, *self.common, *common, "--output_dir", self.d / "z2", "--grid_fisher", "1e-12", "0.05")
        self.assertIn("keeps no sequence", ok.stdout)
        hp = json.loads((self.d / "z2" / "tuning" / "sceptr_clonotype_kmeans" / "hyperparameters.json").read_text())
        self.assertEqual(hp["filters"]["fisher_alpha"], 0.05)
        # run_pipeline with a setting that keeps nothing
        hp["filters"]["fisher_alpha"] = 1e-12
        path = self.d / "hp_empty.json"
        path.write_text(json.dumps(hp))
        cv = run(RUN, *self.common, "--hyperparams_path", path, "--stage", "cv", "--output_dir", self.d / "z3",
                 "--n_jobs", "2", check=False)
        self.assertNotEqual(cv.returncode, 0)
        self.assertIn("kept 0 of", cv.stderr + cv.stdout)
        self.assertNotIn("Traceback", cv.stderr)

    def test_cv_stage_reproduces_tuning_metrics_with_the_winning_value(self):
        hp = self.t / "hyperparameters.json"
        run(RUN, *self.common, "--hyperparams_path", hp, "--stage", "cv", "--output_dir", self.d / "cv",
            "--n_jobs", "2")
        a = json.loads((self.t / "cv_metrics.json").read_text())["macro_roc_auc_mean"]
        b = json.loads((self.d / "cv" / "metrics.json").read_text())["macro_roc_auc_mean"]
        self.assertAlmostEqual(a, b, places=6)


class ExcludeDiseases(unittest.TestCase):
    """--exclude_diseases: tuning drops the diseases, stores the choice, run_pipeline reuses it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.data = cls.d / "data"
        synthetic.make_synthetic_data(cls.data, n_per_disease=6, n_seq=150, n_bulk=9, n_sc=6,
                                      n_public=100, p_public=0.75)
        cls.common = ["--data_dir", cls.data]
        cls.res = run(TUNE, *cls.common, "--seq_filter", "none", "--output_dir", cls.d / "r",
                      "--exclude_diseases", "melanoma", "--n_splits", "3", "--grid_kmeans_k", "20",
                      "--grid_kmeans_radius_q", "1.0", "--min_cluster_size", "10",
                      "--n_trials", "2", "--n_jobs", "2")
        cls.t = cls.d / "r" / "tuning" / "sceptr_none_kmeans"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_samples_dropped_and_choice_stored(self):
        hp = json.loads((self.t / "hyperparameters.json").read_text())
        self.assertEqual(hp["exclude_diseases"], ["Melanoma"])
        self.assertEqual(hp["n_train_samples"], 30)             # 6 diseases x 6 samples - 6
        self.assertIn("Excluding diseases ['Melanoma']", self.res.stdout)

    def test_run_pipeline_reuses_the_stored_exclusion_and_warns_on_a_different_one(self):
        hp = self.t / "hyperparameters.json"
        cv = run(RUN, *self.common, "--hyperparams_path", hp, "--stage", "cv", "--output_dir", self.d / "cv",
                 "--n_jobs", "2")
        self.assertIn("30 training samples", cv.stdout)
        self.assertNotIn("WARNING: excluded diseases", cv.stdout)
        a = json.loads((self.t / "cv_metrics.json").read_text())["macro_roc_auc_mean"]
        b = json.loads((self.d / "cv" / "metrics.json").read_text())["macro_roc_auc_mean"]
        self.assertAlmostEqual(a, b, places=6)                   # same samples -> persisted features reused
        other = run(RUN, *self.common, "--hyperparams_path", hp, "--stage", "cv", "--output_dir", self.d / "cv2",
                    "--exclude_diseases", "HIV", "--n_jobs", "2")
        self.assertIn("WARNING: excluded diseases ['HIV'] differ", other.stdout)

    def test_bad_names_fail_cleanly(self):
        bad = run(TUNE, *self.common, "--seq_filter", "none", "--output_dir", self.d / "x",
                  "--exclude_diseases", "Flu", check=False)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("unknown disease", bad.stderr + bad.stdout)
        whole = run(TUNE, *self.common, "--seq_filter", "none", "--output_dir", self.d / "y",
                    "--exclude_diseases", "CMV", "HIV", check=False)
        self.assertNotEqual(whole.returncode, 0)
        self.assertIn("infection", whole.stderr + whole.stdout)


class KMeansGrid(unittest.TestCase):
    """Stage-1 grid (K x radius quantile) end to end: tuning, CV from the stored hyperparameters, final fit + predictions."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.data = cls.d / "data"
        synthetic.make_synthetic_data(cls.data, n_per_disease=6, n_seq=150, n_bulk=9, n_sc=6,
                                      n_public=100, p_public=0.75)
        cls.common = ["--data_dir", cls.data]
        cls.res = run(TUNE, *cls.common, "--seq_filter", "none",
                      "--output_dir", cls.d / "r", "--n_splits", "3", "--grid_kmeans_k", "20", "40",
                      "--grid_kmeans_radius_q", "0.8", "1.0", "--min_cluster_size", "5", "--n_trials", "2",
                      "--n_jobs", "2")
        cls.t = cls.d / "r" / "tuning" / "sceptr_none_kmeans"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_leaderboard_and_stored_hyperparameters(self):
        board = pd.read_csv(self.t / "stage1_leaderboard.csv")
        self.assertEqual(len(board), 4)                                       # 2 K x 2 q
        self.assertEqual(sorted(board["kmeans_k"].unique().tolist()), [20, 40])
        self.assertTrue((board["unassigned_val_mean"].between(0, 1)).all())
        tight = board[board["kmeans_radius_q"] == 0.8]["unassigned_val_mean"].mean()
        loose = board[board["kmeans_radius_q"] == 1.0]["unassigned_val_mean"].mean()
        self.assertGreaterEqual(tight, loose)                                 # a smaller radius leaves more unassigned
        hp = json.loads((self.t / "hyperparameters.json").read_text())
        c = hp["clustering"]
        self.assertEqual(c["cluster_method"], "kmeans")
        self.assertIn(c["kmeans_k"], (20, 40))
        self.assertIn(c["kmeans_radius_q"], (0.8, 1.0))
        self.assertEqual(c["kmeans_spherical"], "auto")                       # resolved to cosine for SCEPTR at run time
        self.assertEqual(hp["embedding_method"], "sceptr")

    def test_cv_stage_reproduces_tuning_and_fit_predict_runs(self):
        hp = self.t / "hyperparameters.json"
        cv = run(RUN, *self.common, "--hyperparams_path", hp, "--stage", "cv", "--output_dir", self.d / "cv",
                 "--n_jobs", "2")
        self.assertIn("loaded persisted features", cv.stdout)
        a = json.loads((self.t / "cv_metrics.json").read_text())["macro_roc_auc_mean"]
        b = json.loads((self.d / "cv" / "metrics.json").read_text())["macro_roc_auc_mean"]
        self.assertAlmostEqual(a, b, places=6)
        run(RUN, *self.common, "--hyperparams_path", hp, "--stage", "fit_predict", "--output_dir", self.d / "fp",
            "--models_dir", self.d / "models", "--n_jobs", "2")
        pred = pd.read_csv(self.d / "fp" / "predictions_test_bulk.tsv", sep="\t")
        self.assertEqual(len(pred), 9)
        with open(self.d / "models" / "cluster_model.pkl", "rb") as f:
            model = pickle.load(f)
        self.assertEqual(model["cluster_method"], "kmeans")
        self.assertTrue((self.d / "fp" / "cluster_importance.csv").exists())


class SequenceListOnly(unittest.TestCase):
    """The sequence filter needs only raw repertoires: no embeddings."""

    def test_writes_sequence_list_from_raw(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT / "tests" / "data" / "train", Path(d) / "raw" / "train")
            res = run(TUNE, "--seq_filter", "vj", "--sequence_list_only",
                      "--data_dir", d, "--output_dir", Path(d) / "out", "--n_splits", "3",
                      "--min_samples", "2", "--fisher_select", "topk", "--fisher_top_k", "3")
            lst = Path(d) / "out" / "tuning" / "sceptr_vj_kmeans" / "sequence_list.txt"
            self.assertTrue(lst.exists())
            lines = lst.read_text().split()
            self.assertGreater(len(lines), 0)
            self.assertTrue(all("+" in l and "_" in l.split("+")[1] for l in lines))
            self.assertIn("Wrote", res.stdout)

    def test_fisher_grid_writes_report_per_value(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT / "tests" / "data" / "train", Path(d) / "raw" / "train")
            run(TUNE, "--seq_filter", "vj", "--sequence_list_only",
                "--data_dir", d, "--output_dir", Path(d) / "out", "--n_splits", "3", "--min_samples", "2",
                "--fisher_select", "percentile", "--grid_fisher", "5", "50")
            tdir = Path(d) / "out" / "tuning" / "sceptr_vj_kmeans"
            rep = pd.read_csv(tdir / "sequence_filter_report.csv")
            self.assertEqual(sorted(rep["fisher_top_percent"].unique().tolist()), [5.0, 50.0])
            n_lines = len((tdir / "sequence_list.txt").read_text().split())
            solo = run(TUNE, "--seq_filter", "vj", "--sequence_list_only",
                       "--data_dir", d, "--output_dir", Path(d) / "out50", "--n_splits", "3", "--min_samples", "2",
                       "--fisher_select", "percentile", "--fisher_top_percent", "50")
            n50 = len((Path(d) / "out50" / "tuning" / "sceptr_vj_kmeans" / "sequence_list.txt").read_text().split())
            self.assertGreaterEqual(n_lines, n50)      # union over the grid contains the loosest setting

    def test_bare_grid_flag_uses_default_percent_grid_only_for_percentile(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(ROOT / "tests" / "data" / "train", Path(d) / "raw" / "train")
            base = [TUNE, "--seq_filter", "vj", "--sequence_list_only",
                    "--data_dir", d, "--n_splits", "3", "--min_samples", "2"]
            run(*base, "--output_dir", Path(d) / "a", "--fisher_select", "percentile", "--grid_fisher")
            rep = pd.read_csv(Path(d) / "a" / "tuning" / "sceptr_vj_kmeans" / "sequence_filter_report.csv")
            self.assertEqual(sorted(rep["fisher_top_percent"].unique().tolist()), [1.0, 5.0, 20.0])
            bad = run(*base, "--output_dir", Path(d) / "b", "--fisher_select", "fdr", "--grid_fisher", check=False)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("explicitly", bad.stderr + bad.stdout)
            run(*base, "--output_dir", Path(d) / "c", "--fisher_select", "fdr")      # off by default: one value
            rep = pd.read_csv(Path(d) / "c" / "tuning" / "sceptr_vj_kmeans" / "sequence_filter_report.csv")
            self.assertLessEqual(rep["fisher_alpha"].nunique(), 1)       # a single setting (the table may be empty)

    def test_requires_a_filter(self):
        with tempfile.TemporaryDirectory() as d:
            res = run(TUNE, "--seq_filter", "none", "--sequence_list_only",
                      "--data_dir", d, "--output_dir", d, check=False)
            self.assertNotEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()
