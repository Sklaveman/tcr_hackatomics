#!/usr/bin/env python
"""Task 4: final-fit / evaluation entry point.

Consumes ONE already-tuned configuration (``hyperparameters.json`` from
``scripts/tune_hyperparameters.py``); it never searches hyperparameters.

  --stage cv           rerun the stratified 3-class CV with the loaded
                       configuration and report the metrics (TASK.md 4.4)
  --stage fit_predict  refit on all of train/, predict test_bulk and test_sc,
                       write SHAP cluster importance (TASK.md 4.5, 4.6)
  --lodo               additionally run leave-one-disease-out evaluation of the
                       same configuration (evaluation only, one k-means fit per held-out disease)
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src import clustering as cl
from src import embeddings as emb
from src import evaluation as ev
from src import model as mdl
from src import pipeline as pl
from src import preprocessing as prep


def parse_args(argv=None):
    root = prep.project_root()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hyperparams_path", required=True)
    p.add_argument("--stage", choices=["cv", "fit_predict"], required=True)
    p.add_argument("--data_dir", default=str(root / "data"))
    p.add_argument("--output_dir", default=str(root / "results" / "run"))
    p.add_argument("--models_dir", default=None, help="default: <project>/models/<output dir name>")
    p.add_argument("--xgboost_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--max_samples", type=int, default=None, help="CPU dry run")
    p.add_argument("--exclude_diseases", nargs="+", default=None, metavar="DISEASE",
                   help="drop all training samples of these diseases. Default: the list stored in "
                        "hyperparameters.json (what the tuning used); giving a different list only "
                        "warns, since the hyperparameters were tuned on other samples")
    pl.add_embedding_args(p)
    pl.add_kmeans_device_arg(p)
    p.add_argument("--lodo", action="store_true", help="also run leave-one-disease-out")
    p.add_argument("--shap_plots", action="store_true", help="SHAP PNGs (needs the shap package)")
    p.add_argument("--sc_without_log_depth", action="store_true",
                   help="fit_predict: also write test_sc predictions from a booster refit without "
                        "log_depth (it differs ~30x between bulk and single-cell)")
    p.add_argument("--downsample_to", default=None,
                   help="CV only: downsample every repertoire to N sequences, or 'sc_median' "
                        "(median test_sc repertoire size), to estimate single-cell-like performance")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    pl.check_device(args.xgboost_device)
    hp = pl.load_hyperparameters(args.hyperparams_path)
    method = emb.embedding_name(args.sceptr_variant)
    if hp.get("embedding_method", method) != method:
        sys.exit(f"ERROR: --sceptr_variant {args.sceptr_variant} (embeddings '{method}') does not match the "
                 f"hyperparameters ({hp['embedding_method']}): they were tuned for a different "
                 f"embedding")
    if hp.get("random_seed") != args.random_seed:
        print(f"WARNING: --random_seed {args.random_seed} differs from the tuned seed "
              f"{hp.get('random_seed')}; CV folds will differ from the tuning folds")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = args.models_dir or str(prep.project_root() / "models" / out.name)
    features_dir = os.path.join(args.data_dir, "features", method)
    filters = {**pl.DEFAULT_FILTERS, **hp.get("filters", {}), "seq_filter": hp["seq_filter"]}
    cparams = pl.cluster_params_of(hp["clustering"])
    cl.check_available(args.kmeans_device)

    tuned_excl = sorted(hp.get("exclude_diseases") or [])
    excl = args.exclude_diseases if args.exclude_diseases is not None else tuned_excl
    ctx = pl.load_context(args.data_dir, method, args.max_samples, args.random_seed, args.n_jobs,
                          args.xgboost_device, exclude_diseases=excl, kmeans_device=args.kmeans_device)
    if ctx["exclude_diseases"] != tuned_excl:
        print(f"WARNING: excluded diseases {ctx['exclude_diseases']} differ from the ones used in "
              f"tuning ({tuned_excl}); the hyperparameters were tuned on other samples")
    if args.downsample_to is not None:
        if args.stage != "cv":
            sys.exit("ERROR: --downsample_to is for --stage cv")
        if args.downsample_to == "sc_median":
            sizes = [len(k) for k in emb.read_sample_keys(
                ctx["test"]["test_sc"]["h5"], ctx["test"]["test_sc"]["meta"]["sample_id"].tolist()).values()]
            ctx["downsample_to"] = int(np.median(sizes))
        else:
            ctx["downsample_to"] = int(args.downsample_to)
        print(f"Downsampling every repertoire to {ctx['downsample_to']} sequences")
    meta = ctx["meta"]
    print(f"{len(meta)} training samples: {meta['disease_group'].value_counts().to_dict()}")
    params = mdl.xgb_params_from(hp["xgboost"], args.xgboost_device, args.random_seed)

    cv_auc = None
    if args.stage == "cv":
        n_splits = hp.get("n_splits", 5)
        folds = pl.make_folds(meta, n_splits, args.random_seed)
        survivors = [None if (f := pl.fit_fold_filter(ctx, fo["train"], fo["y_train"], fo["disease_train"],
                                                      filters)) is None else f["survivor_keys"]
                     for fo in folds]
        fold_features = pl.build_fold_features(ctx, folds, survivors, cparams, filters,
                                               features_dir=features_dir, n_splits=n_splits)
        metrics, oof, y, P = pl.run_cv(ctx, folds, fold_features, params)
        metrics["configuration"] = {k: hp[k] for k in ("embedding_method", "seq_filter", "clustering",
                                                       "xgboost", "n_splits") if k in hp}
        if ctx["downsample_to"]:
            metrics["downsample_to"] = ctx["downsample_to"]
        metrics["limitation"] = ("stratified CV estimates performance on new patients from the same "
                                 "studies/diseases as train; it cannot rule out that the model partly "
                                 "recognizes study rather than disease class (see --lodo)")
        print("\nCross-validation (out-of-fold):")
        ev.write_metrics(out, metrics)
        ev.plot_roc_ovr(y, P, out / "roc_ovr.png", out / "roc_curves.csv")
        oof.to_csv(out / "oof_predictions.tsv", sep="\t", index=False)
        cv_auc = metrics["macro_roc_auc_pooled"]
    else:
        pl.fit_predict(ctx, hp, args.output_dir, models_dir, features_dir=features_dir,
                       shap_plots=args.shap_plots,
                       sc_without_log_depth=args.sc_without_log_depth)

    if args.lodo:
        print("\n### Leave-one-disease-out")
        m, per, oof = pl.run_lodo(ctx, cparams, filters, params)
        lodo_dir = out / "lodo"
        ev.write_metrics(lodo_dir, m)
        per.to_csv(lodo_dir / "per_disease.csv", index=False)
        oof.to_csv(lodo_dir / "oof_predictions.tsv", sep="\t", index=False)
        print(f"Pooled LODO macro AUC {m['macro_roc_auc_pooled']:.4f}"
              + (f" vs stratified CV {cv_auc:.4f}: a large drop means much of the CV score comes "
                 f"from recognizing diseases/studies" if cv_auc is not None else ""))
        small = per[per["flag_small_n"]]["disease"].tolist()
        if small:
            print(f"Noisy per-disease numbers (n < {ev.SMALL_DISEASE_N}): {small}")
    print(f"\nDone in {time.time() - t0:.0f}s; results in {out}")


if __name__ == "__main__":
    main()
