#!/usr/bin/env python
"""Task 4: hyperparameter search for ONE sequence-filter setting (SCEPTR embeddings + FAISS k-means).

Run once per --seq_filter, before ``run_pipeline.py``. Two stages on identical
stratified 3-class folds (built once):
  1. clustering grid with a fixed baseline classifier: the number of clusters K x the radius quantile q of
     the FAISS k-means clustering (src/clustering.py); K needs its own fit per fold, extra q values are
     nearly free (centroids and searches are shared);
  2. Optuna over XGBoost on the cached per-fold features of the winning
     clustering (no reclustering inside the search);
then the combined configuration's out-of-fold metrics are recomputed.

Writes, under <output_dir>/tuning/<embedding folder>_<seq_filter>_kmeans/ (embedding folder = sceptr, or
sceptr_<variant>):
  cv_metrics.json/.csv, hyperparameters.json, (with a sequence filter)
  sequence_list.txt = union of surviving training sequences over all folds and
  the final fit (of the winning Fisher setting), for the generators' --sequence_list.
  With --grid_fisher, stage 1 runs once per Fisher strictness value and the best wins.
Per-fold feature matrices are persisted to <data_dir>/features/<embedding folder>/cv/.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    p.add_argument("--seq_filter", choices=["none", "clonotype", "kmer", "vj"], default="none")
    p.add_argument("--data_dir", default=str(root / "data"))
    p.add_argument("--output_dir", default=str(root / "results" / "run"))
    p.add_argument("--xgboost_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--max_samples", type=int, default=None, help="CPU dry run")
    p.add_argument("--exclude_diseases", nargs="+", default=None, metavar="DISEASE",
                   help="drop all training samples of these diseases (names as in metadata.tsv, "
                        "case-insensitive), e.g. Melanoma Ovarian_cancer; the choice is stored in "
                        "hyperparameters.json and reused by run_pipeline.py")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--grid_fisher", type=float, nargs="*", default=None,
                   help="strictness values to tune for the Fisher filter, for the setting of "
                        "--fisher_select (percentile: --fisher_top_percent, topk: --fisher_top_k, "
                        "fdr: --fisher_alpha). Each value re-runs stage 1 on its own survivors, so "
                        "cost grows linearly with the number of values. Off by default (the single "
                        "value of that option is used). The bare flag, without values, turns it on "
                        "with the default grid 1 5 20 (percent; --fisher_select percentile only).")
    p.add_argument("--n_trials", type=int, default=60, help="Optuna trials (stage 2)")
    p.add_argument("--sequence_list_only", action="store_true",
                   help="only fit the sequence filter on the raw repertoires, write "
                        "sequence_list.txt and stop (no embeddings needed)")
    pl.add_embedding_args(p)
    pl.add_filter_args(p)
    pl.add_kmeans_args(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    pl.check_device(args.xgboost_device)
    filters = pl.filters_from_args(args, args.seq_filter)
    cl.check_available(args.kmeans_device)
    method = emb.embedding_name(args.sceptr_variant)
    out = Path(args.output_dir) / "tuning" / f"{method}_{args.seq_filter}_kmeans"
    out.mkdir(parents=True, exist_ok=True)
    features_dir = os.path.join(args.data_dir, "features", method)

    if args.sequence_list_only:
        if args.seq_filter == "none":
            sys.exit("ERROR: --sequence_list_only needs --seq_filter other than none")
        ctx = pl.load_context(args.data_dir, method, args.max_samples,
                              args.random_seed, from_raw=True, exclude_diseases=args.exclude_diseases)
        folds = pl.make_folds(ctx["meta"], args.n_splits, args.random_seed)
        variants = pl.filter_variants(filters, args.grid_fisher)
        _, finals, union = pl.sequence_filter_variants(ctx, folds, variants, source="raw")
        pl.write_sequence_list(out / "sequence_list.txt", union)
        skey = pl.STRENGTH_KEY[filters["fisher_select"]]
        if len(union) == 0:
            print(f"WARNING: the Fisher filter kept no sequence: {pl.NO_SURVIVORS_HINT}")
        pd.concat([fin["report"].assign(**{skey: v[skey]}) for v, fin in zip(variants, finals)]
                  ).to_csv(out / "sequence_filter_report.csv", index=False)
        print(f"Wrote {out / 'sequence_list.txt'} ({len(union):,} sequences)")
        return

    ctx = pl.load_context(args.data_dir, method, args.max_samples, args.random_seed, args.n_jobs,
                          args.xgboost_device, exclude_diseases=args.exclude_diseases,
                          kmeans_device=args.kmeans_device)
    meta = ctx["meta"]
    print(f"{len(meta)} training samples: {meta['disease_group'].value_counts().to_dict()}")
    print(f"Clustering: FAISS k-means (K in {args.grid_kmeans_k}, radius quantile q in "
          f"{args.grid_kmeans_radius_q}); the V/J genes enter only through the embedding")

    # 1. folds, built once and reused by both stages
    folds = pl.make_folds(meta, args.n_splits, args.random_seed)

    # 2.+3. sequence filter per fold (+ final fit), then stage 1 (clustering screen), for each
    #       Fisher strictness value; the best (strictness, clustering) pair wins
    if args.grid_fisher is not None and args.seq_filter == "none":
        print("WARNING: --grid_fisher is ignored with --seq_filter none")
    variants = pl.filter_variants(filters, args.grid_fisher if args.seq_filter != "none" else None)
    skey = pl.STRENGTH_KEY[filters["fisher_select"]]
    if args.seq_filter != "none":
        print("Fisher sequence filter:")
        all_fits, finals, _ = pl.sequence_filter_variants(ctx, folds, variants)
    else:
        all_fits, finals = [[None] * len(folds)], [None]

    if args.seq_filter != "none":
        # a setting that keeps no sequence in some fold (or in the final fit) cannot be evaluated
        usable = []
        for vi, v in enumerate(variants):
            zero = [f"fold {i}" for i, x in enumerate(all_fits[vi]) if x["stats"]["n_survivors"] == 0]
            if finals[vi]["stats"]["n_survivors"] == 0:
                zero.append("final fit")
            if zero:
                print(f"WARNING: Fisher setting {skey} = {v[skey]} keeps no sequence in {', '.join(zero)}: skipped")
            else:
                usable.append(vi)
        if not usable:
            sys.exit(f"ERROR: the Fisher filter kept no sequence in some fold for every setting tried: "
                     f"{pl.NO_SURVIVORS_HINT}")
        variants, all_fits, finals = ([x[i] for i in usable] for x in (variants, all_fits, finals))

    km_params = {"kmeans_niter": args.kmeans_niter, "kmeans_nredo": args.kmeans_nredo,
                 "kmeans_max_points_per_centroid": args.kmeans_max_points_per_centroid,
                 "kmeans_spherical": args.kmeans_spherical}
    boards = []
    for vi, v in enumerate(variants):
        title = f" ({skey} = {v[skey]})" if args.seq_filter != "none" and len(variants) > 1 else ""
        print(f"\n### Stage 1: clustering screen{title}")
        survivors = [None if x is None else x["survivor_keys"] for x in all_fits[vi]]
        b_v, _ = pl.screen_kmeans_params(
            ctx, folds, survivors, v, args.grid_kmeans_k, args.grid_kmeans_radius_q,
            args.min_cluster_size, km_params)
        b_v.insert(0, "variant", vi)
        if args.seq_filter != "none":
            b_v.insert(1, skey, v[skey])
        boards.append(b_v)
    board = (pd.concat(boards, ignore_index=True)
               .sort_values("auc_mean", ascending=False, na_position="last").reset_index(drop=True))
    if board["auc_mean"].isna().all():
        sys.exit("ERROR: no clustering configuration produced a valid macro AUC")
    best = board.iloc[0]
    vbest = int(best["variant"])
    filters, survivors, final_fit = variants[vbest], [None if x is None else x["survivor_keys"]
                                                      for x in all_fits[vbest]], finals[vbest]
    if final_fit is not None:
        pl.write_sequence_list(out / "sequence_list.txt",
                                np.unique(np.concatenate(survivors + [final_fit["survivor_keys"]])))
        final_fit["report"].to_csv(out / "sequence_filter_report.csv", index=False)
        if len(variants) > 1:
            print(f"\nBest Fisher strictness: {skey} = {filters[skey]}")
    cparams = {"cluster_method": "kmeans", "kmeans_k": int(best["kmeans_k"]),
               "kmeans_radius_q": float(best["kmeans_radius_q"]),
               "min_cluster_size": args.min_cluster_size, **km_params}
    print(f"Best clustering: {cparams} (mean macro AUC {best['auc_mean']:.4f})")

    # 4. stage 2: per-fold features once, then Optuna on them
    print("\n### Stage 2: XGBoost search")
    fold_features = pl.build_fold_features(ctx, folds, survivors, cparams, filters,
                                           features_dir=features_dir, n_splits=args.n_splits)
    best_xgb, study, trials = mdl.optimize_xgb_params(fold_features, args.n_trials, args.xgboost_device,
                                                      args.random_seed)

    # 5. closing CV pass with the winning clustering + classifier together
    print("\n### Final CV with the combined configuration")
    params = mdl.xgb_params_from(best_xgb, args.xgboost_device, args.random_seed)
    metrics, oof, y, P = pl.run_cv(ctx, folds, fold_features, params)
    metrics["configuration"] = {"embedding_method": method, "seq_filter": args.seq_filter,
                                "clustering": cparams, "xgboost": best_xgb, "n_splits": args.n_splits}
    metrics["stage1_leaderboard"] = board.to_dict(orient="records")
    metrics["stage2_trials"] = trials.to_dict(orient="records")
    metrics["caveat"] = ("hyperparameters were selected on these same folds, so this number is "
                         "optimistic relative to fresh data")
    ev.write_metrics(out, metrics, name="cv_metrics")
    board.to_csv(out / "stage1_leaderboard.csv", index=False)
    trials.to_csv(out / "stage2_trials.csv", index=False)

    hp = {"embedding_method": method, "seq_filter": args.seq_filter,
          "random_seed": args.random_seed, "n_splits": args.n_splits,
          "clustering": cparams,
          "xgboost": best_xgb, "filters": filters,
          "n_train_samples": int(len(meta)), "max_samples": args.max_samples,
          "exclude_diseases": ctx["exclude_diseases"]}
    emb.write_json(out / "hyperparameters.json", hp)
    print(f"\nWrote {out}/cv_metrics.json and hyperparameters.json ({time.time() - t0:.0f}s)")
    return hp, metrics


if __name__ == "__main__":
    main()
