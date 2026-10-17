"""Per-fold orchestration shared by ``tune_hyperparameters.py`` and ``run_pipeline.py``.

One split (a set of training samples and one or more evaluation sets) goes
through the same chain everywhere (CV folds, LODO folds, the final fit):

  Fisher sequence filter (optional, labels, fit on the training samples only)
  -> pool the surviving training sequences -> FAISS k-means (centroids + radii)
  -> cluster-occupancy features for the training and evaluation samples
  -> class-aware prevalence filter (labels, fit on the training samples only)
  -> XGBoost.

The label-based steps never see evaluation samples.
"""
import hashlib
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from src import clustering as cl
from src import embeddings as emb
from src import evaluation as ev
from src import feature_selection as fs
from src import model as mdl
from src import preprocessing as prep
from src.feature_selection import EXTRA_FEATURES

TEST_DATASETS = ("test_bulk", "test_sc")

DEFAULT_FILTERS = {
    "seq_filter": "none", "kmer_k": 3, "with_v": False, "kmer_trim_start": 0, "kmer_trim_end": 0,
    "min_samples": 3,
    "fisher_select": "fdr", "fisher_alpha": 0.05, "fisher_top_k": 1000, "fisher_top_percent": 5.0,
    "prevalence_mode": "class_aware", "min_prevalence": 0.1, "min_prevalence_count": 3,
}
# the one setting that sets the Fisher filter's strictness, per selection mode
STRENGTH_KEY = {"fdr": "fisher_alpha", "topk": "fisher_top_k", "percentile": "fisher_top_percent"}
CLUSTER_KEYS = ("min_cluster_size", "cluster_method", "kmeans_k", "kmeans_radius_q", "kmeans_niter",
                "kmeans_nredo", "kmeans_max_points_per_centroid", "kmeans_spherical")
KMEANS_DEFAULTS = {"kmeans_niter": 25, "kmeans_nredo": 1, "kmeans_max_points_per_centroid": 256,
                   "kmeans_spherical": "auto"}


# ---------------------------------------------------------------------------
# Context and folds
# ---------------------------------------------------------------------------

def drop_diseases(meta, exclude):
    """``meta`` without the samples of the diseases in ``exclude`` (matched
    case-insensitively against the ``disease`` column; a name that does not
    occur is an error that lists the available ones). Every class must keep
    samples. Returns (filtered meta, canonical names of the dropped diseases)."""
    if not exclude:
        return meta, []
    avail = sorted(meta["disease"].unique())
    lookup = {d.lower(): d for d in avail}
    names = []
    for e in exclude:
        if e.lower() not in lookup:
            raise SystemExit(f"ERROR: unknown disease '{e}' in --exclude_diseases; available: {', '.join(avail)}")
        names.append(lookup[e.lower()])
    names = sorted(set(names))
    kept = meta[~meta["disease"].isin(names)].reset_index(drop=True)
    for cls in emb.CLASS_NAMES:
        if not (kept["disease_group"] == cls).any():
            raise SystemExit(f"ERROR: excluding {names} removes every '{cls}' sample; "
                             "all three classes are needed")
    return kept, names


def load_context(data_dir, method=emb.EMBEDDING_METHOD, max_samples=None, seed=42, n_jobs=-1,
                 xgboost_device="cpu", downsample_to=None, from_raw=False, exclude_diseases=None,
                 kmeans_device="cpu"):
    """Everything about the embedded datasets one run needs. Sample IDs come
    from each dataset's ``metadata.tsv``; ``max_samples`` gives a seeded subset
    (spread over classes for ``train``) for CPU dry runs.

    ``from_raw=True`` reads only ``data/raw/train/metadata.tsv`` (no
    embeddings needed): used to build the ``--sequence_list`` before anything
    has been embedded.

    ``exclude_diseases``: training diseases to drop before anything else (also
    before the ``max_samples`` subset); the embeddings on disk are untouched.

    ``method`` is the embedding folder under ``<data_dir>/embeddings`` (``emb.embedding_name``).
    """
    if from_raw:
        meta = prep.load_metadata(os.path.join(data_dir, "raw", "train"))
        meta["y"] = meta["disease_group"].map(emb.LABEL_MAP).astype(int)
        train_dir, h5, config = None, None, {}
    else:
        train_dir = emb.embedding_dir(data_dir, method, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"no embeddings at {train_dir}; run the generator first")
        meta = emb.load_dataset_meta(train_dir)
        h5, config = emb.h5_path(train_dir), emb.load_embedding_config(train_dir)
    meta, excluded = drop_diseases(meta, exclude_diseases)
    if excluded:
        print(f"Excluding diseases {excluded}: {len(meta)} training samples remain")
    if max_samples is not None:
        ids = prep.select_samples(meta, max_samples, seed)
        meta = meta[meta["sample_id"].isin(ids)].reset_index(drop=True)
    ctx = {"data_dir": os.fspath(data_dir), "method": method, "seed": seed, "n_jobs": n_jobs,
           "xgboost_device": xgboost_device, "downsample_to": downsample_to,
           "max_samples": max_samples, "exclude_diseases": excluded, "kmeans_device": kmeans_device,
           "train_dir": train_dir, "h5_train": h5, "meta": meta, "config": config, "test": {}}
    if from_raw:
        return ctx
    for ds in TEST_DATASETS:
        d = emb.embedding_dir(data_dir, method, ds)
        if os.path.isdir(d):
            m = emb.load_dataset_meta(d)
            if max_samples is not None and max_samples < len(m):
                m = m[m["sample_id"].isin(prep.select_samples(m, max_samples, seed))].reset_index(drop=True)
            ctx["test"][ds] = {"dir": d, "h5": emb.h5_path(d), "meta": m}
    if config.get("sequence_list"):
        print("WARNING: train embeddings were generated with --sequence_list "
              f"({config['sequence_list']}): train features are computed over the listed "
              "sequences only while test sets are embedded in full, so log_depth/unassigned are "
              "not comparable between them.")
    return ctx


def load_hyperparameters(path):
    with open(path) as f:
        hp = json.load(f)
    for k in ("seq_filter", "clustering", "xgboost"):
        if k not in hp:
            raise ValueError(f"{path}: missing '{k}'")
    if hp["clustering"].get("cluster_method") != "kmeans":
        raise ValueError(f"{path}: not a FAISS k-means configuration (the graph + Leiden clustering was "
                         "removed); tune again with scripts/tune_hyperparameters.py")
    return hp


def make_folds(meta, n_splits, seed):
    """Stratified (3-class) K-fold, built once and reused by every stage and
    configuration so comparisons are paired."""
    ids, y = meta["sample_id"].to_numpy(), meta["y"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [{"fold": i, "train": ids[tr], "val": ids[va], "y_train": y[tr], "y_val": y[va],
             "disease_train": meta["disease"].to_numpy()[tr], "disease_val": meta["disease"].to_numpy()[va]}
            for i, (tr, va) in enumerate(skf.split(ids, y))]


def make_lodo_folds(meta):
    """One fold per ``disease``: hold out every sample of it."""
    ids, y, dis = meta["sample_id"].to_numpy(), meta["y"].to_numpy(), meta["disease"].to_numpy()
    out = []
    for i, (tr, va) in enumerate(LeaveOneGroupOut().split(ids, y, groups=dis)):
        out.append({"fold": i, "held_out": dis[va][0], "train": ids[tr], "val": ids[va],
                    "y_train": y[tr], "y_val": y[va],
                    "disease_train": dis[tr], "disease_val": dis[va]})
    return out


# ---------------------------------------------------------------------------
# Sequence filter and sequence list
# ---------------------------------------------------------------------------

# strictness grid used by a bare ``--grid_fisher`` (no values), per selection mode
DEFAULT_FISHER_GRID = {"percentile": [1.0, 5.0, 20.0]}


def filter_variants(filters, grid=None):
    """One filter dict per strictness value to tune. ``grid`` is a list of values
    for the setting used by ``filters['fisher_select']`` (``STRENGTH_KEY``).
    ``None``: no tuning, just ``filters`` itself. ``[]`` (a bare
    ``--grid_fisher``): ``DEFAULT_FISHER_GRID`` of the mode, an error for modes
    without a default grid."""
    mode = filters["fisher_select"]
    if grid is None:
        return [dict(filters)]
    if len(grid) == 0:
        if mode not in DEFAULT_FISHER_GRID:
            raise SystemExit(f"ERROR: --grid_fisher without values needs --fisher_select "
                             f"{' or '.join(DEFAULT_FISHER_GRID)} (default grid "
                             f"{DEFAULT_FISHER_GRID.get('percentile')}); for '{mode}' give the values explicitly")
        grid = DEFAULT_FISHER_GRID[mode]
    key = STRENGTH_KEY[mode]
    return [{**filters, key: type(filters[key])(v)} for v in grid]


def fit_fold_filters(ctx, train_ids, y_train, disease_train, variants, source="embeddings"):
    """Fisher sequence filter fitted on the given training samples, once per
    strictness variant in ``variants`` (a list of filter dicts that differ only
    in the selection setting). The presence matrix and the Fisher tests, the
    expensive part, are computed once and shared. Returns a list of fit dicts
    (``survivor_keys``, ``report``, ``stats``), or [None] * len(variants) if
    the filter is off. ``source='raw'`` reads the preprocessed raw repertoires,
    which needs no embeddings (used to build the ``--sequence_list`` before
    embedding)."""
    base = variants[0]
    if base["seq_filter"] == "none":
        return [None] * len(variants)
    if source == "raw":
        raw_dir = os.path.join(ctx["data_dir"], "raw", "train")
        cleaned, _ = prep.preprocess_dataset(raw_dir, list(train_ids))
        sample_keys = [cleaned[s]["key"].to_numpy() for s in train_ids]
    else:
        keys = emb.read_sample_keys(ctx["h5_train"], list(train_ids))
        sample_keys = [keys[s] for s in train_ids]
    tests = fs.fisher_tests(sample_keys, y_train, feature_type=base["seq_filter"], kmer_k=base["kmer_k"],
                            with_v=base["with_v"], min_samples=base["min_samples"],
                            trim_start=base["kmer_trim_start"], trim_end=base["kmer_trim_end"])
    return [fs.finish_filter(tests, y_train, disease=disease_train, select=v["fisher_select"],
                             alpha=v["fisher_alpha"], top_k=v["fisher_top_k"],
                             top_percent=v["fisher_top_percent"]) for v in variants]


NO_SURVIVORS_HINT = ("the Fisher filter is too strict for this number of samples. Loosen it "
                     "(--fisher_select percentile or topk, a larger --fisher_alpha, a smaller "
                     "--min_samples), use a coarser feature (--seq_filter kmer or vj), or use more samples")


def fit_fold_filter(ctx, train_ids, y_train, disease_train, filters, source="embeddings"):
    """Single-setting version of ``fit_fold_filters``; None if the filter is off.
    Stops with a clear message if the filter keeps no sequence at all."""
    fit = fit_fold_filters(ctx, train_ids, y_train, disease_train, [filters], source)[0]
    if fit is not None and fit["stats"]["n_survivors"] == 0:
        raise SystemExit(f"ERROR: the Fisher filter kept 0 of {fit['stats']['n_train_sequences']:,} "
                         f"training sequences ({len(train_ids)} training samples): {NO_SURVIVORS_HINT}")
    return fit


def sequence_filter_variants(ctx, folds, variants, source="embeddings", log=print):
    """Fit the Fisher filter on every CV fold and on all of ``train`` (the final
    fit), for each strictness variant. Returns (fits[v][fold], finals[v], union),
    ``union`` being the surviving keys over all variants, folds and final fits
    (the most permissive variant normally contains the others)."""
    fits = [[] for _ in variants]
    for f in folds:
        for vi, fit in enumerate(fit_fold_filters(ctx, f["train"], f["y_train"], f["disease_train"],
                                                  variants, source)):
            fits[vi].append(fit)
            key = STRENGTH_KEY[variants[vi]["fisher_select"]]
            log(f"  fold {f['fold']} [{key}={variants[vi][key]}]: {fit['stats']['n_survivors']:,} of "
                f"{fit['stats']['n_train_sequences']:,} training sequences survive")
    meta = ctx["meta"]
    finals = fit_fold_filters(ctx, meta["sample_id"].to_numpy(), meta["y"].to_numpy(),
                              meta["disease"].to_numpy(), variants, source)
    union = np.unique(np.concatenate([x["survivor_keys"] for vf in fits for x in vf]
                                     + [x["survivor_keys"] for x in finals]))
    for v, fin in zip(variants, finals):
        key = STRENGTH_KEY[v["fisher_select"]]
        log(f"  final fit [{key}={v[key]}]: {fin['stats']['n_survivors']:,} survive")
    log(f"  union over variants, folds + final: {len(union):,} sequences")
    return fits, finals, union


def sequence_filter_union(ctx, folds, filters, source="embeddings", log=print):
    """Single-setting version of ``sequence_filter_variants``:
    (per-fold fits, final fit, union of surviving keys)."""
    fits, finals, union = sequence_filter_variants(ctx, folds, [filters], source, log)
    return fits[0], finals[0], union


def write_sequence_list(path, keys):
    """One ``cdr3_aa+V_J`` key per line, for the generators' ``--sequence_list``."""
    with open(path, "w") as f:
        for k in keys:
            f.write(k.decode() + "\n")


# ---------------------------------------------------------------------------
# One split: cluster -> features -> prevalence
# ---------------------------------------------------------------------------

def cluster_params_of(hp_or_dict):
    return {k: hp_or_dict[k] for k in CLUSTER_KEYS if k in hp_or_dict}


def kmeans_is_spherical(ctx, cluster_params):
    """``auto``: cosine (spherical) k-means for SCEPTR's unit-norm embeddings, Euclidean for any other."""
    s = cluster_params.get("kmeans_spherical", "auto")
    return (ctx["config"].get("method") == "sceptr") if s == "auto" else (s in (True, "yes", "true"))


def kmeans_settings(ctx, cluster_params):
    p = {**KMEANS_DEFAULTS, **{k: v for k, v in cluster_params.items() if v is not None}}
    return {"spherical": kmeans_is_spherical(ctx, p), "niter": p["kmeans_niter"], "nredo": p["kmeans_nredo"],
            "max_points_per_centroid": p["kmeans_max_points_per_centroid"], "seed": ctx["seed"],
            "device": ctx.get("kmeans_device", "cpu")}


def fit_cluster_split(ctx, train_ids, cluster_params, survivors=None, verbose=False):
    keys, E = emb.concat_embeddings(ctx["h5_train"], list(train_ids), keep_keys=survivors)
    if len(keys) == 0:
        raise ValueError("no training sequences to cluster (is the sequence filter too strict?)")
    return cl.fit_model(keys, E, cluster_params["kmeans_k"], cluster_params["kmeans_radius_q"],
                        cluster_params.get("min_cluster_size", 20), log=print if verbose else (lambda *_: None),
                        **kmeans_settings(ctx, cluster_params))


def features_for(ctx, model, h5, ids, verbose=False, neighbor_cache=None):
    return cl.extract_features(h5, list(ids), model, downsample_to=ctx["downsample_to"],
                               random_state=ctx["seed"], verbose=verbose,
                               device=ctx.get("kmeans_device", "cpu"), neighbor_cache=neighbor_cache)


def select_and_align(f_train, y_train, disease_train, evals, filters):
    """Class-aware (or global) prevalence filter fitted on the training
    features only, then the same columns applied to every evaluation table.
    Returns (names, X_train, {name: X_eval}, prevalence report)."""
    sel = fs.select_clusters_by_prevalence(
        f_train, y_train, mode=filters["prevalence_mode"],
        min_prevalence=filters["min_prevalence"], min_count=filters["min_prevalence_count"])
    if not sel:
        raise ValueError("no clusters passed the prevalence filter")
    names = sel + EXTRA_FEATURES
    report = fs.prevalence_report(f_train, y_train, disease_train, sel, filters["min_prevalence"])
    return (names, cl.align_features(f_train, names, "train"),
            {k: cl.align_features(v, names, k) for k, v in evals.items()}, report)


def run_split(ctx, train_ids, y_train, disease_train, evals, cluster_params, filters,
              survivors=None, verbose=False):
    """The whole chain for one split. ``evals``: {name: (h5, ids)}.
    Returns a dict with the model, raw feature tables, selected matrices."""
    model = fit_cluster_split(ctx, train_ids, cluster_params, survivors, verbose)
    f_train = features_for(ctx, model, ctx["h5_train"], train_ids, verbose)
    f_evals = {k: features_for(ctx, model, h5, ids, verbose) for k, (h5, ids) in evals.items()}
    names, X_train, X_evals, report = select_and_align(f_train, y_train, disease_train, f_evals, filters)
    return {"model": model, "f_train": f_train, "f_evals": f_evals, "names": names,
            "X_train": X_train, "X_evals": X_evals, "prevalence_report": report}


# ---------------------------------------------------------------------------
# Stage 1: clustering screen
# ---------------------------------------------------------------------------

def screen_kmeans_params(ctx, folds, survivors, filters, grid_k, grid_q, min_cluster_size,
                         cluster_params, log=print):
    """
    Stage 1: a grid over the number of clusters K and the radius quantile q (see ``src/clustering.py``).

    K needs its own fit (done once per fold and K); every q for that K reuses the fitted centroids, the
    training assignments and the nearest-centroid searches of the sample sequences (``neighbor_cache``),
    so extra radius values are nearly free. The classifier is held fixed (default_xgb_params), and the score is the macro one-vs-rest ROC-AUC on the held-out fold.
    Returns (leaderboard, per-fold results).
    """
    settings = kmeans_settings(ctx, cluster_params)
    rows = []
    for f, surv in zip(folds, survivors):
        log(f"\n=== fold {f['fold'] + 1}/{len(folds)} ===")
        t_fold = time.time()
        keys, E = emb.concat_embeddings(ctx["h5_train"], list(f["train"]), keep_keys=surv)
        for k in grid_k:
            if len(keys) < k:       # a strict sequence filter can leave fewer sequences than clusters
                log(f"  K={k}: only {len(keys):,} sequences in this fold, fewer than K -> grid point failed")
                rows += [{"fold": f["fold"], "kmeans_k": k, "kmeans_radius_q": q, "n_clusters": 0,
                          "unassigned_val": float("nan"), "n_selected": 0, "macro_auc": float("nan"),
                          "best_iteration": 0} for q in grid_q]
                continue
            base = cl.fit_base(keys, E, k, log=log, **settings)
            cache = {}
            for q in grid_q:
                model = cl.make_model(base, q, min_cluster_size)
                f_tr = features_for(ctx, model, ctx["h5_train"], f["train"], neighbor_cache=cache)
                f_va = features_for(ctx, model, ctx["h5_train"], f["val"], neighbor_cache=cache)
                row = {"fold": f["fold"], "kmeans_k": k, "kmeans_radius_q": q, "n_clusters": model["n_clusters"],
                       "unassigned_val": float(f_va["unassigned"].mean())}
                try:
                    names, X_tr, X_ev, _ = select_and_align(f_tr, f["y_train"], f["disease_train"],
                                                            {"val": f_va}, filters)
                except ValueError:      # nothing passed the prevalence filter
                    rows.append({**row, "n_selected": 0, "macro_auc": float("nan"), "best_iteration": 0})
                    continue
                proba, n_rounds, _ = mdl.fit_and_score(
                    X_tr, f["y_train"], X_ev["val"], f["y_val"],
                    mdl.default_xgb_params(ctx["xgboost_device"], n_train=len(f["y_train"]), random_state=ctx["seed"]),
                    names, random_state=ctx["seed"])
                auc = mdl.macro_auc(f["y_val"], proba)
                rows.append({**row, "n_selected": len(names) - len(EXTRA_FEATURES), "macro_auc": auc,
                             "best_iteration": n_rounds})
                log(f"  K={k:5d} q={q:.2f} -> {model['n_clusters']:5d} clusters kept, "
                    f"{len(names) - len(EXTRA_FEATURES):5d} selected, unassigned(val)={row['unassigned_val']:.2f}, "
                    f"macro AUC={auc:.4f}")
        log(f"  fold {f['fold'] + 1} took {time.time() - t_fold:.0f} s")
    df = pd.DataFrame(rows)
    board = (df.groupby(["kmeans_k", "kmeans_radius_q"])
               .agg(auc_mean=("macro_auc", lambda s: s.mean(skipna=False)),   # NaN if any fold failed
                    auc_std=("macro_auc", "std"), n_failed_folds=("macro_auc", lambda s: int(s.isna().sum())),
                    n_clusters_mean=("n_clusters", "mean"), n_selected_mean=("n_selected", "mean"),
                    unassigned_val_mean=("unassigned_val", "mean"), best_iteration_mean=("best_iteration", "mean"))
               .reset_index().sort_values("auc_mean", ascending=False, na_position="last")
               .reset_index(drop=True))
    return board, df


# ---------------------------------------------------------------------------
# Stage 2 inputs: per-fold features, computed (and persisted) once
# ---------------------------------------------------------------------------

def features_config(ctx, cluster_params, filters, n_splits):
    """What a persisted feature set depends on (written next to it as JSON)."""
    return {"embedding_method": ctx["method"], "random_seed": ctx["seed"], "n_splits": n_splits,
            "clustering": {k: cluster_params.get(k) for k in CLUSTER_KEYS},
            "filters": {k: filters[k] for k in DEFAULT_FILTERS},
            "max_samples": ctx["max_samples"], "downsample_to": ctx["downsample_to"],
            "exclude_diseases": ctx["exclude_diseases"],
            "sequence_list": ctx["config"].get("sequence_list"),
            "samples_sha1": hashlib.sha1(",".join(ctx["meta"]["sample_id"]).encode()).hexdigest()}


def _json_equal(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def build_fold_features(ctx, folds, survivors, cluster_params, filters, features_dir=None,
                        n_splits=None, reuse=True, log=print):
    """Per-fold feature matrices, once, for stage 2 and for the CV stage.

    Clustering (and both label-based filters) are refit inside each fold on its
    training samples only. The full cluster tables (before the prevalence
    filter) are persisted to ``<features_dir>/cv/fold_<k>_{train,val}`` with
    the configuration JSON (and the selected columns); a matching persisted set
    is reloaded instead of recomputed when ``reuse``.
    Returns a list of dicts (X_train, y_train, X_val, y_val, names, fold).
    """
    n_splits = n_splits or len(folds)
    cfg = features_config(ctx, cluster_params, filters, n_splits)
    cv_dir = os.path.join(features_dir, "cv") if features_dir else None
    cfg_path = os.path.join(cv_dir, "config.json") if cv_dir else None
    cached = (reuse and cv_dir and os.path.exists(cfg_path)
              and _json_equal(json.load(open(cfg_path))["config"], cfg)
              and all(emb.table_exists(os.path.join(cv_dir, f"fold_{f['fold']}_{s}"))
                      for f in folds for s in ("train", "val")))
    out, selected = [], {}
    for f, surv in zip(folds, survivors):
        base = os.path.join(cv_dir, f"fold_{f['fold']}") if cv_dir else None
        if cached:
            f_tr, f_va = emb.load_table(base + "_train"), emb.load_table(base + "_val")
            log(f"  fold {f['fold']}: loaded persisted features")
        else:
            model = fit_cluster_split(ctx, f["train"], cluster_params, surv)
            f_tr = features_for(ctx, model, ctx["h5_train"], f["train"])
            f_va = features_for(ctx, model, ctx["h5_train"], f["val"])
        try:
            names, X_tr, X_ev, rep = select_and_align(f_tr, f["y_train"], f["disease_train"],
                                                      {"val": f_va}, filters)
        except ValueError as e:
            raise ValueError(f"fold {f['fold']}: {e} (the chosen clustering produced no usable "
                             f"clusters; with a sequence filter try a smaller --min_cluster_size)") from e
        n_clusters = len([c for c in f_tr.columns if c not in EXTRA_FEATURES])
        log(f"  fold {f['fold']}: {n_clusters} clusters, {len(names) - len(EXTRA_FEATURES)} kept "
            f"({int(rep['only_class_aware'].sum())} only by the class-aware rule)")
        selected[f"fold_{f['fold']}"] = {"selected": names, "n_clusters": n_clusters,
                                         "n_only_class_aware": int(rep["only_class_aware"].sum())}
        if cv_dir and not cached:
            os.makedirs(cv_dir, exist_ok=True)
            emb.save_table(f_tr, base + "_train")
            emb.save_table(f_va, base + "_val")
        out.append({"fold": f["fold"], "X_train": X_tr, "y_train": f["y_train"], "X_val": X_ev["val"],
                    "y_val": f["y_val"], "names": names, "prevalence_report": rep})
    if cv_dir and not cached:
        emb.write_json(cfg_path, {"config": cfg, "folds": selected})
    return out


# ---------------------------------------------------------------------------
# CV and LODO evaluation
# ---------------------------------------------------------------------------

def run_cv(ctx, folds, fold_features, xgb_params, log=print):
    """Out-of-fold predictions with the given booster params (inner stratified
    early stopping per fold, as in tuning). Returns (metrics, oof DataFrame)."""
    parts, y_all, fold_all, dis_all, ids_all = [], [], [], [], []
    for f, ff in zip(folds, fold_features):
        proba, n_rounds, _ = mdl.fit_and_score(ff["X_train"], ff["y_train"], ff["X_val"], ff["y_val"],
                                               xgb_params, ff["names"], random_state=ctx["seed"])
        log(f"  fold {f['fold']}: macro AUC {mdl.macro_auc(ff['y_val'], proba):.4f} "
            f"({n_rounds} rounds)")
        parts.append(proba)
        y_all.append(ff["y_val"])
        fold_all.append(np.full(len(ff["y_val"]), f["fold"]))
        dis_all.append(f["disease_val"])
        ids_all.append(f["val"])
    P, y = np.vstack(parts), np.concatenate(y_all)
    fold, dis, ids = np.concatenate(fold_all), np.concatenate(dis_all), np.concatenate(ids_all)
    metrics = ev.compute_metrics(y, P, fold=fold, disease=dis)
    oof = pd.DataFrame(P, columns=emb.CLASS_NAMES)
    oof.insert(0, "sample_id", ids)
    oof.insert(1, "fold", fold)
    oof.insert(2, "disease", dis)
    oof.insert(3, "true_class", [emb.CLASS_NAMES[i] for i in y])
    return metrics, oof, y, P


def run_lodo(ctx, cluster_params, filters, xgb_params, log=print, only_diseases=None):
    """Leave-one-disease-out: the full per-fold chain (sequence filter,
    clustering, prevalence filter, XGBoost with inner early stopping) is refit
    for every held-out disease, with no search. ``only_diseases`` limits the
    held-out diseases (the training sets still contain all the others), to bound
    the run time. Returns the pooled results."""
    folds = make_lodo_folds(ctx["meta"])
    if only_diseases is not None:
        unknown = sorted(set(only_diseases) - {f["held_out"] for f in folds})
        if unknown:
            raise ValueError(f"unknown diseases {unknown}; available: {sorted(f['held_out'] for f in folds)}")
        folds = [f for f in folds if f["held_out"] in set(only_diseases)]
    parts, ys, dis, ids = [], [], [], []
    for k, f in enumerate(folds):
        log(f"\n--- LODO {k + 1}/{len(folds)}: holding out {f['held_out']} "
            f"({len(f['val'])} samples) ---")
        fit = fit_fold_filter(ctx, f["train"], f["y_train"], f["disease_train"], filters)
        surv = None if fit is None else fit["survivor_keys"]
        r = run_split(ctx, f["train"], f["y_train"], f["disease_train"],
                      {"val": (ctx["h5_train"], f["val"])}, cluster_params, filters, surv)
        proba, n_rounds, _ = mdl.fit_and_score(r["X_train"], f["y_train"], r["X_evals"]["val"], f["y_val"],
                                               xgb_params, r["names"], random_state=ctx["seed"])
        log(f"  accuracy {np.mean(proba.argmax(1) == f['y_val']):.3f} ({n_rounds} rounds)")
        parts.append(proba)
        ys.append(f["y_val"])
        dis.append(f["disease_val"])
        ids.append(f["val"])
    return ev.lodo_metrics(np.concatenate(ids), np.concatenate(ys), np.vstack(parts), np.concatenate(dis))


# ---------------------------------------------------------------------------
# Final fit and prediction
# ---------------------------------------------------------------------------

def fit_predict(ctx, hp, output_dir, models_dir, features_dir=None, shap_plots=False,
                sc_without_log_depth=False, log=print):
    """Refit on all of ``train`` and predict ``test_bulk`` / ``test_sc``.

    Writes ``predictions_<set>.tsv`` (validated against each dataset's
    ``metadata.tsv``), ``cluster_importance.csv`` (SHAP, TASK.md 4.6; the sequences
    of a cluster are in the saved cluster model), the final feature tables, and the fitted cluster model
    and booster to ``models_dir``.
    """
    meta = ctx["meta"]
    filters = {**DEFAULT_FILTERS, **hp.get("filters", {}), "seq_filter": hp["seq_filter"]}
    cparams = cluster_params_of(hp["clustering"])
    train_ids, y, dis = meta["sample_id"].to_numpy(), meta["y"].to_numpy(), meta["disease"].to_numpy()

    fit = fit_fold_filter(ctx, train_ids, y, dis, filters)
    survivors = None if fit is None else fit["survivor_keys"]
    if fit is not None:
        fit["report"].to_csv(os.path.join(output_dir, "sequence_filter_report.csv"), index=False)
        log(f"Sequence filter: {fit['stats']['n_survivors']:,} of {fit['stats']['n_train_sequences']:,} "
            f"training sequences survive")
    evals = {k: (v["h5"], v["meta"]["sample_id"].tolist()) for k, v in ctx["test"].items()}
    r = run_split(ctx, train_ids, y, dis, evals, cparams, filters, survivors, verbose=True)
    names = r["names"]
    log(f"Clusters kept: {len(names) - len(EXTRA_FEATURES)} / {r['model']['n_clusters']}")
    r["prevalence_report"].to_csv(os.path.join(output_dir, "prevalence_report.csv"), index=False)

    params = mdl.xgb_params_from(hp["xgboost"], ctx["xgboost_device"], ctx["seed"])
    booster = mdl.fit_final(r["X_train"], y, names, params, hp["xgboost"]["num_boost_round"])

    for name, X in r["X_evals"].items():
        P = mdl.predict_proba(booster, X, names)
        path = os.path.join(output_dir, f"predictions_{name}.tsv")
        ev.write_predictions(path, ctx["test"][name]["meta"]["sample_id"].tolist(), P,
                             ctx["test"][name]["meta"]["sample_id"].tolist())
        log(f"Wrote {path} ({len(P)} samples)")
    if sc_without_log_depth and "test_sc" in r["X_evals"]:
        # log_depth differs ~30x between bulk and single-cell: a second booster without it
        keep = [i for i, n in enumerate(names) if n != "log_depth"]
        names2 = [names[i] for i in keep]
        booster2 = mdl.fit_final(r["X_train"][:, keep], y, names2, params, hp["xgboost"]["num_boost_round"])
        P2 = mdl.predict_proba(booster2, r["X_evals"]["test_sc"][:, keep], names2)
        ids_sc = ctx["test"]["test_sc"]["meta"]["sample_id"].tolist()
        path = os.path.join(output_dir, "predictions_test_sc_without_log_depth.tsv")
        ev.write_predictions(path, ids_sc, P2, ids_sc)
        log(f"Wrote {path} (booster refit without log_depth)")
    if "test_sc" in r["f_evals"]:
        for label, tbl in (("train", r["f_train"]), ("test_sc", r["f_evals"]["test_sc"])):
            log(f"  unassigned share ({label}): mean {tbl['unassigned'].mean():.3f}")

    imp = ev.shap_importance(booster, r["X_train"], y, names, r["model"]["cluster_names"])
    imp.to_csv(os.path.join(output_dir, "cluster_importance.csv"), index=False)
    log(f"SHAP importance: top feature {imp.loc[0, 'feature']} "
        f"(overall {imp.loc[0, 'importance_overall']:.4f})")
    if shap_plots:
        plot_shap(booster, r["X_train"], names, output_dir)

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "cluster_model.pkl"), "wb") as f:
        pickle.dump(r["model"], f)
    booster.save_model(os.path.join(models_dir, "booster.json"))
    emb.write_json(os.path.join(models_dir, "model_config.json"),
                   {"feature_names": names, "hyperparameters": hp, "classes": emb.CLASS_NAMES})
    if features_dir:
        fin = os.path.join(features_dir, "final")
        os.makedirs(fin, exist_ok=True)
        emb.save_table(r["f_train"], os.path.join(fin, "train"))
        for k, v in r["f_evals"].items():
            emb.save_table(v, os.path.join(fin, k))
        emb.write_json(os.path.join(fin, "config.json"),
                       {"config": features_config(ctx, cparams, filters, hp.get("n_splits")),
                        "selected": names})
    return r, booster, imp


def plot_shap(booster, X, names, output_dir):
    """Optional beeswarm per class; needs the ``shap`` package."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("--shap_plots: the 'shap' package is not installed; skipping plots")
        return
    phi, _ = ev.shap_contributions(booster, X, names)
    for c, cname in enumerate(emb.CLASS_NAMES):
        shap.summary_plot(phi[:, c, :], X, feature_names=names, max_display=20, show=False)
        plt.title(f"SHAP: {cname}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_{cname}.png"), dpi=120)
        plt.close()


# ---------------------------------------------------------------------------
# CLI helpers shared by the two scripts
# ---------------------------------------------------------------------------

def add_filter_args(p):
    """Filter settings of TASK.md 4.1 / 4.1b."""
    d = DEFAULT_FILTERS
    p.add_argument("--prevalence_mode", choices=["class_aware", "global"], default=d["prevalence_mode"])
    p.add_argument("--min_prevalence", type=float, default=d["min_prevalence"])
    p.add_argument("--min_prevalence_count", type=int, default=d["min_prevalence_count"])
    p.add_argument("--min_samples", type=int, default=d["min_samples"],
                   help="Fisher filter: drop features present in fewer training samples")
    p.add_argument("--fisher_select", choices=list(fs.SELECT_MODES), default=d["fisher_select"],
                   help="fdr: BH q <= --fisher_alpha; topk: --fisher_top_k smallest p per class; "
                        "percentile: the --fisher_top_percent %% smallest p per class")
    p.add_argument("--fisher_alpha", type=float, default=d["fisher_alpha"])
    p.add_argument("--fisher_top_k", type=int, default=d["fisher_top_k"])
    p.add_argument("--fisher_top_percent", type=float, default=d["fisher_top_percent"])
    p.add_argument("--kmer_k", type=int, default=d["kmer_k"])
    p.add_argument("--with_v", action="store_true", help="k-mer features combined with the V gene")
    p.add_argument("--kmer_trim_start", type=int, default=d["kmer_trim_start"],
                   help="k-mer filter: cut this many residues off the start of every CDR3 first")
    p.add_argument("--kmer_trim_end", type=int, default=d["kmer_trim_end"],
                   help="k-mer filter: cut this many residues off the end of every CDR3 first")


def filters_from_args(args, seq_filter):
    return {"seq_filter": seq_filter, "kmer_k": args.kmer_k, "with_v": args.with_v,
            "kmer_trim_start": args.kmer_trim_start, "kmer_trim_end": args.kmer_trim_end,
            "min_samples": args.min_samples, "fisher_select": args.fisher_select,
            "fisher_alpha": args.fisher_alpha, "fisher_top_k": args.fisher_top_k,
            "fisher_top_percent": args.fisher_top_percent,
            "prevalence_mode": args.prevalence_mode, "min_prevalence": args.min_prevalence,
            "min_prevalence_count": args.min_prevalence_count}


def add_kmeans_args(p):
    """FAISS k-means settings (tuning script) and the device of the k-means search."""
    add_kmeans_device_arg(p)
    p.add_argument("--grid_kmeans_k", type=int, nargs="+", default=[300, 1000, 3000],
                   help="stage 1: numbers of clusters K to try")
    p.add_argument("--grid_kmeans_radius_q", type=float, nargs="+", default=[0.8, 0.9, 1.0],
                   help="stage 1: radius quantiles q in (0, 1]: a cluster's radius is the q-th "
                        "quantile of its training members' distances to the centroid; sequences farther from "
                        "their nearest centroid than that radius are unassigned (1.0 = the farthest member)")
    p.add_argument("--min_cluster_size", type=int, default=20,
                   help="clusters with fewer ASSIGNED training sequences (inside their radius) are dropped")
    p.add_argument("--kmeans_niter", type=int, default=KMEANS_DEFAULTS["kmeans_niter"])
    p.add_argument("--kmeans_nredo", type=int, default=KMEANS_DEFAULTS["kmeans_nredo"],
                   help="independent k-means runs; the best (lowest inertia) is kept")
    p.add_argument("--kmeans_max_points_per_centroid", type=int,
                   default=KMEANS_DEFAULTS["kmeans_max_points_per_centroid"],
                   help="FAISS subsamples the training set to about this many points per centroid")
    p.add_argument("--kmeans_spherical", choices=["auto", "yes", "no"], default=KMEANS_DEFAULTS["kmeans_spherical"],
                   help="cosine distance on normalised vectors; auto = yes for SCEPTR embeddings")


def add_kmeans_device_arg(p):
    p.add_argument("--kmeans_device", choices=["cpu", "cuda"], default="cpu",
                   help="where FAISS k-means fits and searches. cpu is the default; cuda needs faiss-gpu (the "
                        "PyPI wheel faiss-gpu-cu12 works on V100; the conda-forge build lacks kernels for some "
                        "GPU generations: CUDA error 209). Independent of --xgboost_device")


def add_embedding_args(p):
    p.add_argument("--sceptr_variant", default="default",
                   help="SCEPTR variant the embeddings were made with: reads data/embeddings/sceptr "
                        "(default) or data/embeddings/sceptr_<variant>")


def check_device(device):
    """``cuda`` must be requested explicitly and must exist (XGBoost built with CUDA)."""
    if device == "cuda":
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA", False):
            raise SystemExit("ERROR: --xgboost_device cuda requested but this XGBoost build has no CUDA support")
