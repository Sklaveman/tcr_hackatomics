"""Metrics, LODO summaries, prediction files and SHAP importance
(TASK.md 4.4, 4.5, 4.6).

All metrics are computed on out-of-fold predictions; the test sets have no
labels locally.
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.utils.class_weight import compute_sample_weight

from src.embeddings import CLASS_NAMES
from src.model import macro_auc

SMALL_DISEASE_N = 10   # per-disease numbers below this are flagged as noisy


# ---------------------------------------------------------------------------
# 4.4 metrics
# ---------------------------------------------------------------------------

def compute_metrics(y, proba, fold=None, disease=None):
    """The 4.4 metric set on out-of-fold predictions.

    ``fold`` (array of fold ids per sample) gives the per-fold macro AUC
    (mean +- sd); ``disease`` gives per-disease accuracy.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = proba.argmax(axis=1)
    labels = list(range(len(CLASS_NAMES)))
    m = {"n_samples": int(len(y)),
         "macro_roc_auc_pooled": macro_auc(y, proba),
         "roc_auc_ovr": {n: float(roc_auc_score(y == c, proba[:, c]))
                         for c, n in enumerate(CLASS_NAMES)},
         "f1_macro": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
         "f1_weighted": float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)),
         "balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    if fold is not None:
        fold = np.asarray(fold)
        per_fold = np.array([macro_auc(y[fold == f], proba[fold == f]) for f in np.unique(fold)])
        m["macro_roc_auc_per_fold"] = [float(v) for v in per_fold]
        m["macro_roc_auc_mean"] = float(np.nanmean(per_fold))
        m["macro_roc_auc_sd"] = float(np.nanstd(per_fold, ddof=1)) if np.sum(~np.isnan(per_fold)) > 1 else float("nan")
    cm = confusion_matrix(y, pred, labels=labels)
    m["confusion_matrix_counts"] = cm.tolist()
    m["confusion_matrix_row_normalized"] = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist()
    m["confusion_matrix_note"] = "rows = true class, columns = predicted (argmax); order " + ", ".join(CLASS_NAMES)
    if disease is not None:
        d = pd.DataFrame({"disease": np.asarray(disease), "ok": pred == y})
        g = d.groupby("disease")["ok"].agg(["size", "mean"])
        m["per_disease_accuracy"] = {k: {"n": int(r["size"]), "accuracy": float(r["mean"])}
                                     for k, r in g.iterrows()}
    return m


def plot_roc_ovr(y, proba, png_path, csv_path):
    """One-vs-rest ROC curves: PNG plus the curve points as CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = np.asarray(y)
    rows = []
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c, name in enumerate(CLASS_NAMES):
        fpr, tpr, thr = roc_curve(y == c, proba[:, c])
        auc = roc_auc_score(y == c, proba[:, c])
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
        rows.append(pd.DataFrame({"class": name, "fpr": fpr, "tpr": tpr, "threshold": thr}))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("One-vs-rest ROC (out-of-fold)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    pd.concat(rows).to_csv(csv_path, index=False)


def flatten(d, prefix=""):
    """Flat ``{key: value}`` view of a nested metrics dict (for ``metrics.csv``)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list) and v and isinstance(v[0], list):
            for i, row in enumerate(v):
                for j, val in enumerate(row):
                    out[f"{key}[{i}][{j}]"] = val
        elif isinstance(v, list):
            for i, val in enumerate(v):
                out[f"{key}[{i}]"] = val
        else:
            out[key] = v
    return out


def write_metrics(out_dir, metrics, name="metrics"):
    """``<name>.json`` plus a flat ``<name>.csv``; prints the headline numbers."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.Series(flatten(metrics)).rename("value").rename_axis("metric").to_csv(
        os.path.join(out_dir, f"{name}.csv"))
    for k in ("macro_roc_auc_mean", "macro_roc_auc_sd", "macro_roc_auc_pooled",
              "f1_macro", "f1_weighted", "balanced_accuracy"):
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")


# ---------------------------------------------------------------------------
# LODO (4.3)
# ---------------------------------------------------------------------------

def lodo_metrics(sample_ids, y, proba, disease):
    """Pooled out-of-fold LODO evaluation.

    Each held-out fold has one class only, so per-fold AUC is undefined: macro
    AUC is computed on the pooled predictions. Per held-out disease: n,
    accuracy (share predicted as the correct class) and mean predicted
    probability of the true class; diseases with fewer than
    ``SMALL_DISEASE_N`` samples are flagged. Returns (metrics, per_disease, oof).
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    disease = np.asarray(disease)
    pred = proba.argmax(axis=1)
    true_p = proba[np.arange(len(y)), y]
    per = (pd.DataFrame({"disease": disease, "true_class": [CLASS_NAMES[i] for i in y],
                         "correct": pred == y, "p_true": true_p})
           .groupby(["disease", "true_class"])
           .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                mean_p_true_class=("p_true", "mean")).reset_index())
    per["flag_small_n"] = per["n"] < SMALL_DISEASE_N
    metrics = compute_metrics(y, proba)
    metrics["note"] = ("macro AUC is on pooled LODO predictions; held-out folds each contain a "
                       "single class, so per-fold AUC is undefined")
    oof = pd.DataFrame(proba, columns=CLASS_NAMES)
    oof.insert(0, "sample_id", list(sample_ids))
    oof.insert(1, "disease", disease)
    oof.insert(2, "true_class", [CLASS_NAMES[i] for i in y])
    return metrics, per, oof


# ---------------------------------------------------------------------------
# 4.5 prediction files
# ---------------------------------------------------------------------------

def write_predictions(path, sample_ids, proba, expected_ids, tol=1e-4):
    """Write ``sample_id`` + one probability column per class (fixed order),
    after validating against the dataset's ``metadata.tsv`` IDs: exact row
    count, no missing or extra IDs, no NaNs, rows summing to 1."""
    proba = np.asarray(proba, dtype=float)
    ids = list(sample_ids)
    if proba.shape != (len(ids), len(CLASS_NAMES)):
        raise ValueError(f"probability matrix shape {proba.shape} != ({len(ids)}, {len(CLASS_NAMES)})")
    df = pd.DataFrame(proba, columns=CLASS_NAMES)
    df.insert(0, "sample_id", ids)
    df = df.set_index("sample_id").reindex(list(expected_ids)).reset_index()
    validate_predictions_frame(df, expected_ids, tol)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")
    validate_predictions_file(path, expected_ids, tol)
    return df


def validate_predictions_frame(df, expected_ids, tol=1e-4):
    expected_ids = list(expected_ids)
    if list(df.columns) != ["sample_id"] + CLASS_NAMES:
        raise ValueError(f"bad columns {list(df.columns)}")
    if len(df) != len(expected_ids) or list(df["sample_id"]) != expected_ids:
        raise ValueError("sample_id rows do not match metadata.tsv exactly")
    if df.isna().any().any():
        raise ValueError("NaN in predictions (missing sample_id or probability)")
    s = df[CLASS_NAMES].to_numpy().sum(axis=1)
    if not np.allclose(s, 1.0, atol=tol):
        raise ValueError(f"probabilities do not sum to 1 (max deviation {np.abs(s - 1).max():.2e})")


def validate_predictions_file(path, expected_ids, tol=1e-4):
    validate_predictions_frame(pd.read_csv(path, sep="\t", dtype={"sample_id": str}),
                               expected_ids, tol)


# ---------------------------------------------------------------------------
# 4.6 SHAP cluster importance
# ---------------------------------------------------------------------------

def shap_contributions(booster, X, feature_names):
    """Exact TreeSHAP per class: ``(n_samples, n_classes, n_features)`` on the
    raw-margin scale, plus the bias ``(n_samples, n_classes)``."""
    contrib = booster.predict(xgb.DMatrix(X, feature_names=feature_names), pred_contribs=True)
    return contrib[:, :, :-1], contrib[:, :, -1]


def check_additivity(booster, X, feature_names):
    """Max |sum(contributions) + bias - output_margin| (should be ~1e-6)."""
    phi, bias = shap_contributions(booster, X, feature_names)
    margin = booster.predict(xgb.DMatrix(X, feature_names=feature_names), output_margin=True)
    return float(np.abs(phi.sum(axis=2) + bias - margin).max())


def shap_importance(booster, X, y, feature_names, cluster_names):
    """Per-feature SHAP importance table (TASK.md 4.6).

    importance_<class> = weighted mean |SHAP| (balanced per-sample weights, as
    in training); importance_overall = sum over classes; direction_<class> =
    Spearman correlation between the feature value and its SHAP value for that
    class; top_class = class with the largest importance.
    """
    phi, _ = shap_contributions(booster, X, feature_names)
    w = compute_sample_weight("balanced", y)
    imp = (w[:, None, None] * np.abs(phi)).sum(axis=0) / w.sum()          # (classes, features)
    df = pd.DataFrame({
        "feature": feature_names,
        "feature_type": ["cluster" if f in set(cluster_names) else "extra" for f in feature_names],
        "n_samples_occupied": (X > 0).sum(axis=0),
        "importance_overall": imp.sum(axis=0)})
    for c, name in enumerate(CLASS_NAMES):
        df[f"importance_{name}"] = imp[c]
    for c, name in enumerate(CLASS_NAMES):
        d = []
        for j in range(X.shape[1]):
            if np.ptp(X[:, j]) == 0 or np.ptp(phi[:, c, j]) == 0:
                d.append(0.0)
            else:
                d.append(float(spearmanr(X[:, j], phi[:, c, j])[0]))
        df[f"direction_{name}"] = d
    df["top_class"] = [CLASS_NAMES[i] for i in imp.argmax(axis=0)]
    return df.sort_values("importance_overall", ascending=False).reset_index(drop=True)
