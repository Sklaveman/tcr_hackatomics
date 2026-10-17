"""3-class XGBoost fitting and tuning (TASK.md 4.3).

Ported from ``utils_new_approach.py``: ``binary:logistic`` + ``scale_pos_weight``
became ``multi:softprob`` + balanced per-sample weights, early stopping is on
``mlogloss``, and every split (outer and inner) is stratified on the 3-class
label. The primary score is the macro one-vs-rest ROC-AUC.
"""
import logging

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

N_CLASSES = 3


def macro_auc(y, proba):
    """Macro one-vs-rest ROC-AUC; NaN if a class is absent from ``y``."""
    if len(np.unique(y)) < N_CLASSES:
        return float("nan")
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro",
                               labels=list(range(N_CLASSES))))


def default_xgb_params(device="cpu", n_train=None, random_state=42):
    """
    A deliberately regularized baseline, for screening clustering parameters.

    min_child_weight is an absolute count of patients, so it has to scale with
    the cohort. At 400 patients a value of 5 is mild; at 30 it prevents the
    trees from splitting at all and the model returns a constant, which looks
    exactly like "the clustering found no signal". Pass `n_train` to have it
    capped at 5% of the training fold.
    """
    mcw = 5 if n_train is None else max(1, min(5, int(0.05 * n_train)))
    return {
        "objective": "multi:softprob", "num_class": N_CLASSES, "eval_metric": "mlogloss",
        "tree_method": "hist", "device": device, "seed": random_state,
        "learning_rate": 0.03, "max_depth": 3, "min_child_weight": mcw,
        "lambda": 1.0, "alpha": 0.1, "subsample": 0.8, "colsample_bytree": 0.5,
    }


def xgb_params_from(hp, device="cpu", random_state=42):
    """Booster params from a ``hyperparameters.json`` ``xgboost`` block."""
    keys = ("learning_rate", "max_depth", "min_child_weight", "lambda", "alpha",
            "subsample", "colsample_bytree")
    p = default_xgb_params(device, random_state=random_state)
    p.update({k: hp[k] for k in keys if k in hp})
    return p


def inner_split(y, inner_val_size=0.25, random_state=42):
    """Stratified (inner train, inner eval) index split of the training fold,
    one fold of a ``StratifiedKFold``. Returns None when a class has too few
    samples to stratify (tiny dry-run folds)."""
    n_splits = max(2, int(round(1.0 / inner_val_size)))
    n_splits = min(n_splits, int(np.bincount(y, minlength=N_CLASSES)[np.unique(y)].min()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return next(iter(skf.split(np.zeros(len(y)), y)))


def _dmatrix(X, y, names):
    return xgb.DMatrix(X, label=y, feature_names=names,
                       weight=compute_sample_weight("balanced", y))


def fit_and_score(X_train, y_train, X_val, y_val, xgb_params, feature_names,
                  num_boost_round=2000, early_stopping_rounds=50,
                  inner_val_size=0.25, random_state=42):
    """
    Train with early stopping on an INNER stratified split of the training
    fold, then predict the held-out fold.

    Early stopping and scoring on the same fold makes the reported AUC
    optimistic, which is how utils.py did it. Separating them costs some
    training data but gives a usable estimate. Balanced per-sample weights are
    applied to the inner train and inner eval sets, so early stopping targets
    the balanced ``mlogloss``.

    If the training fold is too small to stratify an inner split, the model is
    trained for a fixed 100 rounds without early stopping.

    Returns (proba (n_val, 3), n_rounds_used, booster).
    """
    split = inner_split(y_train, inner_val_size, random_state)
    dva = xgb.DMatrix(X_val, feature_names=feature_names)
    if split is None:
        booster = xgb.train(xgb_params, _dmatrix(X_train, y_train, feature_names),
                            num_boost_round=100, verbose_eval=False)
        return booster.predict(dva), 100, booster
    tr, es = split
    booster = xgb.train(xgb_params, _dmatrix(X_train[tr], y_train[tr], feature_names),
                        num_boost_round=num_boost_round,
                        evals=[(_dmatrix(X_train[es], y_train[es], feature_names), "inner")],
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    n_rounds = booster.best_iteration + 1
    return booster.predict(dva, iteration_range=(0, n_rounds)), n_rounds, booster


def fit_final(X, y, feature_names, xgb_params, num_boost_round):
    """Final booster on all training samples, fixed number of rounds."""
    return xgb.train(xgb_params, _dmatrix(X, y, feature_names),
                     num_boost_round=int(num_boost_round), verbose_eval=False)


def predict_proba(booster, X, feature_names):
    """Softmax probabilities ``(n, 3)``; rows sum to 1."""
    return booster.predict(xgb.DMatrix(X, feature_names=feature_names))


def optimize_xgb_params(fold_features, n_trials=60, device="cpu", random_state=42):
    """
    Stage 2: tune XGBoost on features that are already computed.

    `fold_features` is a list of dicts with X_train, y_train, X_val, y_val,
    names, one per fold, produced once with the clustering configuration
    chosen in stage 1. No clustering happens inside the search, so hundreds of
    trials cost minutes instead of days. The objective is the mean macro
    one-vs-rest ROC-AUC over folds.

    Returns (best params incl. num_boost_round, study, trials DataFrame).
    """
    logging.getLogger("optuna").setLevel(logging.WARNING)
    # min_child_weight is a Hessian sum (~0.22 per sample at the start for 3
    # classes), so an upper bound of 20 would forbid every split on a small
    # training fold and the search would see only constant models; cap it at
    # 5% of the smallest training fold (same rule as default_xgb_params).
    mcw_max = max(1, min(20, int(0.05 * min(len(f["y_train"]) for f in fold_features))))

    def objective(trial):
        params = default_xgb_params(device, random_state=random_state)
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, mcw_max),
            "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        })
        scores, iters = [], []
        for f in fold_features:
            proba, n_rounds, _ = fit_and_score(f["X_train"], f["y_train"], f["X_val"],
                                               f["y_val"], params, f["names"],
                                               random_state=random_state)
            scores.append(macro_auc(f["y_val"], proba))
            iters.append(n_rounds)
        trial.set_user_attr("num_boost_round", int(np.mean(iters)))
        trial.set_user_attr("auc_std", float(np.nanstd(scores)))
        return float(np.nanmean(scores))

    sampler = TPESampler(seed=random_state, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = dict(study.best_params)
    best["num_boost_round"] = study.best_trial.user_attrs["num_boost_round"]
    trials = study.trials_dataframe()[["number", "value", "params_learning_rate", "params_max_depth",
                                       "params_min_child_weight", "params_lambda", "params_alpha",
                                       "params_subsample", "params_colsample_bytree",
                                       "user_attrs_num_boost_round", "user_attrs_auc_std"]]
    print(f"Best mean macro AUC: {study.best_value:.4f} "
          f"(sd across folds {study.best_trial.user_attrs['auc_std']:.4f})")
    return best, study, trials
