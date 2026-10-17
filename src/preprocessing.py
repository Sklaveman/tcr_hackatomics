"""Shared preprocessing (TASK.md section 5).

Used by both embedding generators and by the pipeline, so all of them see
identical sequences. Every step is applied in the order of section 5 and the
rows removed by each step are counted per sample (step 8 of the spec).

Two steps go beyond section 5 and are flagged below; both were found while
building Task 2 and are pending confirmation (see ``drop_missing_vj`` and the
comma-ambiguous J check).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

MAX_KEY_LEN = 120  # embedding keys are stored as S120

# 14 single-member V genes that appear either bare ("TRBV2") or "-1"-suffixed
# ("TRBV2-1") depending on the sample (batch effect, see Task 1 EDA). We keep
# the bare form, in all three datasets.
CANONICAL_V_GENES = ("TRBV1", "TRBV2", "TRBV9", "TRBV13", "TRBV14", "TRBV15",
                     "TRBV16", "TRBV17", "TRBV18", "TRBV19", "TRBV26",
                     "TRBV27", "TRBV28", "TRBV30")
_SUFFIXED_V_GENES = {g + "-1" for g in CANONICAL_V_GENES}

REPERTOIRE_COLUMNS = ["cdr3_aa", "v_gene", "j_gene", "clone_count"]
LOG_COLUMNS = ["n_raw_rows", "n_noncanonical", "n_too_long", "n_vgene_junk",
               "n_ambiguous_vj", "n_missing_vj", "n_zero_clonecount",
               "n_v_renamed", "n_dup_rows_collapsed", "n_clonotypes"]

# Canonical CDR3: starts with C, ends with F/W, only the 20 standard amino
# acids (this also excludes '*' and '_').
_CANONICAL_CDR3 = r"^C[ACDEFGHIKLMNPQRSTVWY]*[FW]$"


def project_root():
    """Project root, resolved from this file (never hard-coded)."""
    return Path(__file__).resolve().parents[1]


def load_metadata(dataset_dir):
    """Read ``metadata.tsv`` of one dataset, dropping the sparse HLA columns."""
    meta = pd.read_csv(os.path.join(dataset_dir, "metadata.tsv"), sep="\t", dtype=str)
    return meta.drop(columns=["MHCI", "MHCII"], errors="ignore")


def load_repertoire(path):
    """Step 1: load one repertoire file, keeping the four modelling columns."""
    return pd.read_csv(path, sep="\t", usecols=REPERTOIRE_COLUMNS,
                       dtype={"cdr3_aa": str, "v_gene": str, "j_gene": str,
                              "clone_count": "int64"})


def preprocess_repertoire(df, max_cdr3_len=None, drop_missing_vj=True):
    """Apply steps 2-7 to one repertoire. Returns (clean_df, log_dict).

    ``max_cdr3_len`` is an optional length cap (step 2); SCEPTR has no known
    length limit, so the generator passes ``None``.

    ``drop_missing_vj`` is NOT in section 5: ~21% of rows in most ``train``
    samples have an empty ``v_gene`` (and a few have an empty ``j_gene``).
    Such rows cannot form a ``cdr3_aa+V_J`` key, so they are dropped like the
    junk/ambiguous V genes. Same for comma-ambiguous J calls (``test_bulk``).
    Counted separately so the effect is visible.
    """
    log = {"n_raw_rows": len(df)}
    df = df[REPERTOIRE_COLUMNS]

    # step 2: non-canonical CDR3s, then (model specific) over-long CDR3s
    ok = df["cdr3_aa"].fillna("").str.match(_CANONICAL_CDR3)
    log["n_noncanonical"] = int((~ok).sum())
    df = df[ok]
    if max_cdr3_len is not None:
        ok = df["cdr3_aa"].str.len() <= max_cdr3_len
    else:
        ok = pd.Series(True, index=df.index)
    log["n_too_long"] = int((~ok).sum())
    df = df[ok]

    v = df["v_gene"].fillna("")
    j = df["j_gene"].fillna("")

    # step 3: V genes with '_' or '+' break the sequence key
    ok = ~(v.str.contains("_", regex=False) | v.str.contains("+", regex=False))
    log["n_vgene_junk"] = int((~ok).sum())
    df, v, j = df[ok], v[ok], j[ok]

    # step 4: ambiguous comma-separated V calls (and J calls, see docstring)
    ok = ~(v.str.contains(",", regex=False) | j.str.contains(",", regex=False))
    log["n_ambiguous_vj"] = int((~ok).sum())
    df, v, j = df[ok], v[ok], j[ok]

    # beyond section 5: rows without a V or J call
    if drop_missing_vj:
        ok = (v != "") & (j != "")
    else:
        ok = pd.Series(True, index=df.index)
    log["n_missing_vj"] = int((~ok).sum())
    df = df[ok]

    # step 5: clone_count <= 0
    ok = df["clone_count"] > 0
    log["n_zero_clonecount"] = int((~ok).sum())
    df = df[ok].copy()

    # step 6: canonicalize the 14 single-member V genes ('-1' stripped)
    renamed = df["v_gene"].isin(_SUFFIXED_V_GENES)
    log["n_v_renamed"] = int(renamed.sum())
    df.loc[renamed, "v_gene"] = df.loc[renamed, "v_gene"].str[:-2]

    # step 7: collapse duplicate cdr3_aa+V+J rows (after step 6, which can
    # newly collide rows that differed only by naming convention)
    n_before = len(df)
    df = df.groupby(["cdr3_aa", "v_gene", "j_gene"], as_index=False,
                    sort=False)["clone_count"].sum()
    log["n_dup_rows_collapsed"] = n_before - len(df)
    log["n_clonotypes"] = len(df)

    df["key"] = make_keys(df)
    assert df["key"].str.len().max() <= MAX_KEY_LEN if len(df) else True
    return df, log


def make_keys(df):
    """Embedding key ``"{cdr3_aa}+{v_gene}_{j_gene}"``."""
    return df["cdr3_aa"] + "+" + df["v_gene"] + "_" + df["j_gene"]


def preprocess_dataset(dataset_dir, sample_ids, **kwargs):
    """Preprocess the given samples of one dataset.

    Returns ({sample_id: clean_df}, log_df) with one log row per sample.
    Sample IDs come from metadata, not from directory listings.
    """
    cleaned, logs = {}, []
    for sid in sample_ids:
        df = load_repertoire(os.path.join(dataset_dir, f"{sid}.tsv"))
        cleaned[sid], log = preprocess_repertoire(df, **kwargs)
        logs.append({"sample_id": sid, **log})
    return cleaned, pd.DataFrame(logs, columns=["sample_id"] + LOG_COLUMNS)


def select_samples(meta, max_samples=None, seed=42):
    """Sample IDs to process. With ``max_samples``, a seeded subset spread as
    evenly as possible over ``disease_group`` (if present), for dry runs."""
    ids = meta["sample_id"].tolist()
    if max_samples is None or max_samples >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    if "disease_group" not in meta.columns:
        return sorted(rng.choice(ids, max_samples, replace=False).tolist())
    groups = [rng.permutation(g["sample_id"].to_numpy()).tolist()
              for _, g in meta.groupby("disease_group")]
    picked = []
    while len(picked) < max_samples and any(groups):
        for g in groups:
            if g and len(picked) < max_samples:
                picked.append(g.pop())
    return sorted(picked)
