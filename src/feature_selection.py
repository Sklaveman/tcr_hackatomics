"""Label-based feature selection (TASK.md 4.1 and 4.1b).

* ``fit_sequence_filter``: one-vs-rest one-sided Fisher exact tests on a
  sample x feature presence matrix (clonotypes, CDR3 k-mers, or V/J genes),
  used to decide which training sequences enter the k-NN graph.
* ``select_clusters_by_prevalence``: the cluster-occupancy column filter,
  global (label-free, the original behaviour) or class-aware (default).

Both use labels, so inside cross-validation they must be fitted on the
fold's training samples only (the caller's job; neither function sees
validation data).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom
from sklearn.feature_extraction.text import CountVectorizer

EXTRA_FEATURES = ["unassigned", "evenness", "log_depth"]
FEATURE_TYPES = ("clonotype", "kmer", "vj")


# ---------------------------------------------------------------------------
# Sample x feature presence matrix
# ---------------------------------------------------------------------------

def _concat_keys(sample_keys):
    """All keys as one array, kept as bytes when they are bytes (a str copy of
    tens of millions of S120 keys would be 4x larger)."""
    if not sample_keys:
        return np.array([], dtype=object)
    return np.concatenate([np.asarray(k_) for k_ in sample_keys])


def _decode(uniq):
    """Unique keys (bytes or str) -> object array of str."""
    s = pd.Series(np.asarray(uniq, dtype=object))
    if len(s) and isinstance(s.iloc[0], bytes):
        s = s.str.decode("ascii")
    return s.to_numpy(dtype=object)


def _binarize(m):
    m = m.tocsr()
    m.sum_duplicates()
    m.data[:] = 1
    return m


class VKmerNames:
    """Lazy names ``V|kmer`` of the observed (V gene, k-mer) columns. Building
    millions of f-strings up front costs gigabytes; only the tested/selected
    ones are ever needed, via ``take``/``[]``."""

    def __init__(self, v_names, kmers, columns):
        self.v_names, self.kmers = np.asarray(v_names, dtype=object), np.asarray(kmers, dtype=object)
        self.columns = np.asarray(columns, dtype=np.int64)       # index into the (V x k-mer) grid

    def __len__(self):
        return len(self.columns)

    def take(self, idx):
        c = self.columns[np.asarray(idx, dtype=np.int64)]
        n_k = len(self.kmers)
        return self.v_names[c // n_k] + "|" + self.kmers[c % n_k]

    def __getitem__(self, idx):
        if np.ndim(idx) == 0:
            return self.take([idx])[0]
        return self.take(idx)

    def __iter__(self):
        return iter(self.take(np.arange(len(self))))


def take_names(names, idx):
    """``names[idx]`` as str, for plain arrays and lazy ``VKmerNames`` alike."""
    if isinstance(names, VKmerNames):
        return names.take(idx).astype(str)
    return np.asarray(names, dtype=str)[idx]


def build_presence_matrix(sample_keys, feature_type="clonotype", k=3, with_v=False,
                          trim_start=0, trim_end=0):
    """Binary sample x feature matrix from the samples' ``cdr3_aa+V_J`` keys.

    ``sample_keys`` is a list (one entry per sample, in matrix row order) of
    key arrays. Returns a dict:
      X            csr (n_samples, n_features), 0/1
      names        feature names (clonotype key / k-mer / gene)
      unique_keys  every distinct key seen (str array)
      key_features function: bool mask over ``unique_keys`` -> for a set of
                   selected feature columns, which keys carry one of them
    Vectorized: factorize + COO assembly, no Python loops over sequences
    (CDR3 k-mers come from ``CountVectorizer``).

    k-mer features only: the first ``trim_start`` and last ``trim_end`` residues
    of every CDR3 are cut off before the k-mers are taken, so motifs from the
    conserved ends (e.g. ``CAS...F``) are ignored. A CDR3 too short to leave a
    k-mer carries no feature at all.
    """
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"feature_type must be one of {FEATURE_TYPES}")
    n_samples = len(sample_keys)
    all_keys = _concat_keys(sample_keys)
    rows = np.repeat(np.arange(n_samples), [len(k_) for k_ in sample_keys])
    key_codes, unique_keys = pd.factorize(all_keys)
    del all_keys
    unique_keys = _decode(unique_keys)
    n_keys = len(unique_keys)
    S = _binarize(sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, key_codes)),
                                shape=(n_samples, n_keys)))

    parts = pd.Series(unique_keys).str.split("+", n=1, expand=True) if n_keys else None
    if feature_type == "clonotype":
        return {"X": S, "names": unique_keys, "unique_keys": unique_keys,
                "key_features": lambda sel: np.isin(np.arange(n_keys), sel)}

    cdr3 = parts[0].to_numpy()
    vj = parts[1].str.split("_", n=1, expand=True)
    v, j = vj[0].to_numpy(), vj[1].to_numpy()

    if feature_type == "vj":
        v_codes, v_names = pd.factorize(v)
        j_codes, j_names = pd.factorize(j)
        n_v = len(v_names)
        B = sp.csr_matrix(
            (np.ones(2 * n_keys, dtype=np.int8),
             (np.concatenate([np.arange(n_keys)] * 2),
              np.concatenate([v_codes, n_v + j_codes]))),
            shape=(n_keys, n_v + len(j_names)))
        X = _binarize(S.astype(np.int32) @ B.astype(np.int32))
        names = np.concatenate([np.asarray(v_names, dtype=object), np.asarray(j_names, dtype=object)])
        return {"X": X, "names": names, "unique_keys": unique_keys,
                "key_features": lambda sel: np.asarray(B[:, sel].sum(axis=1)).ravel() > 0}

    # k-mers: a "unit" is a CDR3 (or a CDR3+V pair when with_v)
    cdr3_codes, cdr3_uniq = pd.factorize(cdr3)
    if trim_start or trim_end:
        cdr3_uniq = pd.Series(cdr3_uniq).str.slice(trim_start, -trim_end if trim_end else None).to_numpy()
    cv = CountVectorizer(analyzer="char", ngram_range=(k, k), binary=True, lowercase=False,
                         dtype=np.int8)
    try:
        M_cdr3 = cv.fit_transform(cdr3_uniq).tocsr()
    except ValueError as e:
        raise ValueError(f"no CDR3 is long enough for k={k} after trimming {trim_start} from the start and "
                         f"{trim_end} from the end ({e})") from None
    kmers = np.asarray(cv.get_feature_names_out(), dtype=object)
    if with_v:
        unit_ids = pd.factorize(pd.Series(cdr3_codes).astype(str) + "|" + pd.Series(v).astype(str))[0]
        n_units = unit_ids.max() + 1 if n_keys else 0
        first = np.zeros(n_units, dtype=np.int64)
        first[unit_ids] = np.arange(n_keys)               # any key of the unit
        v_codes, v_names = pd.factorize(v)
        unit_cdr3, unit_v = cdr3_codes[first], v_codes[first]
        base = M_cdr3[unit_cdr3].tocoo()
        n_k = len(kmers)
        M_unit = sp.csr_matrix((base.data, (base.row, base.col + n_k * unit_v[base.row])),
                               shape=(n_units, n_k * len(v_names)))
        used = np.unique(M_unit.indices)                 # (V, k-mer) pairs that occur at all
        M_unit = M_unit[:, used]
        names = VKmerNames(v_names, kmers, used)
    else:
        unit_ids, M_unit, names = cdr3_codes, M_cdr3, kmers
        n_units = M_unit.shape[0]
    S_unit = _binarize(sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, unit_ids[key_codes])),
        shape=(n_samples, n_units)))
    X = _binarize(S_unit.astype(np.int32) @ M_unit.astype(np.int32))

    def key_features(sel):
        unit_hit = np.asarray(M_unit[:, sel].sum(axis=1)).ravel() > 0
        return unit_hit[unit_ids]
    return {"X": X, "names": names, "unique_keys": unique_keys, "key_features": key_features}


# ---------------------------------------------------------------------------
# One-vs-rest Fisher exact tests
# ---------------------------------------------------------------------------

SELECT_MODES = ("fdr", "topk", "percentile")


def fisher_ovr(X, y, n_classes=3, min_samples=3):
    """One-sided (enrichment) Fisher exact test of every feature, per class.

    For class ``c``: 2x2 table (sample carries the feature or not) x (sample is
    in ``c`` or not). With N samples, n_c in the class, k carriers and a of
    them in the class, the one-sided p-value is
    ``hypergeom.sf(a - 1, N, k, n_c)``, identical to
    ``scipy.stats.fisher_exact(table, alternative='greater')``; it is computed
    once per distinct (a, k) pair. Features carried by fewer than
    ``min_samples`` samples are not tested (p = NaN).

    Returns {class: dict(p, a, k, odds_ratio)} of arrays over features.
    """
    y = np.asarray(y)
    N = X.shape[0]
    k = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    tested = (k >= min_samples)
    out = {}
    for c in range(n_classes):
        in_c = y == c
        n_c = int(in_c.sum())
        a = np.asarray(X[in_c].sum(axis=0)).ravel().astype(np.int64)
        p = np.full(len(k), np.nan)
        pair = a[tested] * (N + 1) + k[tested]
        uniq, inv = np.unique(pair, return_inverse=True)
        ua, uk = uniq // (N + 1), uniq % (N + 1)
        p[tested] = hypergeom.sf(ua - 1, N, uk, n_c)[inv]
        # odds ratio with a +0.5 correction (reporting only)
        b, cc, d = n_c - a, k - a, (N - n_c) - (k - a)
        orr = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (cc + 0.5))
        out[c] = {"p": p, "a": a, "k": k, "n_c": n_c, "odds_ratio": orr}
    return out


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values; NaN entries are ignored."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    q[ok] = adj
    return q


def select_features(res, names, select="fdr", alpha=0.05, top_k=1000, top_percent=5.0):
    """Per class, the selected feature columns.

    ``fdr``: BH within the class's family of tests, keep q <= alpha.
    ``topk``: the ``top_k`` smallest p-values.
    ``percentile``: the ``top_percent`` % smallest p-values (equivalently, the
    features above that percentile of -log10 p) among the features TESTED for
    that class, i.e. carried by >= ``min_samples`` training samples. The count
    is ``ceil(top_percent / 100 * n_tested)``, at least 1.
    For ``topk`` and ``percentile``, ties (p-values are discrete) are broken by
    larger in-class carrier fraction, then by feature name, so the result does
    not depend on feature order.
    Returns {class: int array of columns}.
    """
    if select not in SELECT_MODES:
        raise ValueError(f"select must be one of {SELECT_MODES}")
    if select == "percentile" and not (0 < top_percent <= 100):
        raise ValueError("top_percent must be in (0, 100]")
    sel = {}
    for c, r in res.items():
        p = r["p"]
        if select == "fdr":
            q = bh_adjust(p)
            sel[c] = np.where(q <= alpha)[0]
        else:
            idx = np.where(~np.isnan(p))[0]
            n_keep = top_k if select == "topk" else max(1, int(np.ceil(top_percent / 100.0 * len(idx))))
            frac = r["a"][idx] / max(r["n_c"], 1)
            order = np.lexsort((take_names(names, idx), -frac, p[idx]))
            sel[c] = idx[order][:n_keep]
    return sel


def fisher_tests(sample_keys, y, feature_type="clonotype", kmer_k=3, with_v=False,
                 min_samples=3, n_classes=3, trim_start=0, trim_end=0):
    """The expensive part of the sequence filter: presence matrix and one-vs-rest
    Fisher tests on the given (training) samples. Independent of the selection
    rule, so several strictness settings can share it (see ``finish_filter``)."""
    space = build_presence_matrix(sample_keys, feature_type, k=kmer_k, with_v=with_v,
                                  trim_start=trim_start, trim_end=trim_end)
    res = fisher_ovr(space["X"], np.asarray(y), n_classes=n_classes, min_samples=min_samples)
    return {"space": space, "res": res}


def fit_sequence_filter(sample_keys, y, disease=None, feature_type="clonotype", kmer_k=3,
                        with_v=False, min_samples=3, select="fdr", alpha=0.05,
                        top_k=1000, top_percent=5.0, n_classes=3, report_top=200,
                        trim_start=0, trim_end=0):
    """Fit the Fisher sequence filter on the given (training) samples
    (``fisher_tests`` + ``finish_filter``). See ``finish_filter`` for the result."""
    tests = fisher_tests(sample_keys, y, feature_type, kmer_k, with_v, min_samples, n_classes,
                         trim_start, trim_end)
    return finish_filter(tests, y, disease, select, alpha, top_k, top_percent, report_top)


def finish_filter(tests, y, disease=None, select="fdr", alpha=0.05, top_k=1000,
                  top_percent=5.0, report_top=200):
    """Apply one selection rule to precomputed ``fisher_tests``.

    A training sequence survives if it carries at least one feature selected
    for ANY class. Returns a dict:
      survivor_keys   S120 array of surviving keys
      selected        {class: array of selected feature names}
      report          DataFrame of the top selected features per class with
                      carriers, p, q, odds ratio and (if ``disease`` is given)
                      the number of diseases of that class in which the
                      feature occurs (a single-disease feature is a likely
                      study marker)
      stats           per-class counts: tested, selected, sequences kept, and
                      the union (``n_survivors`` of ``n_train_sequences``)
    """
    y = np.asarray(y)
    space, res = tests["space"], tests["res"]
    X, names = space["X"], space["names"]
    sel = select_features(res, names, select=select, alpha=alpha, top_k=top_k, top_percent=top_percent)

    survive = np.zeros(len(space["unique_keys"]), dtype=bool)
    stats = {}
    for c, cols in sel.items():
        kept_c = space["key_features"](cols) if len(cols) else np.zeros_like(survive)
        survive |= kept_c
        stats[int(c)] = {"n_tested": int(np.sum(~np.isnan(res[c]["p"]))),
                         "n_features_selected": int(len(cols)),
                         "n_sequences_kept": int(kept_c.sum())}
    stats["n_train_sequences"] = int(len(survive))
    stats["n_survivors"] = int(survive.sum())

    rows = []
    disease = None if disease is None else np.asarray(disease)
    for c, cols in sel.items():
        q = bh_adjust(res[c]["p"])
        top = cols[np.argsort(res[c]["p"][cols], kind="stable")][:report_top] if len(cols) else cols
        for f in top:
            n_dis = None
            if disease is not None:
                col = X[:, f].toarray().ravel() > 0
                in_c = y == c
                n_dis = int(len({d for d in disease[in_c & col]}))
            rows.append({"class": int(c), "feature": names[f], "carriers_in_class": int(res[c]["a"][f]),
                         "carriers_total": int(res[c]["k"][f]), "p": float(res[c]["p"][f]),
                         "q": float(q[f]), "odds_ratio": float(res[c]["odds_ratio"][f]),
                         "n_diseases_in_class": n_dis})
    return {"survivor_keys": np.asarray(space["unique_keys"][survive]).astype("S120"),
            "selected": {int(c): take_names(names, cols) for c, cols in sel.items()},
            "report": pd.DataFrame(rows), "stats": stats}


# ---------------------------------------------------------------------------
# Cluster prevalence filter (4.1b)
# ---------------------------------------------------------------------------

def select_clusters_by_prevalence(features, y, mode="class_aware", min_prevalence=0.1,
                                  min_count=3, n_classes=3):
    """Cluster-occupancy columns to keep (the extra columns are always kept by
    the caller: ``selected + EXTRA_FEATURES``).

    ``global``: occupied in >= ``min_prevalence`` of all samples (label-free;
    ``y`` is ignored).
    ``class_aware``: occupied in >= ``min_prevalence`` of the samples of at
    least one class, and in at least ``min_count`` samples of that class. It
    keeps every cluster ``global`` keeps (global prevalence is the class-size
    weighted mean of the class prevalences), plus clusters concentrated in
    one class. Uses labels: fit on training samples only.
    """
    if mode not in ("global", "class_aware"):
        raise ValueError("mode must be 'global' or 'class_aware'")
    cols = [c for c in features.columns if c not in EXTRA_FEATURES]
    occ = features[cols].to_numpy() > 0
    if mode == "global":
        keep = occ.mean(axis=0) >= min_prevalence
    else:
        y = np.asarray(y)
        keep = np.zeros(len(cols), dtype=bool)
        for c in range(n_classes):
            in_c = y == c
            n_c = int(in_c.sum())
            if n_c == 0:
                continue
            cnt = occ[in_c].sum(axis=0)
            keep |= (cnt / n_c >= min_prevalence) & (cnt >= min_count)
    return [c for c, k_ in zip(cols, keep) if k_]


def prevalence_report(features, y, disease, selected, min_prevalence=0.1, n_classes=3):
    """Per kept cluster: prevalence per class, the winning class, how many
    diseases of that class are occupied, and whether it is kept ONLY because
    of the class-aware rule (it fails the global rule at the same threshold)."""
    y = np.asarray(y)
    disease = np.asarray(disease)
    occ = features[selected].to_numpy() > 0
    prev = np.column_stack([occ[y == c].mean(axis=0) if (y == c).any() else np.zeros(len(selected))
                            for c in range(n_classes)])
    best = prev.argmax(axis=1)
    n_dis = [len({d for d in disease[(y == b) & occ[:, j]]}) for j, b in enumerate(best)]
    return pd.DataFrame({
        "cluster": selected, "global_prevalence": occ.mean(axis=0),
        **{f"prevalence_class_{c}": prev[:, c] for c in range(n_classes)},
        "best_class": best, "n_diseases_in_best_class": n_dis,
        "only_class_aware": occ.mean(axis=0) < min_prevalence})
