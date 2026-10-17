"""FAISS k-means clustering with per-cluster radii, and the per-sample cluster-occupancy features.

The cluster model is a dict (``cluster_names``, ``n_clusters``, ``keys_sorted`` / ``labels_sorted`` for the
members of a cluster, centroids and radii); ``extract_features`` turns it into the features XGBoost uses:

  fit      k-means on the training sequences of a split (``--kmeans_device cuda`` -> FAISS GPU), K centroids
  radius   for every cluster, the q-th quantile of its training members' distances to the centroid
           (q = 1: the farthest member; q = 0.9: tighter, the outer 10 % of every cluster is cut off)
  assign   nearest centroid; the sequence is unassigned (-1) when its distance exceeds that cluster's radius
  drop     clusters with fewer than ``min_cluster_size`` assigned training members are dropped (-1) and the
           others renumbered 0..K'-1

The nearest centroid and its distance do not depend on q, so several radii share one search: ``assign``
caches them per patient (``cache``/``cache_key``) and a model for another q only re-applies the radii.
Distances: Euclidean, or cosine distance (1 - cos) with ``spherical=True`` (unit-norm embeddings such as
SCEPTR; the vectors are normalised and FAISS runs spherical k-means). The V/J genes are not used here:
they only matter through the embedding itself.

FAISS is imported lazily, so the rest of the pipeline works in environments without it (``sceptr_env`` has it).
"""
import time
import weakref
import zlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.embeddings import get_patient_embeddings, valid_key_mask
from src.feature_selection import EXTRA_FEATURES  # noqa: F401  (re-exported)

_INDEXES = {}          # id(centroids) -> (weakref, {device: faiss index}); never pickled with a model
SEARCH_CHUNK = 262_144


def import_faiss():
    try:
        import faiss
    except ImportError:
        raise SystemExit("ERROR: the clustering needs FAISS (faiss-cpu or faiss-gpu); it is in "
                         "env/sceptr_env_*.yml (conda activate sceptr_env)") from None
    return faiss


def _gpu_index(index):
    """The given CPU index moved to GPU 0 (one GPU is plenty for flat search over <= a few thousand centroids)."""
    faiss = import_faiss()
    return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)


_PROBE = """
import sys
import numpy as np
import faiss
if not (hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0):
    sys.exit("FAISS sees no GPU")
res = faiss.StandardGpuResources()
x = np.random.default_rng(0).random((2000, 16)).astype("float32")
for make in (faiss.IndexFlatL2, faiss.IndexFlatIP):
    g = faiss.index_cpu_to_gpu(res, 0, make(16))
    g.add(x)
    g.search(x[:10], 1)
    clus = faiss.Clustering(16, 8)
    clus.niter = 2
    clus.train(x, faiss.index_cpu_to_gpu(res, 0, make(16)))
print("ok")
"""


def probe_gpu(timeout=180):
    """Try a tiny real FAISS GPU computation (add, search, k-means) in a SEPARATE process: when the GPU
    architecture is not in the FAISS build, FAISS aborts the whole process ("CUDA error 209 no kernel image"),
    which would kill a pipeline run. Returns (ok, message)."""
    import subprocess
    import sys
    try:
        res = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"the GPU probe did not finish within {timeout} s"
    if res.returncode == 0:
        return True, "ok"
    tail = (res.stderr.strip() or res.stdout.strip() or f"exit code {res.returncode}")[-600:]
    return False, f"exit code {res.returncode}: {tail}"


def check_available(device):
    """Fail early and clearly: FAISS missing, or a GPU requested but FAISS cannot use one.

    Caveats: the CPU-only conda build for macOS exposes the GPU API and silently falls back to the CPU, so
    this cannot tell a CPU-only build from a GPU build there (the fit log prints the index type in use); and
    the conda-forge faiss-gpu build only contains kernels for some GPU generations, see the message below.
    """
    import_faiss()
    if device != "cuda":
        return
    ok, msg = probe_gpu()
    if not ok:
        raise SystemExit(
            f"ERROR: --kmeans_device cuda requested but FAISS cannot use the GPU ({msg}).\n"
            f"'no kernel image' (CUDA error 209) means this FAISS build has no code for your GPU's architecture "
            f"(the conda-forge CUDA 12 build covers compute capability 5.3, 6.2, 7.2, 7.5, 8.0, 8.6, 8.9 and 9.0; "
            f"check yours with: nvidia-smi --query-gpu=name,compute_cap --format=csv; the PyPI wheel "
            f"faiss-gpu-cu12 also covers 7.0). Or use --kmeans_device cpu.")


def _prepare(X, spherical):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if spherical:
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norm, 1e-12)
    return X


def _index_for(centroids, spherical, device):
    faiss = import_faiss()
    for k in [k for k, (r, _) in _INDEXES.items() if r() is None]:
        del _INDEXES[k]
    entry = _INDEXES.get(id(centroids))
    if entry is None or entry[0]() is not centroids:
        entry = (weakref.ref(centroids), {})
        _INDEXES[id(centroids)] = entry
    if device not in entry[1]:
        d = centroids.shape[1]
        index = faiss.IndexFlatIP(d) if spherical else faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(centroids, dtype=np.float32))
        if device == "cuda":
            index = _gpu_index(index)
        entry[1][device] = index
    return entry[1][device]


def nearest_centroid(centroids, X, spherical, device="cpu"):
    """(index of the nearest centroid, distance to it) for every row of ``X``; distance is Euclidean, or
    1 - cosine similarity when ``spherical``."""
    index = _index_for(centroids, spherical, device)
    X = _prepare(X, spherical)
    idx = np.empty(len(X), dtype=np.int64)
    dist = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), SEARCH_CHUNK):
        d, i = index.search(X[s:s + SEARCH_CHUNK], 1)
        idx[s:s + SEARCH_CHUNK] = i[:, 0]
        dist[s:s + SEARCH_CHUNK] = (np.maximum(1.0 - d[:, 0], 0.0) if spherical
                                    else np.sqrt(np.maximum(d[:, 0], 0.0)))
    return idx, dist


def fit_centroids(E, k, spherical, device="cpu", niter=25, nredo=1, max_points_per_centroid=256, seed=42):
    """K centroids (k, d) float32. ``max_points_per_centroid`` makes FAISS subsample the training set to about
    that many points per centroid, so fitting time does not grow with the number of sequences."""
    faiss = import_faiss()
    if len(E) < k:
        raise ValueError(f"fewer sequences ({len(E)}) than clusters ({k}) for k-means")
    X = _prepare(E, spherical)
    d = X.shape[1]
    clus = faiss.Clustering(d, int(k))                       # what faiss.Kmeans does, with an explicit index
    clus.niter, clus.nredo, clus.seed = int(niter), int(nredo), int(seed)
    clus.spherical, clus.verbose = bool(spherical), False
    clus.max_points_per_centroid = int(max_points_per_centroid)
    index = faiss.IndexFlatIP(d) if spherical else faiss.IndexFlatL2(d)
    if device == "cuda":
        index = _gpu_index(index)
    clus.train(X, index)
    fit_centroids.last_index_type = type(index).__name__
    return np.ascontiguousarray(faiss.vector_float_to_array(clus.centroids).reshape(int(k), d), dtype=np.float32)


def fit_base(keys, E, k, spherical, device="cpu", niter=25, nredo=1, max_points_per_centroid=256, seed=42,
             log=print):
    """Fit k-means and assign every training sequence once. The result serves any radius quantile
    (``make_model``) without refitting or searching again."""
    t0 = time.time()
    centroids = fit_centroids(E, k, spherical, device, niter, nredo, max_points_per_centroid, seed)
    idx, dist = nearest_centroid(centroids, E, spherical, device)
    log(f"  k-means K={k}: {len(E):,} sequences, fitted + assigned in {time.time() - t0:.1f} s "
        f"({device}; FAISS index {getattr(fit_centroids, 'last_index_type', '?')})")
    return {"centroids": centroids, "spherical": bool(spherical), "keys": np.asarray(keys),
            "train_idx": idx, "train_dist": dist, "k": int(k)}


def radii(train_idx, train_dist, k, q):
    """Per-cluster radius: the smallest distance that covers at least a fraction ``q`` of the cluster's
    training members (q = 1: the farthest member). Empty clusters get radius -1 (nothing is assigned)."""
    if not 0 < q <= 1:
        raise ValueError("the radius quantile must be in (0, 1]")
    order = np.lexsort((train_dist, train_idx))
    d_sorted = train_dist[order]
    counts = np.bincount(train_idx, minlength=k)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.clip(np.ceil(q * counts).astype(np.int64) - 1, 0, np.maximum(counts - 1, 0))
    r = np.full(k, -1.0, dtype=np.float32)
    nonempty = counts > 0
    r[nonempty] = d_sorted[(starts + rank)[nonempty]]
    return r


def make_model(base, q, min_cluster_size):
    """The cluster model for one radius quantile (see the module docstring)."""
    k = base["k"]
    r = radii(base["train_idx"], base["train_dist"], k, q)
    inside = base["train_dist"] <= r[base["train_idx"]]
    counts = np.bincount(base["train_idx"][inside], minlength=k)
    kept = counts >= max(int(min_cluster_size), 1)
    remap = np.full(k, -1, dtype=np.int64)
    remap[kept] = np.arange(int(kept.sum()))
    labels = np.where(inside, remap[base["train_idx"]], -1)
    order = np.argsort(base["keys"])
    n_clusters = int(kept.sum())
    return {
        "cluster_method": "kmeans", "centroids": base["centroids"], "radius": r, "remap": remap,
        "spherical": base["spherical"], "n_clusters": n_clusters,
        "cluster_names": [f"c{i}" for i in range(n_clusters)],
        "keys_sorted": np.ascontiguousarray(base["keys"][order]),
        "labels_sorted": np.ascontiguousarray(labels[order]), "labels_unsorted": labels,
        "kmeans_k": k, "kmeans_radius_q": float(q), "min_cluster_size": int(min_cluster_size),
        "share_assigned_train": float((labels >= 0).mean()),
    }


def assign(model, embeddings, device="cpu", cache=None, cache_key=None):
    """Labels (-1 = unassigned) for the rows of ``embeddings``: nearest centroid, then the cluster's radius.
    ``cache`` keeps the nearest centroid and distance per ``cache_key`` (they do not depend on the radius)."""
    cached = None if cache is None else cache.get(cache_key)
    if cached is not None and cached[0].shape == (len(embeddings),):
        idx, dist = cached
    else:
        idx, dist = nearest_centroid(model["centroids"], embeddings, model["spherical"], device)
        if cache is not None:
            cache[cache_key] = (idx, dist)
    labels = model["remap"][idx]
    labels[dist > model["radius"][idx]] = -1
    return labels


def fit_model(keys, E, k, q, min_cluster_size, spherical, device="cpu", niter=25, nredo=1,
              max_points_per_centroid=256, seed=42, log=print):
    """Fit and build the model for one (K, q) in one go (final fits and CV folds)."""
    base = fit_base(keys, E, k, spherical, device, niter, nredo, max_points_per_centroid, seed, log=log)
    return make_model(base, q, min_cluster_size)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def extract_features(h5_file_path, patient_ids, model, downsample_to=None, random_state=42, verbose=True,
                     device="cpu", neighbor_cache=None):
    """
    Build a patients x features DataFrame.

    Columns: one proportion per kept cluster, then
      unassigned  - share of sequences in no kept cluster
      evenness    - Pielou evenness over the cluster proportions
      log_depth   - log of the patient's distinct-clonotype count

    Proportions are over distinct CDR3+V+J rows, so an expanded clone counts
    once. log_depth is included because repertoire size varies between
    samples and cluster occupancy partly tracks it; if the model leans on this
    column, depth is carrying label information and that is worth knowing.

    ``downsample_to`` randomly keeps at most that many sequences per patient
    (seeded per patient), to mimic the single-cell repertoire size.

    ``device`` is where the nearest-centroid search runs; ``neighbor_cache`` is a
    dict that keeps each patient's nearest centroids and distances, shared by
    models of the same fit that differ only in the radius quantile.
    """
    names = model["cluster_names"] + EXTRA_FEATURES
    K = model["n_clusters"]
    rows = []

    it = tqdm(patient_ids, desc="Extracting features") if verbose else patient_ids
    for pid in it:
        sequences, embeddings = get_patient_embeddings(h5_file_path, pid)
        mask = valid_key_mask(sequences)
        sequences, embeddings = sequences[mask], embeddings[mask]
        if downsample_to is not None and len(sequences) > downsample_to:
            rng = np.random.default_rng([random_state, zlib.crc32(str(pid).encode())])
            keep = np.sort(rng.choice(len(sequences), downsample_to, replace=False))
            sequences, embeddings = sequences[keep], embeddings[keep]
        n_seq = len(sequences)

        if n_seq == 0:
            rows.append(np.zeros(K + 3))
            continue

        labels = assign(model, embeddings, device, cache=neighbor_cache,
                        cache_key=(str(h5_file_path), str(pid)))

        props = np.zeros(K)
        assigned = labels[labels >= 0]
        if len(assigned):
            uniq, counts = np.unique(assigned, return_counts=True)
            props[uniq] = counts / n_seq
        unassigned = 1.0 - props.sum()

        total = props.sum()
        if total > 0:
            p = props / total
            nz = p[p > 0]
            evenness = float(-(nz * np.log(nz)).sum() / np.log(K)) if K > 1 else 0.0
        else:
            evenness = 0.0

        rows.append(np.concatenate([props, [unassigned, evenness, np.log(n_seq)]]))

    return pd.DataFrame(np.array(rows), index=list(patient_ids), columns=names)


def align_features(features, feature_names, table_name="features"):
    """
    Reindex to exactly `feature_names`, by name, and raise on anything missing.
    Train and test matrices are built through this one path, so they cannot
    drift out of column alignment.
    """
    missing = [f for f in feature_names if f not in features.columns]
    if missing:
        raise ValueError(f"{table_name}: {len(missing)} features absent, e.g. {missing[:5]}")
    out = features.loc[:, feature_names].to_numpy(dtype=float)
    if np.isnan(out).any():
        raise ValueError(f"{table_name}: NaN after alignment")
    return out
