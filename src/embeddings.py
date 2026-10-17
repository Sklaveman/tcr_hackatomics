"""Embedding output format (TASK.md, "Embedding output format (Tasks 2 and 3)").

Writer and validators used by the SCEPTR generator, plus the readers the
pipeline needs. The format is the one ``utils_new_approach.py`` uses:

- one CSV per sample, ``<sample_id>_embed.csv``;
- index: key ``"{cdr3_aa}+{v_gene}_{j_gene}"``, unique per sample, <= 120 chars;
- columns: the embedding dimensions, then a trailing ``vj_call`` column
  (the pipeline drops it with ``values[:, :-1]``);
- ``per_patient_tcr_embed.h5`` per dataset, ``metadata.tsv`` copied unchanged.

The embedding dimension is inferred from the files (columns - 1), never
hard-coded.
"""
import json
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

KEY_DTYPE = "S120"  # keys kept as bytes, ~4x less memory than U120
H5_NAME = "per_patient_tcr_embed.h5"


def write_sample_embeddings(output_dir, sample_id, keys, vj_calls, embeddings):
    """Write ``<sample_id>_embed.csv``: key index, dims 0..d-1, then ``vj_call``."""
    df = pd.DataFrame(np.asarray(embeddings, dtype=np.float32),
                      index=pd.Index(keys, name=None))
    df["vj_call"] = np.asarray(vj_calls)
    path = os.path.join(output_dir, f"{sample_id}_embed.csv")
    df.to_csv(path, float_format="%.6g")
    return path


H5_DTYPES = ("float32", "float16")
FLOAT16_MAX = 60000.0       # float16 overflows to inf above 65504


def _as_h5_dtype(values, dtype):
    """``values`` as ``dtype``; float16 is refused when it would overflow (SCEPTR vectors are unit length)."""
    if dtype not in H5_DTYPES:
        raise ValueError(f"h5 dtype must be one of {H5_DTYPES}, got {dtype!r}")
    values = np.asarray(values)
    if dtype == "float16" and values.size and np.abs(values).max() > FLOAT16_MAX:
        raise ValueError("embedding values too large for float16 (max abs > 60000): use float32")
    return values.astype(dtype)


def save_patient_embeddings(folder_path, output_h5_path, patient_ids, dtype="float32"):
    """Store each patient's keys and embeddings (``dtype``: float32 or float16) under its own HDF5 group."""
    with h5py.File(output_h5_path, "w") as h5f:
        for pid in tqdm(patient_ids, desc="Building HDF5"):
            df = pd.read_csv(os.path.join(folder_path, f"{pid}_embed.csv"), index_col=0)
            grp = h5f.create_group(f"patients/{pid}")
            grp.create_dataset("tcr_sequences", data=df.index.values.astype("S"),
                               compression="gzip")
            grp.create_dataset("embeddings", data=_as_h5_dtype(df.values[:, :-1].astype(np.float32), dtype),
                               compression="gzip")


def h5_dtype_of(h5_file):
    """Storage dtype of the embeddings in an HDF5 file ('float32' / 'float16'), from its first patient."""
    with h5py.File(h5_file, "r") as f:
        for pid in f["patients"]:
            return str(f[f"patients/{pid}/embeddings"].dtype)
    return None


def convert_h5_dtype(src, dst, dtype):
    """Copy ``src`` to ``dst`` with the embeddings stored as ``dtype`` (patient by patient, so memory stays at one
    sample). Keys are copied unchanged."""
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for pid in tqdm(list(fin["patients"]), desc=f"Converting to {dtype}"):
            grp = fout.create_group(f"patients/{pid}")
            grp.create_dataset("tcr_sequences", data=fin[f"patients/{pid}/tcr_sequences"][:], compression="gzip")
            grp.create_dataset("embeddings", data=_as_h5_dtype(fin[f"patients/{pid}/embeddings"][:], dtype),
                               compression="gzip")


def remove_sample_csvs(output_dir):
    """Delete the per-sample ``*_embed.csv`` files of one dataset folder; returns how many."""
    n = 0
    for f in os.listdir(output_dir):
        if f.endswith("_embed.csv"):
            os.remove(os.path.join(output_dir, f))
            n += 1
    return n


def get_patient_embeddings(h5_file_path, patient_id):
    """Return (sequence keys as bytes, embeddings) for one patient."""
    with h5py.File(h5_file_path, "r") as h5f:
        grp = h5f[f"/patients/{patient_id}"]
        return grp["tcr_sequences"][:], grp["embeddings"][:].astype(np.float32, copy=False)


def valid_key_mask(sequences):
    """Keys without a '+' carry no V/J annotation."""
    return np.array([b"+" in s for s in sequences], dtype=bool)


def validate_embedding_dir(output_dir, sample_ids, expected_dim=None):
    """Validate on write (TASK.md): sample counts, dims + 1 columns, unique
    keys, no NaNs, loads via ``get_patient_embeddings``, ``valid_key_mask`` all
    true. Raises ``ValueError`` on the first problem; returns a summary dict.
    """
    h5_path = os.path.join(output_dir, H5_NAME)
    found = sorted(f[:-len("_embed.csv")] for f in os.listdir(output_dir)
                   if f.endswith("_embed.csv"))
    if found != sorted(sample_ids):
        raise ValueError(f"sample count mismatch: {len(found)} files vs "
                         f"{len(sample_ids)} expected samples")
    n_keys, dim = 0, expected_dim
    for sid in sample_ids:
        df = pd.read_csv(os.path.join(output_dir, f"{sid}_embed.csv"), index_col=0)
        d = df.shape[1] - 1
        if dim is None:
            dim = d
        if d != dim or df.columns[-1] != "vj_call":
            raise ValueError(f"{sid}: expected {dim} dims + vj_call, got {d} + "
                             f"last column {df.columns[-1]!r}")
        if not df.index.is_unique:
            raise ValueError(f"{sid}: duplicate keys")
        if df.isna().any().any():
            raise ValueError(f"{sid}: NaN values")
        if df.index.str.len().max() > 120:
            raise ValueError(f"{sid}: key longer than 120 characters")

        keys, emb = get_patient_embeddings(h5_path, sid)
        if len(keys) != len(df) or emb.shape != (len(df), dim):
            raise ValueError(f"{sid}: HDF5 shape disagrees with CSV")
        if not valid_key_mask(keys).all():
            raise ValueError(f"{sid}: keys without '+'")
        n_keys += len(df)
    return {"n_samples": len(sample_ids), "embedding_dim": dim, "n_keys": n_keys}


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Used by the SCEPTR generator and finalize_embeddings.py: sequence bookkeeping and the
# dataset-level writer, so every embedding folder has exactly the same format.
# ---------------------------------------------------------------------------

def read_sequence_list(path):
    """Keys (``cdr3_aa+V_J``), one per line, of the training sequences to embed."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def apply_sequence_list(cleaned, keys):
    """Keep only listed clonotypes in every sample (in place). Returns the
    number of rows dropped because they were not in the list."""
    n_dropped = 0
    for sid, df in cleaned.items():
        mask = df["key"].isin(keys)
        n_dropped += int((~mask).sum())
        cleaned[sid] = df[mask].reset_index(drop=True)
    return n_dropped


def unique_sequences(cleaned):
    """One row per unique ``cdr3_aa+V_J`` key over all samples of a dataset."""
    cols = ["key", "cdr3_aa", "v_gene", "j_gene"]
    return (pd.concat([df[cols] for df in cleaned.values()], ignore_index=True)
            .drop_duplicates("key").reset_index(drop=True))


def write_dataset_embeddings(out_dir, cleaned, uniq, z, meta, log, config, h5_dtype="float32"):
    """Write the full output of one dataset and validate it.

    ``uniq`` are the unique keys (``unique_sequences``) and ``z`` their
    embeddings, row-aligned. Writes per-sample CSVs, the per-patient HDF5,
    ``metadata.tsv``, ``preprocessing_log.tsv`` and ``embedding_config.json``
    (which records ``h5_dtype``). Returns the validation summary.
    """
    out_dir = os.fspath(out_dir)
    index = pd.Index(uniq["key"])
    vj = (uniq["v_gene"] + "_" + uniq["j_gene"]).to_numpy()
    for sid, df in cleaned.items():
        rows = index.get_indexer(df["key"])
        write_sample_embeddings(out_dir, sid, df["key"].to_numpy(), vj[rows], z[rows])
    save_patient_embeddings(out_dir, os.path.join(out_dir, H5_NAME), list(cleaned), dtype=h5_dtype)
    meta.to_csv(os.path.join(out_dir, "metadata.tsv"), sep="\t", index=False)
    log.to_csv(os.path.join(out_dir, "preprocessing_log.tsv"), sep="\t", index=False)
    summary = validate_embedding_dir(out_dir, list(cleaned), expected_dim=z.shape[1])
    write_json(os.path.join(out_dir, "embedding_config.json"),
               {**config, "latent_dim": z.shape[1], "h5_dtype": h5_dtype, **summary})
    return summary


# ---------------------------------------------------------------------------
# Ingestion for the pipeline (Task 4.2): works for either embedding method
# ---------------------------------------------------------------------------

CLASS_NAMES = ["autoimmunity", "cancer", "infection"]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}  # fixed, used everywhere


EMBEDDING_METHOD = "sceptr"


def embedding_name(variant="default"):
    """Folder name under ``embeddings/``: ``sceptr``, or ``sceptr_<variant>`` for another SCEPTR variant."""
    return EMBEDDING_METHOD if variant in (None, "default") else f"{EMBEDDING_METHOD}_{variant}"


def embedding_dir(data_dir, method, dataset):
    """``<data_dir>/embeddings/<method>/<dataset>`` (mirrors ``data/raw``); ``method`` = ``embedding_name(...)``."""
    return os.path.join(os.fspath(data_dir), "embeddings", method, dataset)


def load_embedding_config(emb_dir):
    with open(os.path.join(emb_dir, "embedding_config.json")) as f:
        return json.load(f)


def load_dataset_meta(emb_dir):
    """``metadata.tsv`` of one embedded dataset (sample IDs come from here,
    never from directory listings). Adds integer label ``y`` when the dataset
    is labelled."""
    meta = pd.read_csv(os.path.join(emb_dir, "metadata.tsv"), sep="\t", dtype=str)
    meta = meta.drop(columns=["MHCI", "MHCII"], errors="ignore")
    if "disease_group" in meta.columns:
        unknown = set(meta["disease_group"]) - set(LABEL_MAP)
        if unknown:
            raise ValueError(f"unknown disease_group values: {unknown}")
        meta["y"] = meta["disease_group"].map(LABEL_MAP).astype(int)
    return meta


def h5_path(emb_dir):
    return os.path.join(emb_dir, H5_NAME)


def infer_embedding_dim(emb_dir, sample_id):
    """Embedding dimension from the data itself (never hard-coded)."""
    _, e = get_patient_embeddings(h5_path(emb_dir), sample_id)
    return e.shape[1]


def _hash_keys(keys):
    return pd.util.hash_array(keys.astype(object))


def concat_embeddings(h5_file, patient_ids, keep_keys=None):
    """Pool the unique sequences of the given patients.

    Returns ``(keys, embeddings)``: ``keys`` as ``S120``, ``embeddings`` float32
    ``(n_unique, dim)``. Keys without a '+' are dropped. With ``keep_keys``
    (an array of keys) only those sequences are pooled (the Fisher filter's
    survivors). Two passes with 64-bit key hashes keep memory at one row per
    unique sequence instead of one per (patient, sequence); a hash collision
    (about 1e-5 chance over 25M sequences) would drop one distinct sequence.
    """
    keep = None if keep_keys is None else np.sort(np.asarray(keep_keys, dtype=KEY_DTYPE))

    def read(pid, f):
        grp = f[f"/patients/{pid}"]
        keys = grp["tcr_sequences"][:]
        mask = valid_key_mask(keys)
        if keep is not None and len(keep):
            pos = np.clip(np.searchsorted(keep, keys), 0, len(keep) - 1)
            mask &= keep[pos] == keys            # membership without re-sorting `keep` per patient
        elif keep is not None:
            mask[:] = False
        return grp, mask

    hashes, sizes = [], []
    with h5py.File(h5_file, "r") as f:
        for pid in patient_ids:
            grp, mask = read(pid, f)
            hashes.append(_hash_keys(grp["tcr_sequences"][:][mask]))
            sizes.append(int(mask.sum()))
    all_hashes = np.concatenate(hashes) if hashes else np.empty(0, dtype=np.uint64)
    is_first = np.zeros(len(all_hashes), dtype=bool)
    is_first[np.unique(all_hashes, return_index=True)[1]] = True
    del hashes, all_hashes

    keys_out, emb_out, start = [], [], 0
    with h5py.File(h5_file, "r") as f:
        for pid, n in zip(patient_ids, sizes):
            grp, mask = read(pid, f)
            first = is_first[start:start + n]
            start += n
            keys_out.append(grp["tcr_sequences"][:][mask][first])
            emb_out.append(grp["embeddings"][:][mask][first].astype(np.float32))
    if not keys_out:
        return np.empty(0, dtype=KEY_DTYPE), np.empty((0, 0), dtype=np.float32)
    return (np.concatenate(keys_out).astype(KEY_DTYPE), np.vstack(emb_out))


def read_sample_keys(h5_file, patient_ids):
    """``{sample_id: array of S120 keys}`` (only keys with V/J)."""
    out = {}
    with h5py.File(h5_file, "r") as f:
        for pid in patient_ids:
            keys = f[f"/patients/{pid}/tcr_sequences"][:]
            out[pid] = keys[valid_key_mask(keys)]
    return out


def save_table(df, path_no_ext):
    """Parquet when ``pyarrow`` is available, else ``.csv.gz`` (TASK.md 4.2).
    Returns the written path."""
    try:
        import pyarrow  # noqa: F401
        path = path_no_ext + ".parquet"
        df.to_parquet(path)
    except ImportError:
        path = path_no_ext + ".csv.gz"
        df.to_csv(path)
    return path


def load_table(path_no_ext):
    if os.path.exists(path_no_ext + ".parquet"):
        return pd.read_parquet(path_no_ext + ".parquet")
    return pd.read_csv(path_no_ext + ".csv.gz", index_col=0)


def table_exists(path_no_ext):
    return os.path.exists(path_no_ext + ".parquet") or os.path.exists(path_no_ext + ".csv.gz")
