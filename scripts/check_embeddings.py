#!/usr/bin/env python
"""Check generated embedding folders: were all samples and all sequences processed?

For every <data_dir>/embeddings/<folder>/<dataset> folder (or the ones selected) it reports problems at three
levels. Run it in any environment that has pandas, numpy and h5py (e.g. sceptr_env).

1. Dataset level (always)
   * embedding_config.json exists (the generator writes it last: no file = unfinished or interrupted run);
   * the samples in the raw metadata.tsv (cut to the same --max_samples subset the run used) are all present as
     <sample_id>_embed.csv, as groups of per_patient_tcr_embed.h5, in the embedded metadata.tsv and in
     preprocessing_log.tsv, with nothing extra.
2. Sample level (always)
   * the CSV has the key index, <dim> embedding columns and a trailing 'vj_call' column, unique keys that contain
     '+', and the same number of rows as the HDF5 group; <dim> equals the dimension in the config.
   * --check_values additionally loads every HDF5 embedding and looks for NaN/inf and all-zero rows.
3. Sequence level (--check_sequences; reads the raw repertoires, so it takes a few minutes on the full data)
   * the raw repertoire is preprocessed again exactly as the generator did (same filters, --sequence_list for train) and the expected keys are compared with the keys in the CSV:
       missing  expected key not in the CSV. 'Explained' when the config says its gene was dropped on purpose
                (the V genes SCEPTR rejects). Unexplained missing keys are
                problems.
       extra    key in the CSV that the raw data does not give (always a problem).

Usage:
  python scripts/check_embeddings.py                                  # everything under <project>/data/embeddings
  python scripts/check_embeddings.py --folders sceptr --datasets train --check_sequences
  python scripts/check_embeddings.py --data_dir data/tutorial --raw_dir data/tutorial/raw --check_sequences

Flags:
  --data_dir PATH      folder that contains embeddings/                                  [<project>/data]
  --raw_dir PATH       folder with the raw datasets (<raw_dir>/<dataset>)                [<data_dir>/raw]
  --folders F [F ...]  embedding folders under embeddings/ (sceptr, sceptr_<variant>)    [all that exist]
  --datasets D [D ...] dataset folders to check                                          [all that exist]
  --check_values       load every embedding from the HDF5 and look for NaN/inf/all-zero rows
  --check_sequences    compare the CSV keys with the re-preprocessed raw repertoires
  --max_examples N     example keys / samples listed per problem                          [5]
  --report PATH        also write all findings as JSON
A dataset whose CSVs were deleted (config: csv_removed) is checked from the HDF5 alone.
Exit code 0 when no problem was found, 1 otherwise.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd

from src import embeddings as emb
from src import preprocessing as prep


def parse_args(argv=None):
    root = prep.project_root()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=str(root / "data"))
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--folders", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--check_values", action="store_true")
    p.add_argument("--check_sequences", action="store_true")
    p.add_argument("--max_examples", type=int, default=5)
    p.add_argument("--report", default=None)
    return p.parse_args(argv)


def ids_of_tsv(path):
    return None if not os.path.exists(path) else set(pd.read_csv(path, sep="\t", dtype=str)["sample_id"])


def compare(name, expected, found, problems, k):
    """Record missing/extra sample ids of one source."""
    if found is None:
        problems.append(f"{name}: not found")
        return
    missing, extra = sorted(expected - found), sorted(found - expected)
    if missing:
        problems.append(f"{name}: {len(missing)} of {len(expected)} samples missing, e.g. {missing[:k]}")
    if extra:
        problems.append(f"{name}: {len(extra)} unexpected samples, e.g. {extra[:k]}")


def explained_genes(cfg):
    """V/J gene names the generator dropped on purpose (read from its config)."""
    genes = set()
    for key in ("unseen_genes_rows", "invalid_v_genes"):
        genes |= set((cfg.get(key) or {}).keys())
    return genes


def check_dataset(method, dataset, d, raw_dir, args):
    res = {"method": method, "dataset": dataset, "path": d, "problems": [], "notes": [], "n_samples_expected": None,
           "n_samples_found": None, "n_keys": 0, "missing_explained": 0, "missing_unexplained": 0, "extra": 0}
    prob, k = res["problems"], args.max_examples
    cfg_path = os.path.join(d, "embedding_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if not cfg:
        prob.append("no embedding_config.json: the dataset is unfinished or the run was interrupted")

    # ---- 1. samples
    raw = os.path.join(raw_dir, dataset)
    expected = None
    if os.path.exists(os.path.join(raw, "metadata.tsv")):
        meta = prep.load_metadata(raw)
        ids = prep.select_samples(meta, cfg.get("max_samples"), cfg.get("random_seed", 42))
        expected = set(ids)
    else:
        res["notes"].append(f"no raw metadata at {raw}: expected samples taken from the embedded metadata.tsv")
        expected = ids_of_tsv(os.path.join(d, "metadata.tsv"))
    if expected is None:
        prob.append("neither raw nor embedded metadata.tsv found; cannot tell which samples are expected")
        return res
    res["n_samples_expected"] = len(expected)
    csv_removed = bool(cfg.get("csv_removed"))          # set when the CSVs were deleted after a successful check
    csv_ids = {f[:-len("_embed.csv")] for f in os.listdir(d) if f.endswith("_embed.csv")}
    partial = [f for f in os.listdir(d) if f.endswith(".tmp")]
    if partial:
        prob.append(f"leftover temporary files: {partial[:k]}")
    if not csv_removed:
        compare("per-sample CSV files", expected, csv_ids, prob, k)
    h5_file = os.path.join(d, emb.H5_NAME)
    h5_ids = None
    if os.path.exists(h5_file):
        try:
            with h5py.File(h5_file, "r") as h:
                h5_ids = set(h["patients"].keys()) if "patients" in h else set()
        except Exception as e:            # a file cut off while it was being written (interrupted run)
            prob.append(f"{emb.H5_NAME} is corrupt or incomplete ({type(e).__name__}: {str(e)[:120]}); "
                        f"it has to be rebuilt")
    if h5_ids is not None or not os.path.exists(h5_file):
        compare(emb.H5_NAME, expected, h5_ids, prob, k)
    if csv_removed:
        csv_ids = set(h5_ids or ())                      # the HDF5 is the only copy: the checks read it instead
        res["notes"].append("the per-sample CSVs were deleted earlier (config: csv_removed); "
                            "the samples are checked from the HDF5")
    res["n_samples_found"] = len(csv_ids)
    compare("metadata.tsv (embedded)", expected, ids_of_tsv(os.path.join(d, "metadata.tsv")), prob, k)
    compare("preprocessing_log.tsv", expected, ids_of_tsv(os.path.join(d, "preprocessing_log.tsv")), prob, k)

    # ---- 2. per-sample files
    dim = cfg.get("latent_dim") or cfg.get("embedding_dim")
    bad_files = {}
    expected_keys = {}
    if args.check_sequences:
        if os.path.isdir(raw):
            cleaned, _ = prep.preprocess_dataset(raw, sorted(csv_ids & expected), max_cdr3_len=None)
            if cfg.get("sequence_list") and dataset == "train":
                if os.path.exists(cfg["sequence_list"]):
                    emb.apply_sequence_list(cleaned, emb.read_sequence_list(cfg["sequence_list"]))
                else:
                    res["notes"].append(f"sequence_list {cfg['sequence_list']} no longer exists: its effect was not applied")
            expected_keys = cleaned
        else:
            prob.append(f"--check_sequences needs the raw data at {raw}")
    genes = explained_genes(cfg)
    if args.check_sequences and not cfg:
        res["notes"].append("no config: sequences dropped on purpose (unseen / rejected genes) cannot be told apart "
                            "from missing ones, so they are all reported as unexplained")
    h5r = h5py.File(h5_file, "r") if h5_ids is not None else None     # None when missing or corrupt
    try:
        for sid in sorted(csv_ids & expected):
            path = os.path.join(d, f"{sid}_embed.csv")
            msgs = []
            try:
                if csv_removed:                          # keys and dimension come from the HDF5 group
                    grp = h5r[f"patients/{sid}"]
                    keys = pd.Series(grp["tcr_sequences"][:].astype(str))
                    if dim is not None and grp["embeddings"].shape[1] != dim:
                        msgs.append(f"{grp['embeddings'].shape[1]} embedding columns, config says {dim}")
                else:
                    cols = pd.read_csv(path, nrows=0).columns
                    keys = pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str)
            except Exception as e:                       # unreadable / truncated file
                bad_files[sid] = f"cannot be read ({type(e).__name__})"
                continue
            if not csv_removed:
                if len(cols) < 2 or cols[-1] != "vj_call":
                    msgs.append("last column is not vj_call")
                elif dim is not None and len(cols) - 2 != dim:
                    msgs.append(f"{len(cols) - 2} embedding columns, config says {dim}")
            if not keys.is_unique:
                msgs.append(f"{int(keys.duplicated().sum())} duplicated keys")
            if not keys.str.contains("+", regex=False).all():
                msgs.append("keys without '+'")
            res["n_keys"] += len(keys)
            if h5r is not None and sid in h5_ids:
                try:
                    grp = h5r[f"patients/{sid}"]
                    if grp["embeddings"].shape[0] != len(keys) or grp["tcr_sequences"].shape[0] != len(keys):
                        msgs.append(f"HDF5 has {grp['embeddings'].shape[0]} rows, expected {len(keys)}")
                    elif args.check_values:
                        e = grp["embeddings"][:]
                        if not np.isfinite(e).all():
                            msgs.append("NaN/inf in the HDF5 embeddings")
                        zero = int((np.abs(e).sum(axis=1) == 0).sum())
                        if zero:
                            msgs.append(f"{zero} all-zero embedding rows")
                except Exception as e:
                    msgs.append(f"HDF5 group unreadable ({type(e).__name__})")
            if sid in expected_keys:
                df = expected_keys[sid]
                have, want = set(keys), set(df["key"])
                miss = df[~df["key"].isin(have)]
                why = miss["v_gene"].isin(genes) | miss["j_gene"].isin(genes)
                res["missing_explained"] += int(why.sum())
                if (~why).any():
                    res["missing_unexplained"] += int((~why).sum())
                    msgs.append(f"{int((~why).sum())} of {len(want)} expected sequences missing, e.g. "
                                f"{miss.loc[~why, 'key'].head(k).tolist()}")
                extra = have - want
                if extra:
                    res["extra"] += len(extra)
                    msgs.append(f"{len(extra)} sequences that the raw data does not give, e.g. {sorted(extra)[:k]}")
            if msgs:
                bad_files[sid] = "; ".join(msgs)
    finally:
        if h5r is not None:
            h5r.close()
    if bad_files:
        prob.append(f"{len(bad_files)} of {len(csv_ids & expected)} sample files have problems")
        for sid, m in list(bad_files.items())[:k]:
            prob.append(f"  {sid}: {m}")
    res["bad_samples"] = bad_files
    return res


def delete_csvs_if_ok(d, raw_dir, dataset=None, max_examples=5):
    """Run every check (values and sequences against the raw data) on the dataset folder ``d`` and delete its
    per-sample CSVs only if no problem was found and ``embedding_config.json`` exists (``dataset`` = the raw dataset name, default the folder name). The config then gets
    ``csv_removed: true`` so later checks read the HDF5 instead. Returns (deleted, message)."""
    d = os.fspath(d)
    cfg_path = os.path.join(d, "embedding_config.json")
    if not os.path.exists(cfg_path):
        return False, "no embedding_config.json (unfinished dataset)"
    cfg = json.load(open(cfg_path))
    if cfg.get("csv_removed"):
        return False, "already deleted earlier"
    args = argparse.Namespace(check_values=True, check_sequences=True, max_examples=max_examples)
    res = check_dataset("", dataset or os.path.basename(d.rstrip("/")), d, os.fspath(raw_dir), args)
    if res["problems"]:
        return False, f"{len(res['problems'])} problem(s) found, e.g. {res['problems'][:3]}"
    n = emb.remove_sample_csvs(d)
    cfg["csv_removed"] = True
    emb.write_json(cfg_path, cfg)
    return True, f"{n} files removed after a successful check"


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    raw_dir = args.raw_dir or str(data_dir / "raw")
    base = data_dir / "embeddings"
    if not base.is_dir():
        sys.exit(f"ERROR: no embeddings folder at {base}")
    methods = args.folders or sorted(p.name for p in base.iterdir() if p.is_dir())
    results = []
    for m in methods:
        mdir = base / m
        if not mdir.is_dir():
            sys.exit(f"ERROR: no folder {mdir}")
        datasets = args.datasets or sorted(p.name for p in mdir.iterdir() if p.is_dir())
        for ds in datasets:
            d = mdir / ds
            if not d.is_dir():
                results.append({"method": m, "dataset": ds, "path": str(d), "n_samples_expected": None,
                                "n_samples_found": 0, "n_keys": 0, "missing_explained": 0, "missing_unexplained": 0,
                                "extra": 0, "notes": [], "problems": ["folder does not exist (not started)"]})
                continue
            print(f"checking {m}/{ds} ...", flush=True)
            results.append(check_dataset(m, ds, str(d), raw_dir, args))

    print()
    rows = [{"method": r["method"], "dataset": r["dataset"],
             "samples found/expected": f"{r['n_samples_found']}/{r['n_samples_expected']}",
             "sequences (keys)": r["n_keys"],
             **({"missing (explained)": r["missing_explained"], "missing (UNEXPLAINED)": r["missing_unexplained"],
                 "extra": r["extra"]} if args.check_sequences else {}),
             "status": "OK" if not r["problems"] else f"{len(r['problems'])} problem(s)"} for r in results]
    print(pd.DataFrame(rows).to_string(index=False))
    n_bad = 0
    for r in results:
        for n in r["notes"]:
            print(f"\nNOTE {r['method']}/{r['dataset']}: {n}")
        if r["problems"]:
            n_bad += 1
            print(f"\nPROBLEMS in {r['method']}/{r['dataset']}:")
            for p_ in r["problems"]:
                print("  -", p_)
    if not args.check_sequences:
        print("\n(sequence level not checked: add --check_sequences to compare the keys with the raw repertoires)")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"wrote {args.report}")
    print("\nALL OK" if n_bad == 0 else f"\n{n_bad} of {len(results)} dataset folders have problems")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
