#!/usr/bin/env python
"""Finish a SCEPTR embedding folder whose per-sample CSVs are complete but whose last stage was interrupted.

The generator writes <sample_id>_embed.csv for every sample first, and only then build
per_patient_tcr_embed.h5, metadata.tsv, preprocessing_log.tsv and (last) embedding_config.json. If the run dies in
that last stage (corrupt/missing HDF5, no config), the expensive part is already done. This script redoes only the
cheap part, WITHOUT embedding anything:

  1. re-preprocesses the raw repertoires exactly as the generator did and works out which sequences each sample's
     CSV must contain (minus the rows whose V gene SCEPTR rejects, found by asking the model, as the
     generator does);
  2. verifies that every sample has a CSV with exactly those keys and one consistent embedding size;
     if anything is missing or wrong it stops and writes nothing: those samples must be embedded again;
  3. rebuilds per_patient_tcr_embed.h5 from the CSVs (replacing a corrupt one), rewrites metadata.tsv and
     preprocessing_log.tsv, validates the folder with the same checks as the generator, and writes
     embedding_config.json last.

The config cannot be fully recovered: batch size, device and run time are unknown (written as null); the dropped
V genes are recomputed; the file is marked "reconstructed_by". It cannot prove that the vectors in the CSVs came from
the model/variant you name here: pass the same --variant (and --sequence_list, --max_samples) as the original run.

Run it in sceptr_env.

Example (the interrupted SCEPTR test_bulk dataset):
  python scripts/finalize_embeddings.py --input_path data/raw/test_bulk \\
      --output_path data/embeddings/sceptr/test_bulk
  python scripts/check_embeddings.py --datasets test_bulk --check_sequences

Flags:
  --input_path PATH     the data/raw/<dataset> folder                                          (required)
  --output_path PATH    the embedding folder, e.g. data/embeddings/sceptr/<dataset>            (required)
  --variant V           SCEPTR variant of the original run                                      [default]
  --sequence_list FILE  the list the original 'train' run used (train only)                     [none]
  --max_samples N       the value the original run used (a dry-run subset)                      [none]
  --random_seed N       the seed the original run used                                         [42]
  --h5_dtype T          float32 | float16: storage type of the rebuilt HDF5                    [float32]
  --dry_run             only verify and print what would be written
  --force               also finalize a folder that already has an embedding_config.json
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src import embeddings as emb
from src import preprocessing as prep


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--variant", default="default")
    p.add_argument("--sequence_list", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--h5_dtype", choices=emb.H5_DTYPES, default="float32",
                   help="storage type of the rebuilt HDF5 (use the one of the original run)")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    in_dir, out_dir = Path(args.input_path), Path(args.output_path)
    dataset = in_dir.name
    if not out_dir.is_dir():
        sys.exit(f"ERROR: no embedding folder at {out_dir}")
    cfg_path = out_dir / "embedding_config.json"
    if cfg_path.exists() and not args.force:
        sys.exit(f"ERROR: {cfg_path} exists: the dataset is already marked finished (--force to finalize anyway)")

    meta = prep.load_metadata(in_dir)
    sample_ids = prep.select_samples(meta, args.max_samples, args.random_seed)
    if args.max_samples is not None:
        meta = meta[meta["sample_id"].isin(sample_ids)]
    cleaned, log = prep.preprocess_dataset(in_dir, sample_ids, max_cdr3_len=None)   # SCEPTR has no length limit
    print(f"Preprocessed {len(cleaned)} samples of '{dataset}': {log['n_clonotypes'].sum():,} clonotypes")

    import generate_sceptr_embeddings as gen
    model = gen.load_model(SimpleNamespace(n_jobs=None, device="cpu", variant=args.variant, batch_size=512))
    rejected, n_rejected = gen.apply_v_policy(cleaned, model, args.variant, "drop")
    print(f"SCEPTR rejects {n_rejected:,} rows (V genes {sorted(rejected)}); they are not in the CSVs by design")
    n_not_in_list = 0
    if args.sequence_list is not None and dataset == "train":
        n_not_in_list = emb.apply_sequence_list(cleaned, emb.read_sequence_list(args.sequence_list))
        print(f"--sequence_list: {n_not_in_list:,} rows not in the list")

    # ---- verify every CSV against what the generator would have written
    problems, dims, n_keys = [], set(), 0
    for sid, df in cleaned.items():
        path = out_dir / f"{sid}_embed.csv"
        if not path.exists():
            problems.append(f"{sid}: no CSV")
            continue
        try:
            cols = pd.read_csv(path, nrows=0).columns
            keys = pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str)
        except Exception as e:
            problems.append(f"{sid}: CSV cannot be read ({type(e).__name__})")
            continue
        if len(cols) < 2 or cols[-1] != "vj_call":
            problems.append(f"{sid}: last column is not vj_call")
            continue
        dims.add(len(cols) - 2)
        have, want = set(keys), set(df["key"])
        if not keys.is_unique:
            problems.append(f"{sid}: duplicated keys")
        if have != want:
            problems.append(f"{sid}: {len(want - have)} expected keys missing, {len(have - want)} unexpected keys")
        n_keys += len(keys)
    if len(dims) > 1:
        problems.append(f"CSVs have different embedding sizes: {sorted(dims)}")
    if problems:
        print(f"\nNOT finalized: {len(problems)} problem(s); nothing was written. These samples need to be embedded again:")
        for m in problems[:20]:
            print("  -", m)
        return 1
    dim = dims.pop()
    print(f"All {len(cleaned)} CSVs match the expected sequences ({n_keys:,} keys, {dim} dimensions).")
    if args.dry_run:
        print("--dry_run: nothing written")
        return 0

    # ---- rebuild the rest; the config goes last
    h5 = out_dir / emb.H5_NAME
    tmp = out_dir / (emb.H5_NAME + ".tmp")
    emb.save_patient_embeddings(str(out_dir), str(tmp), list(cleaned), dtype=args.h5_dtype)
    os.replace(tmp, h5)
    meta.to_csv(out_dir / "metadata.tsv", sep="\t", index=False)
    log.to_csv(out_dir / "preprocessing_log.tsv", sep="\t", index=False)
    summary = emb.validate_embedding_dir(str(out_dir), list(cleaned), expected_dim=dim)
    config = {"method": "sceptr", "variant": args.variant, "dataset": dataset, "random_seed": args.random_seed,
              "max_samples": args.max_samples, "sequence_list": args.sequence_list,
              "n_rows_not_in_sequence_list": n_not_in_list, "batch_size": None, "device": None,
              "elapsed_seconds": None, "invalid_v_policy": "drop", "n_rows_dropped_invalid_v": n_rejected,
              "invalid_v_genes": rejected, "chunk_size": None,
              "reconstructed_by": "scripts/finalize_embeddings.py (CSVs of an interrupted run; batch size, device, "
                                  "run time and the model that made the vectors are not recorded)"}
    emb.write_json(str(cfg_path), {**config, "latent_dim": dim, "h5_dtype": args.h5_dtype, **summary})
    print(f"Finalized {out_dir}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
