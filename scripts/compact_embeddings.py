#!/usr/bin/env python
"""Shrink existing embedding folders: store the HDF5 vectors as float16 and/or delete the per-sample CSVs.

The pipeline reads only <dataset>/per_patient_tcr_embed.h5; the CSVs are an intermediate that is only needed to
check or repair a dataset. For every selected finished dataset (one with an embedding_config.json):

  --h5_dtype float16   rewrite the HDF5 with the vectors as float16 (patient by patient into a temporary file,
                       then it replaces the original; needs free disk space of about the new file). Skipped if it
                       is already that type. float16 halves the vectors; on tutorial data the nearest centroid was
                       unchanged for 99.998 % of sequences. Do not mix types within one experiment; the type is
                       recorded in embedding_config.json ("h5_dtype"). float32 -> float16 -> float32 does not
                       restore the original values.
  --delete_csv         run the full checks of check_embeddings.py (values, sequences vs the raw data) and delete the
                       CSVs only if no problem is found; otherwise they are kept and the problems are printed. The
                       config gets "csv_removed": true, and check_embeddings.py then checks from the HDF5.

Unfinished datasets (no config) are skipped untouched. With both flags the conversion runs first, then the checks.

Usage:
  python scripts/compact_embeddings.py --h5_dtype float16 --delete_csv
  python scripts/compact_embeddings.py --folders sceptr --datasets train --delete_csv

Flags:
  --data_dir PATH      folder that contains embeddings/                                  [<project>/data]
  --raw_dir PATH       folder with the raw datasets (<raw_dir>/<dataset>)                [<data_dir>/raw]
  --folders F [F ...]  embedding folders under embeddings/                               [all that exist]
  --datasets D [D ...] dataset folders                                                   [all that exist]
  --h5_dtype T         float32 | float16: convert the HDF5 to this type                   [no conversion]
  --delete_csv         delete the CSVs after a successful check
Exit code 1 if a requested step could not be done for some dataset.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_embeddings
from src import embeddings as emb
from src import preprocessing as prep


def parse_args(argv=None):
    root = prep.project_root()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=str(root / "data"))
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--folders", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--h5_dtype", choices=emb.H5_DTYPES, default=None)
    p.add_argument("--delete_csv", action="store_true")
    return p.parse_args(argv)


def convert(d, dtype):
    """Rewrite the dataset's HDF5 as ``dtype``. Returns (done, message)."""
    h5 = os.path.join(d, emb.H5_NAME)
    if not os.path.exists(h5):
        return False, "no HDF5 file"
    try:
        have = emb.h5_dtype_of(h5)
    except Exception as e:
        return False, f"HDF5 unreadable ({type(e).__name__}); not converted"
    if have == dtype:
        return True, f"already {dtype}"
    tmp = h5 + ".tmp"
    try:
        emb.convert_h5_dtype(h5, tmp, dtype)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, f"conversion failed ({type(e).__name__}: {e}); original kept"
    before, after = os.path.getsize(h5), os.path.getsize(tmp)
    os.replace(tmp, h5)
    cfg_path = os.path.join(d, "embedding_config.json")
    cfg = emb.load_embedding_config(d)
    cfg["h5_dtype"] = dtype
    emb.write_json(cfg_path, cfg)
    return True, f"{have} -> {dtype}: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB"


def main(argv=None):
    args = parse_args(argv)
    if args.h5_dtype is None and not args.delete_csv:
        sys.exit("ERROR: nothing to do: give --h5_dtype and/or --delete_csv")
    data_dir = Path(args.data_dir)
    raw_dir = args.raw_dir or str(data_dir / "raw")
    base = data_dir / "embeddings"
    if not base.is_dir():
        sys.exit(f"ERROR: no embeddings folder at {base}")
    failed = 0
    for m in args.folders or sorted(p.name for p in base.iterdir() if p.is_dir()):
        for ds in args.datasets or sorted(p.name for p in (base / m).iterdir() if p.is_dir()):
            d = str(base / m / ds)
            tag = f"{m}/{ds}"
            if not os.path.exists(os.path.join(d, "embedding_config.json")):
                print(f"[skip] {tag}: no embedding_config.json (unfinished dataset, left untouched)")
                continue
            if args.h5_dtype:
                ok, msg = convert(d, args.h5_dtype)
                print(f"[{'ok' if ok else 'FAILED'}] {tag} dtype: {msg}")
                if not ok:
                    failed += 1
                    continue
            if args.delete_csv:
                ok, msg = check_embeddings.delete_csvs_if_ok(d, raw_dir)
                kept = not ok and msg != "already deleted earlier"
                print(f"[{'ok' if ok else ('KEPT' if kept else 'ok')}] {tag} csv: {msg}")
                failed += int(kept)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
