"""Build the tiny CPU test fixture ``tests/data`` from ``data/raw`` (read-only).

3 train samples per ``disease_group`` (distinct diseases where possible), 2
from each test set, each cut to ``--rows`` random rows. The 2 test_bulk samples
are chosen among those with comma-ambiguous V calls, so the fixture exercises
that preprocessing step (train samples are seeded-random).
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocessing as prep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--raw_dir", default=str(prep.project_root() / "data" / "raw"))
    p.add_argument("--out_dir", default=str(prep.project_root() / "tests" / "data"))
    args = p.parse_args()
    rng = np.random.default_rng(args.random_seed)
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    stats = pd.read_csv(prep.project_root() / "results" / "eda" / "sample_stats.csv.gz",
                        dtype={"sample_id": str}).set_index(["dataset", "sample_id"])

    picks = {}
    meta = prep.load_metadata(raw / "train")
    chosen = []
    for _, g in meta.groupby("disease_group"):
        order = rng.permutation(len(g))
        g = g.iloc[order].drop_duplicates("disease")
        chosen += g["sample_id"].tolist()[:3]
    picks["train"] = chosen
    for d in ("test_bulk", "test_sc"):
        m = prep.load_metadata(raw / d)
        ids = m["sample_id"].tolist()
        if d == "test_bulk":
            amb = [s for s in ids if stats.loc[(d, s), "n_vgene_ambiguous"] > 0]
            ids = amb or ids
        picks[d] = rng.choice(ids, 2, replace=False).tolist()

    if out.exists():
        shutil.rmtree(out)
    for d, ids in picks.items():
        (out / d).mkdir(parents=True)
        m = prep.load_metadata(raw / d)
        m[m["sample_id"].isin(ids)].to_csv(out / d / "metadata.tsv", sep="\t", index=False)
        for s in ids:
            df = pd.read_csv(raw / d / f"{s}.tsv", sep="\t", dtype=str, keep_default_na=False)
            df = df.sample(n=min(args.rows, len(df)), random_state=args.random_seed)
            df.to_csv(out / d / f"{s}.tsv", sep="\t", index=False)
        print(d, ids)


if __name__ == "__main__":
    main()
