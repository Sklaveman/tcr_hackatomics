"""Synthetic embedded datasets for the Task 4 tests.

Writes ``<data_dir>/embeddings/<method>/{train,test_bulk,test_sc}`` in exactly
the format the generators produce (per-sample ``*_embed.csv``, the per-patient
HDF5, ``metadata.tsv``, ``embedding_config.json``), so the whole pipeline is
exercised end to end without running SCEPTR.

The signal is planted: every class has its own embedding clusters that its
samples over-use (with a per-sample strength), plus a few class-specific
public clonotypes shared by most samples of that class, so the model can be
checked to *learn* something and the Fisher filter to find the planted
clonotypes.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import embeddings as emb

DISEASES = {"autoimmunity": ["MS", "T1D"], "cancer": ["Melanoma", "NSCLC"],
            "infection": ["CMV", "HIV"]}


def make_synthetic_data(data_dir, method="sceptr", seed=0, dim=32, n_per_disease=10,
                        n_seq=300, n_bulk=12, n_sc=6, signal=(0.08, 0.35), seq_only=False,
                        n_public=5, p_public=0.7):
    """Create the dataset; returns {'public': {class: [keys]}, 'meta': train meta}."""
    rng = np.random.default_rng(seed)
    n_bg, n_cls = 12, 4
    bg_centers = rng.normal(size=(n_bg, dim)) * 1.5
    cls_centers = {c: rng.normal(size=(n_cls, dim)) * 1.5 for c in range(3)}
    public = {c: [f"CASSPUB{c}X{i}F+TRBV{c + 1}_TRBJ1-{i % 2 + 1}" for i in range(n_public)]
              for c in range(3)}
    public_emb = {k: cls_centers[c][i % n_cls] + rng.normal(scale=0.1, size=dim)
                  for c in range(3) for i, k in enumerate(public[c])}
    counter = [0]

    def sample(cls, n, strength):
        """One repertoire: (keys, embeddings)."""
        n_sig = rng.binomial(n, strength) if cls is not None else 0
        centers = np.vstack([
            bg_centers[rng.integers(0, n_bg, n - n_sig)],
            *( [cls_centers[cls][rng.integers(0, n_cls, n_sig)]] if n_sig else [] )])
        x = centers + rng.normal(scale=0.25, size=centers.shape)
        keys = []
        for _ in range(len(x)):
            counter[0] += 1
            keys.append(f"CASSQ{counter[0]}F+TRBV{rng.integers(1, 6)}_TRBJ{rng.integers(1, 3)}-1")
        return keys, x

    def write(dataset, rows, labelled):
        d = emb.embedding_dir(data_dir, method, dataset)
        os.makedirs(d, exist_ok=True)
        meta = []
        for sid, (disease, group, keys, x) in rows.items():
            vj = [k.split("+")[1] for k in keys]
            emb.write_sample_embeddings(d, sid, keys, vj, x)
            meta.append({"sample_id": sid, **({"disease": disease, "disease_group": group}
                                              if labelled else {})})
        meta = pd.DataFrame(meta)
        emb.save_patient_embeddings(d, os.path.join(d, emb.H5_NAME), list(rows))
        meta.to_csv(os.path.join(d, "metadata.tsv"), sep="\t", index=False)
        emb.write_json(os.path.join(d, "embedding_config.json"),
                       {"method": method, "dataset": dataset, "latent_dim": dim,
                        "seq_only": seq_only, "mode": "train"})
        return meta

    train, i = {}, 0
    for c, group in enumerate(emb.CLASS_NAMES):
        for disease in DISEASES[group]:
            for _ in range(n_per_disease):
                i += 1
                keys, x = sample(c, n_seq, rng.uniform(*signal))
                if rng.random() < p_public:  # class-specific public clonotypes
                    keys += public[c]
                    x = np.vstack([x, [public_emb[k] for k in public[c]]])
                train[f"sample_{i}"] = (disease, group, keys, x)
    meta = write("train", train, True)

    def unlabelled(n, start, width, size):
        rows = {}
        for j in range(n):
            keys, x = sample(j % 3, size, rng.uniform(*signal))
            rows[f"sample_{start + j:0{width}d}"] = (None, None, keys, x)
        return rows
    write("test_bulk", unlabelled(n_bulk, i + 1, 1, n_seq), False)   # shares train's ID space
    write("test_sc", unlabelled(n_sc, 1, 3, max(30, n_seq // 3)), False)
    return {"public": public, "meta": meta}
