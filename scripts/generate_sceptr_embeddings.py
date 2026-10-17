#!/usr/bin/env python
"""SCEPTR embedding generator.

Embeds the unique ``cdr3_aa+V_J`` clonotypes of one ``data/raw/<dataset>`` folder with a
pretrained SCEPTR model (https://github.com/yutanagano/sceptr; 64-d unit-norm vectors for
the default variant) and writes the shared embedding format (see ``src/embeddings.py``).

What to know about SCEPTR (checked against sceptr 1.2.0 in ``sceptr_env``):
  * input here is the beta chain only: columns ``TRBV`` (V gene) and ``CDR3B``. SCEPTR derives CDR1/2
    from the V gene, so the V gene is part of the embedding; the J gene is not used by the model;
  * V genes must be functional IMGT names. The project's canonical names work (TRBV30, TRBV10-1);
    non-functional or unknown ones (TRBV1, 7-5, 12-1, */OR9-2, TRBVA-1, ...; about 1.3 % of train and
    test_bulk rows) are rejected by the model. ``--invalid_v_policy drop`` (default) removes those rows
    and logs them; ``error`` stops instead. ``--variant cdr3_only`` ignores the V gene and drops nothing;
  * no CDR3 length limit was found (4 to 80 residues all worked), so no length filter is applied;
  * SCEPTR is pretrained: there is no train mode, and test datasets are embedded like ``train``.
Run it in ``sceptr_env`` (``env/sceptr_env_*.yml``).
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import embeddings as emb
from src import preprocessing as prep

VARIANTS = ("default", "b_sceptr", "cdr3_only", "finetuned", "large", "small", "tiny", "blosum",
            "average_pooling", "left_aligned", "mlm_only", "cdr3_only_mlm_only", "dropout_noise_only",
            "shuffled_data", "synthetic_data")
PROBE_CDR3 = "CASSLGQETQYF"          # any valid CDR3, used to ask the model whether it accepts a V gene


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input_path", required=True, help="a data/raw/<dataset> folder")
    p.add_argument("--output_path", required=True, help="e.g. data/embeddings/sceptr/<dataset>")
    p.add_argument("--variant", choices=VARIANTS, default="default",
                   help="SCEPTR model variant (sceptr.variant.<name>); 'default' is the published "
                        "SCEPTR, 'cdr3_only' ignores the V gene")
    p.add_argument("--batch_size", type=int, default=512, help="TCRs per forward pass (SCEPTR default 512)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--chunk_size", type=int, default=200_000,
                   help="unique sequences per call to the model (bounds memory)")
    p.add_argument("--invalid_v_policy", choices=["error", "drop"], default="drop",
                   help="V genes the model rejects: stop, or drop those rows and log them")
    p.add_argument("--sequence_list", default=None,
                   help="text file, one 'cdr3_aa+V_J' key per line. For the 'train' dataset only these "
                        "sequences are embedded; test datasets are always embedded in full")
    p.add_argument("--h5_dtype", choices=emb.H5_DTYPES, default="float32",
                   help="storage type of the vectors in per_patient_tcr_embed.h5. float16 halves the vectors; "
                        "measured on tutorial data: nearest centroid unchanged for 99.998 %% of sequences, "
                        "so results differ negligibly but not bit-for-bit. Do not mix types within one experiment")
    p.add_argument("--delete_csv", action="store_true",
                   help="after the dataset is written, run the full checks of check_embeddings.py (values and "
                        "sequences vs the raw data) and delete the per-sample CSVs only if all pass; otherwise "
                        "they are kept. The pipeline reads only the HDF5")
    p.add_argument("--max_samples", type=int, default=None, help="dry runs")
    p.add_argument("--n_jobs", type=int, default=None,
                   help="CPU threads for PyTorch (default: all cores)")
    p.add_argument("--random_seed", type=int, default=42,
                   help="SCEPTR inference is deterministic; the seed only picks the --max_samples subset")
    return p.parse_args(argv)


def load_model(args):
    import torch
    import sceptr
    if args.n_jobs:
        torch.set_num_threads(args.n_jobs)
    model = getattr(sceptr.variant, args.variant)()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("ERROR: --device cuda requested but PyTorch has no CUDA device")
        model.enable_hardware_acceleration()
    else:
        model.disable_hardware_acceleration()
    model.set_batch_size(args.batch_size)
    return model


def tcr_frame(df, variant):
    """The input table SCEPTR expects (beta chain only)."""
    out = pd.DataFrame({"CDR3B": df["cdr3_aa"].to_numpy()})
    if variant != "cdr3_only":
        out["TRBV"] = df["v_gene"].to_numpy()
    return out


def accepted_v_genes(model, v_genes, variant):
    """Ask the model itself which V gene names it accepts (one probe per distinct name)."""
    if variant == "cdr3_only":
        return set(v_genes)
    ok = set()
    for v in v_genes:
        try:
            model.calc_vector_representations(pd.DataFrame({"TRBV": [v], "CDR3B": [PROBE_CDR3]}))
            ok.add(v)
        except Exception:                                    # rejected: not a functional IMGT name
            pass
    return ok


def apply_v_policy(cleaned, model, variant, policy):
    """Drop (or refuse) rows whose V gene the model rejects. Returns ({gene: rows}, n_dropped)."""
    names = sorted({v for df in cleaned.values() for v in df["v_gene"].unique()})
    ok = accepted_v_genes(model, names, variant)
    rejected, n_dropped = {}, 0
    for sid, df in cleaned.items():
        bad = ~df["v_gene"].isin(ok)
        for g, c in df.loc[bad, "v_gene"].value_counts().items():
            rejected[g] = rejected.get(g, 0) + int(c)
        if policy == "drop" and bad.any():
            cleaned[sid] = df[~bad].reset_index(drop=True)
            n_dropped += int(bad.sum())
    if rejected and policy == "error":
        sys.exit(f"ERROR: V genes SCEPTR rejects (rows): {rejected}. Pass --invalid_v_policy drop to drop "
                 f"those rows and log them, or use --variant cdr3_only.")
    return rejected, n_dropped


def embed(model, uniq, args):
    """SCEPTR vectors for all unique sequences, in chunks, as float32 (n, dim)."""
    parts = []
    for start in range(0, len(uniq), args.chunk_size):
        c = uniq.iloc[start:start + args.chunk_size]
        parts.append(model.calc_vector_representations(tcr_frame(c, args.variant)).astype(np.float32))
        print(f"  embedded {min(start + args.chunk_size, len(uniq)):,}/{len(uniq):,}", flush=True)
    return np.vstack(parts)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    in_dir, out_dir = Path(args.input_path), Path(args.output_path)
    dataset = in_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = prep.load_metadata(in_dir)
    sample_ids = prep.select_samples(meta, args.max_samples, args.random_seed)
    if args.max_samples is not None:
        meta = meta[meta["sample_id"].isin(sample_ids)]       # subset metadata for dry runs

    # no known CDR3 length limit (see the module docstring): no length filter
    cleaned, log = prep.preprocess_dataset(in_dir, sample_ids, max_cdr3_len=None)
    print(f"Preprocessed {len(cleaned)} samples of '{dataset}': {log['n_clonotypes'].sum():,} clonotypes")

    model = load_model(args)
    rejected, n_rejected = apply_v_policy(cleaned, model, args.variant, args.invalid_v_policy)
    if rejected:
        print(f"Dropped {n_rejected:,} rows whose V gene SCEPTR rejects: {rejected}")

    n_not_in_list = 0
    if args.sequence_list is not None and dataset == "train":
        n_not_in_list = emb.apply_sequence_list(cleaned, emb.read_sequence_list(args.sequence_list))
        print(f"--sequence_list: kept only listed sequences ({n_not_in_list:,} rows not in the list)")

    uniq = emb.unique_sequences(cleaned)
    if len(uniq) == 0:
        sys.exit("ERROR: no sequences left to embed")
    print(f"Embedding {len(uniq):,} unique sequences with SCEPTR variant '{args.variant}' on {args.device}")
    z = embed(model, uniq, args)

    summary = emb.write_dataset_embeddings(
        out_dir, cleaned, uniq, z, meta, log, {
            "method": "sceptr", "variant": args.variant, "dataset": dataset,
            "random_seed": args.random_seed, "batch_size": args.batch_size,
            "max_samples": args.max_samples, "sequence_list": args.sequence_list,
            "invalid_v_policy": args.invalid_v_policy, "n_rows_dropped_invalid_v": n_rejected,
            "invalid_v_genes": rejected, "n_rows_not_in_sequence_list": n_not_in_list,
            "device": args.device, "chunk_size": args.chunk_size,
            "elapsed_seconds": round(time.time() - t0, 1)}, h5_dtype=args.h5_dtype)
    if args.delete_csv:
        import check_embeddings
        ok, msgs = check_embeddings.delete_csvs_if_ok(out_dir, in_dir.parent, dataset)
        print(("CSVs deleted: " if ok else "CSVs kept: ") + msgs)
    print(f"Done in {time.time() - t0:.1f}s: {summary}")
    return summary


if __name__ == "__main__":
    main()
