# TCR repertoire -> disease class

Predict the disease class (`autoimmunity` / `cancer` / `infection`) of a T-cell receptor beta (TRB)
repertoire from its CDR3 sequences. The primary metric is the macro one-vs-rest ROC-AUC.

## Data

Expected under `data/raw/` (not part of the repository; never modified by the code):

| folder | samples | labels | role |
|---|---|---|---|
| `train/` | 411 bulk repertoires | yes (`disease`, `disease_group`) | training / cross-validation |
| `test_bulk/` | 117 bulk repertoires | no | primary prediction target |
| `test_sc/` | 128 single-cell repertoires | no | secondary prediction target |

Each folder holds one tab-separated file per sample (`sample_id, locus, cdr3_nt, cdr3_aa, v_gene, j_gene,
clone_count`, one row per clonotype) and a `metadata.tsv`. `train` and `test_bulk` share one ID space
(`sample_1` ... `sample_528`); `test_sc` has its own zero-padded one (`sample_001` ... `sample_128`), so samples are
always keyed on `(dataset, sample_id)`. The `MHCI`/`MHCII` columns are dropped on load.

Preprocessing (`src/preprocessing.py`, shared by every entry point): canonical CDR3 filter, removal of rows with
missing or ambiguous V/J calls, V-gene name harmonisation (14 single-member V genes appear as `TRBV2` or `TRBV2-1`
depending on the sample, a batch effect; the `-1` suffix is stripped everywhere), and collapsing of
identical `cdr3_aa + V + J` clonotypes (summing `clone_count`).

## Pipeline

```
raw repertoires -> preprocessing -> SCEPTR embedding per unique clonotype (64-d)
  -> [optional] Fisher sequence filter -> FAISS k-means (K clusters, per-cluster radius)
  -> per-sample cluster-occupancy features -> class-aware prevalence filter -> XGBoost
```

1. **Embeddings.** The pretrained SCEPTR model embeds every unique `cdr3_aa + V_J` clonotype
   (`scripts/generate_sceptr_embeddings.py`); results are pooled into one HDF5 file per dataset.
2. **Sequence filter** (`--seq_filter none|clonotype|kmer|vj`). A label-based one-vs-rest Fisher test keeps
   disease-associated clonotypes / k-mers / V genes.
3. **Clustering** (`src/clustering.py`). FAISS k-means on the training sequences; a sequence joins its nearest
   cluster only if it lies within that cluster's radius (a quantile of its members' distances), otherwise it is
   "unassigned". Clusters with too few training members are dropped.
4. **Features.** Per sample: share of its sequences in each cluster, the unassigned share, Pielou evenness,
   `log_depth`. A class-aware prevalence filter removes uninformative clusters.
5. **Model.** XGBoost (`multi:softprob`). Tuning is two-stage: a grid over K x radius quantile with a fixed
   classifier, then Optuna for the XGBoost parameters. Evaluation is stratified 5-fold CV with the clustering and
   every label-based step **refit inside each fold** (no leakage); optional leave-one-disease-out (LODO) and SHAP
   cluster importance.

Layout: `src/` (library), `scripts/` (embedding, tuning and check scripts), `run_pipeline.py` (final fit /
evaluation), `notebooks/` (01 EDA, 02 tutorial on a tiny subset, 03 final model on full data), `tests/`.
A diagram is in `fig/pipeline_scheme.png`.

## Setup

```bash
conda env create -f env/sceptr_env_linux.yml     # Linux server (CUDA 12 builds)
# conda env create -f env/sceptr_env_mac.yml     # macOS, CPU only
conda activate sceptr_env
```

GPU is optional. Devices are chosen with flags and default to CPU; requesting `cuda` when it is unavailable fails with
a clear message: `--device` (SCEPTR embeddings), `--xgboost_device`, `--kmeans_device` (FAISS; needs the PyPI wheel
`faiss-gpu-cu12`). `python scripts/check_gpu_stack.py` shows which components can use the GPU on a machine.
Random seed: `--random_seed` (default 42).

## Getting predictions

Run everything from the repository root, inside `sceptr_env`. Every script has `--help`; the `.sh` wrappers print
their documented flags with `bash scripts/<name>.sh --help`.

**1. Embeddings** (all three datasets; finished datasets are skipped on a rerun):

```bash
bash scripts/run_embeddings.sh --device cuda --batch_size 2048      # or --device cpu
```

Output: `data/embeddings/sceptr/{train,test_bulk,test_sc}/` (per-sample CSVs, `per_patient_tcr_embed.h5`,
`embedding_config.json`). Optional disk savers: `--h5_dtype float16` and `--delete_csv` (see `--help`). Verify:

```bash
python scripts/check_embeddings.py --folders sceptr --check_sequences
```

**2. Hyperparameter tuning** on `train` (one run per filter setting):

```bash
bash scripts/run_tuning.sh --seq_filter kmer --xgboost_device cuda --kmeans_device cuda
```

Output: `results/run/tuning/sceptr_kmer_kmeans/` with `hyperparameters.json` (chosen K, radius quantile and XGBoost
parameters), `cv_metrics.json`, `stage1_leaderboard.csv`, `stage2_trials.csv`. Several filters can be given
(`--seq_filter none kmer`) and compared by their CV macro AUC. Use `--max_samples 40 --n_trials 5` first as a quick
smoke test.

**3. Final fit and predictions** (refits on all of `train`, predicts the test sets):

```bash
python run_pipeline.py --stage fit_predict \
    --hyperparams_path results/run/tuning/sceptr_kmer_kmeans/hyperparameters.json \
    --xgboost_device cuda --kmeans_device cuda --output_dir results/final
```

Output in `results/final/`:

- `predictions_test_bulk.tsv`, `predictions_test_sc.tsv`: one row per sample, columns
  `sample_id  autoimmunity  cancer  infection` (class probabilities);
- `cluster_importance.csv` (SHAP importance per cluster), prevalence report, feature tables;
- the fitted cluster model and booster in `models/final/` (`--models_dir` to change).

Optional flags: `--stage cv` (reuse the persisted fold features for a CV report), `--lodo` (leave-one-disease-out),
`--shap_plots`, `--sc_without_log_depth` (extra `test_sc` predictions from a model without `log_depth`, which differs
strongly between bulk and single-cell data), `--downsample_to N|sc_median` (with `--stage cv`).

The test sets have no labels in `data/raw`, so there is no test-set score; model quality is judged from the CV and
LODO results.

## Notebooks

- `notebooks/02_tcr_pipeline_tutorial.ipynb`: the Python API on a tiny subset (`data/tutorial/`, not in the repository since `data/` is ignored), about 1 minute on CPU;
  illustrates the mechanics only (24 training samples, CV AUC is at chance level).
- `notebooks/03_final_model_full_data.ipynb`: the final fit, validation of the predictions and feature-importance
  analysis; set the paths in the first cell.

## Tests

```bash
bash scripts/run_tests.sh              # quick set, throttled, about 1 minute
bash scripts/run_tests.sh --full       # adds the end-to-end and embedding-generator tests (many minutes)
```
