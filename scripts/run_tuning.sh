#!/usr/bin/env bash
# Hyperparameter tuning on the training set (scripts/tune_hyperparameters.py): SCEPTR embeddings, FAISS k-means
# clustering, XGBoost, for one or several sequence-filter settings.
#
# Usage:  bash scripts/run_tuning.sh [flags]
# Flags take a value ("--xgboost_device cuda" or "--xgboost_device=cuda"); list flags take several values
# ("--grid_kmeans_k 300 1000"); (switch) flags take none. Flags you do not give are not forwarded,
# so the defaults of tune_hyperparameters.py apply (shown in brackets below).
#
# HOW A RUN IS ORGANISED
#   Tuning has two stages: stage 1 searches the CLUSTERING settings (number of clusters K x radius quantile Q),
#   stage 2 the XGBoost settings (Optuna). One tuning run is made for every --seq_filter value, one after another,
#   because the runs share <data_dir>/features/sceptr/cv/ (a later run overwrites the earlier one's persisted
#   fold features). Each run writes <output_dir>/tuning/sceptr_<seq_filter>_kmeans/ (hyperparameters.json,
#   cv_metrics.json, stage1_leaderboard.csv, ...). A run whose hyperparameters.json already exists is skipped
#   unless --force is given, so an interrupted batch can be restarted with the same command. Only the `train`
#   embeddings (<data_dir>/embeddings/sceptr/train) are needed, except with --sequence_list_only, which reads
#   <data_dir>/raw/train instead.
#
# ===== 1. WHAT TO RUN ==============================================================================
#   --seq_filter F [F ...]   Fisher sequence filter(s): none | clonotype | kmer | vj           [none]
#   --sceptr_variant V       SCEPTR variant of the embeddings: reads embeddings/sceptr, or embeddings/sceptr_<V>
#                            (results then go to tuning/sceptr_<V>_<seq_filter>_kmeans)         [default]
#   --max_samples N          use only N training samples (smoke test)                          [all]
#   --exclude_diseases D [D ...]  drop all training samples of these diseases (names as in metadata.tsv,
#                            case-insensitive, e.g. Melanoma Ovarian_cancer). Stored in hyperparameters.json
#                            and reused by run_pipeline.py. Results go to the same sceptr_<seq_filter>_kmeans
#                            folder, so use a different --output_dir than for the full-data run   [none]
#   --force (switch)         redo runs that already have their result file
#   --dry_run (switch)       print the commands and exit without running anything
#
# ===== 2. PATHS, DEVICES, ENVIRONMENT ==============================================================
#   --data_dir PATH          folder with embeddings/sceptr/train (and raw/train for --sequence_list_only);
#                            per-fold features are persisted under <data_dir>/features           [<project>/data]
#   --output_dir PATH        results root; runs go to <output_dir>/tuning/sceptr_<seq_filter>_kmeans [<project>/results/run]
#   --xgboost_device D       cpu | cuda: the XGBoost device (cuda is checked up front). It does NOT move k-means:
#                            that has its own --kmeans_device (section 4)                        [cuda]
#   --random_seed N          seed for folds, clustering, Optuna                                [42]
#   --n_jobs N               CPU workers (-1 = all)                                            [-1]
#   --conda_env NAME         conda env to activate if it is not already active ('' = do not activate;
#                            it must contain faiss and xgboost)                                [sceptr_env]
#   --python EXE             python executable to use after activation                         [python]
#   --log_dir PATH           one log file per run is written here                              [<project>/logs]
#
# ===== 3. CROSS-VALIDATION AND CLASSIFIER SEARCH ===================================================
#   --n_splits N             stratified CV folds                                               [5]
#   --n_trials N             stage 2: Optuna trials for XGBoost                                [60]
#
# ===== 4. K-MEANS CLUSTERING (stage 1) =============================================================
#   Stage 1 fits k-means for every K and, for every radius quantile Q, assigns each sequence to its nearest
#   centroid unless it lies beyond that cluster's radius.
#   --grid_kmeans_k K [K ...]        numbers of clusters K to try                               [300 1000 3000]
#   --grid_kmeans_radius_q Q [Q ...] radius quantiles in (0, 1] to try. A cluster's radius is the Q-quantile of
#                            its training members' distances to the centroid; a sequence farther than that from
#                            its nearest centroid is unassigned (1 = the farthest member, 0.8 = cut off the outer
#                            20 %). Extra values are nearly free (the searches are shared)       [0.8 0.9 1.0]
#   --min_cluster_size N     clusters with fewer ASSIGNED training sequences (inside their radius) are dropped [20]
#   --kmeans_niter N         k-means iterations                                                 [25]
#   --kmeans_nredo N         independent k-means runs, best kept                                [1]
#   --kmeans_max_points_per_centroid N  FAISS subsamples the training set to about N points per centroid [256]
#   --kmeans_spherical S     auto | yes | no: cosine distance on normalised vectors (auto = yes for SCEPTR) [auto]
#   --kmeans_device D        cpu | cuda: where FAISS fits and searches (independent of --xgboost_device).
#                            cuda needs faiss-gpu (the PyPI wheel faiss-gpu-cu12 works on V100; the conda-forge
#                            build lacks kernels for some GPU generations); it is probed first   [cpu]
#
# ===== 5. SEQUENCE FILTER (used when --seq_filter is not none) =====================================
#   --fisher_select S        fdr | topk | percentile: how features are selected per class      [fdr]
#   --fisher_alpha A         fdr: Benjamini-Hochberg q threshold                               [0.05]
#   --fisher_top_k K         topk: number of smallest p-values kept per class                  [1000]
#   --fisher_top_percent P   percentile: % of tested features kept per class                   [5.0]
#   --grid_fisher [V ...]    tune the strictness of the selected mode: stage 1 runs once per value
#                            (cost grows linearly) and the best wins. Off unless the flag is given.
#                            Without values: the default grid 1 5 20 (percent; needs
#                            --fisher_select percentile); other modes need explicit values      [off]
#   --min_samples N          drop features present in fewer training samples than this         [3]
#   --kmer_k K               k-mer length when --seq_filter kmer                               [3]
#   --with_v (switch)        combine k-mers with the V gene
#   --kmer_trim_start N      cut N residues off the start of every CDR3 before taking k-mers  [0]
#   --kmer_trim_end N        cut N residues off the end of every CDR3 before taking k-mers    [0]
#   --sequence_list_only (switch)  only fit the filter on the raw repertoires and write
#                            sequence_list.txt (no embeddings needed); needs --seq_filter other than none
#
# ===== 6. CLUSTER-PREVALENCE FILTER =================================================================
#   --prevalence_mode M      class_aware | global                                              [class_aware]
#   --min_prevalence X       minimum share of samples (of a class, or of all) occupying a cluster [0.1]
#   --min_prevalence_count N minimum number of occupying samples in that class                 [3]
#
#   -h, --help               print this header
#
# Examples:
#   bash scripts/run_tuning.sh --seq_filter none kmer --xgboost_device cuda --kmeans_device cuda \
#       --grid_kmeans_k 300 1000 --grid_kmeans_radius_q 0.8 0.9 1.0
#   Fisher grid:  bash scripts/run_tuning.sh --seq_filter kmer --fisher_select percentile --grid_fisher
#   Fisher list only:  bash scripts/run_tuning.sh --seq_filter kmer --sequence_list_only
#   Smoke test:  bash scripts/run_tuning.sh --max_samples 30 --n_trials 3 --xgboost_device cpu --dry_run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEQ_FILTERS=(none); DATA_DIR="$ROOT/data"; OUTPUT_DIR="$ROOT/results/run"; DEVICE=cuda
SEED=""; N_JOBS=""; MAX_SAMPLES=""; FORCE=0; DRY_RUN=0; VARIANT=default
CONDA_ENV=sceptr_env; PYTHON=python; LOG_DIR="$ROOT/logs"
N_SPLITS=""; MIN_CLUSTER_SIZE=""; N_TRIALS=""
FISHER_SELECT=""; FISHER_ALPHA=""; FISHER_TOP_K=""; FISHER_TOP_PERCENT=""; GRID_FISHER=(); GRID_FISHER_BARE=0
MIN_SAMPLES=""; KMER_K=""; WITH_V=0; TRIM_START=""; TRIM_END=""; LIST_ONLY=0; EXCLUDE=()
PREV_MODE=""; MIN_PREV=""; MIN_PREV_COUNT=""
KM_DEVICE=cpu; GRID_KM_K=(); GRID_KM_Q=(); KM_NITER=""; KM_NREDO=""; KM_MAXPTS=""; KM_SPHERICAL=""

die() { echo "ERROR: $*" >&2; exit 2; }
is_int() { [[ "$1" =~ ^-?[0-9]+$ ]]; }
is_num() { [[ "$1" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; }

# split "--flag=value" into "--flag" "value"
ARGS=()
for a in "$@"; do
  if [[ "$a" == --*=* ]]; then ARGS+=("${a%%=*}" "${a#*=}"); else ARGS+=("$a"); fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

LIST=()   # filled by collect: all following tokens up to the next flag
NCONSUMED=0
collect() { LIST=(); NCONSUMED=0
  while [ $# -gt 0 ] && [[ "$1" != -* || "$1" =~ ^-[0-9] ]]; do LIST+=("$1"); NCONSUMED=$((NCONSUMED + 1)); shift; done
  [ ${#LIST[@]} -gt 0 ] || die "$FLAG needs at least one value"; }
one() { [ $# -gt 0 ] || die "$FLAG needs a value"; REPLY="$1"; }

while [ $# -gt 0 ]; do
  FLAG="$1"; shift
  case "$FLAG" in
    -h|--help)  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
    --seq_filter) collect "$@"; SEQ_FILTERS=("${LIST[@]}"); shift "$NCONSUMED";;
    --data_dir) one "$@"; DATA_DIR="$REPLY"; shift;;
    --output_dir) one "$@"; OUTPUT_DIR="$REPLY"; shift;;
    --xgboost_device)   one "$@"; DEVICE="$REPLY"; shift;;
    --random_seed) one "$@"; SEED="$REPLY"; shift;;
    --n_jobs)   one "$@"; N_JOBS="$REPLY"; shift;;
    --max_samples) one "$@"; MAX_SAMPLES="$REPLY"; shift;;
    --exclude_diseases) collect "$@"; EXCLUDE=("${LIST[@]}"); shift "$NCONSUMED";;
    --force)    FORCE=1;;
    --dry_run)  DRY_RUN=1;;
    --conda_env) one "$@"; CONDA_ENV="$REPLY"; shift;;
    --sceptr_variant) one "$@"; VARIANT="$REPLY"; shift;;
    --grid_kmeans_k) collect "$@"; GRID_KM_K=("${LIST[@]}"); shift "$NCONSUMED";;
    --grid_kmeans_radius_q) collect "$@"; GRID_KM_Q=("${LIST[@]}"); shift "$NCONSUMED";;
    --kmeans_device) one "$@"; KM_DEVICE="$REPLY"; shift;;
    --kmeans_niter) one "$@"; KM_NITER="$REPLY"; shift;;
    --kmeans_nredo) one "$@"; KM_NREDO="$REPLY"; shift;;
    --kmeans_max_points_per_centroid) one "$@"; KM_MAXPTS="$REPLY"; shift;;
    --kmeans_spherical) one "$@"; KM_SPHERICAL="$REPLY"; shift;;
    --python)   one "$@"; PYTHON="$REPLY"; shift;;
    --log_dir)  one "$@"; LOG_DIR="$REPLY"; shift;;
    --n_splits) one "$@"; N_SPLITS="$REPLY"; shift;;
    --min_cluster_size) one "$@"; MIN_CLUSTER_SIZE="$REPLY"; shift;;
    --n_trials) one "$@"; N_TRIALS="$REPLY"; shift;;
    --fisher_select) one "$@"; FISHER_SELECT="$REPLY"; shift;;
    --fisher_alpha) one "$@"; FISHER_ALPHA="$REPLY"; shift;;
    --fisher_top_k) one "$@"; FISHER_TOP_K="$REPLY"; shift;;
    --fisher_top_percent) one "$@"; FISHER_TOP_PERCENT="$REPLY"; shift;;
    --grid_fisher) GRID_FISHER=(); GRID_FISHER_BARE=1
                   while [ $# -gt 0 ] && [[ "$1" != -* || "$1" =~ ^-[0-9] ]]; do GRID_FISHER+=("$1"); GRID_FISHER_BARE=0; shift; done;;
    --min_samples) one "$@"; MIN_SAMPLES="$REPLY"; shift;;
    --kmer_k)   one "$@"; KMER_K="$REPLY"; shift;;
    --with_v)   WITH_V=1;;
    --kmer_trim_start) one "$@"; TRIM_START="$REPLY"; shift;;
    --kmer_trim_end)   one "$@"; TRIM_END="$REPLY"; shift;;
    --sequence_list_only) LIST_ONLY=1;;
    --prevalence_mode) one "$@"; PREV_MODE="$REPLY"; shift;;
    --min_prevalence) one "$@"; MIN_PREV="$REPLY"; shift;;
    --min_prevalence_count) one "$@"; MIN_PREV_COUNT="$REPLY"; shift;;
    *) die "unknown flag '$FLAG' (see --help)";;
  esac
done

# ---- validation ------------------------------------------------------------------------------------
case "$KM_DEVICE" in cpu|cuda) ;; *) die "--kmeans_device must be cpu or cuda";; esac
[ -z "$KM_SPHERICAL" ] || case "$KM_SPHERICAL" in auto|yes|no) ;; *) die "--kmeans_spherical must be auto, yes or no";; esac
for v in ${GRID_KM_K[@]+"${GRID_KM_K[@]}"}; do is_int "$v" || die "--grid_kmeans_k values must be integers, got '$v'"; done
for v in ${GRID_KM_Q[@]+"${GRID_KM_Q[@]}"}; do
  is_num "$v" && awk "BEGIN{exit !($v > 0 && $v <= 1)}" || die "--grid_kmeans_radius_q values must be in (0, 1], got '$v'"
done
for kv in "kmeans_niter:$KM_NITER" "kmeans_nredo:$KM_NREDO" "kmeans_max_points_per_centroid:$KM_MAXPTS"; do
  [ -z "${kv#*:}" ] || is_int "${kv#*:}" || die "--${kv%%:*} must be an integer, got '${kv#*:}'"
done
case "$DEVICE" in cuda|cpu) ;; *) die "--xgboost_device must be cuda or cpu";; esac
for f in "${SEQ_FILTERS[@]}"; do
  case "$f" in none|clonotype|kmer|vj) ;; *) die "--seq_filter values must be none, clonotype, kmer or vj (got '$f')";; esac
  [ "$LIST_ONLY" = 1 ] && [ "$f" = none ] && die "--sequence_list_only needs a --seq_filter other than none"
done
[ -z "$FISHER_SELECT" ] || case "$FISHER_SELECT" in fdr|topk|percentile) ;; *) die "--fisher_select must be fdr, topk or percentile";; esac
[ -z "$PREV_MODE" ] || case "$PREV_MODE" in class_aware|global) ;; *) die "--prevalence_mode must be class_aware or global";; esac
for kv in "random_seed:$SEED" "n_jobs:$N_JOBS" "max_samples:$MAX_SAMPLES" "n_splits:$N_SPLITS" "min_cluster_size:$MIN_CLUSTER_SIZE" \
          "n_trials:$N_TRIALS" "fisher_top_k:$FISHER_TOP_K" "min_samples:$MIN_SAMPLES" "kmer_k:$KMER_K" \
          "min_prevalence_count:$MIN_PREV_COUNT" "kmer_trim_start:$TRIM_START" "kmer_trim_end:$TRIM_END"; do
  [ -z "${kv#*:}" ] || is_int "${kv#*:}" || die "--${kv%%:*} must be an integer, got '${kv#*:}'"
done
for kv in "fisher_alpha:$FISHER_ALPHA" "fisher_top_percent:$FISHER_TOP_PERCENT" "min_prevalence:$MIN_PREV"; do
  [ -z "${kv#*:}" ] || is_num "${kv#*:}" || die "--${kv%%:*} must be a number, got '${kv#*:}'"
done
for v in ${GRID_FISHER[@]+"${GRID_FISHER[@]}"}; do is_num "$v" || die "grid values must be numbers, got '$v'"; done

NAME=sceptr; [ "$VARIANT" != default ] && NAME="sceptr_$VARIANT"
if [ "$LIST_ONLY" = 1 ]; then
  [ -d "$DATA_DIR/raw/train" ] || die "missing $DATA_DIR/raw/train"
else
  [ -f "$DATA_DIR/embeddings/$NAME/train/embedding_config.json" ] || \
    die "no train embeddings in $DATA_DIR/embeddings/$NAME/train (run scripts/run_embeddings.sh first)"
fi

# ---- environment -----------------------------------------------------------------------------------
if [ "$DRY_RUN" != 1 ] && [ -n "$CONDA_ENV" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# FAISS must be importable, and the GPU path must work when --kmeans_device cuda: check it up front
if [ "$DRY_RUN" != 1 ]; then
  "$PYTHON" -c "import sys; sys.path.insert(0, '$ROOT'); from src import clustering as cl; cl.check_available('$KM_DEVICE'); print('FAISS ok for device $KM_DEVICE')"
fi

# XGBoost is the only GPU user here: fail early and clearly if it has no CUDA support
if [ "$DEVICE" = cuda ] && [ "$DRY_RUN" != 1 ]; then
  "$PYTHON" - <<'PYEOF'
import sys
import xgboost as xgb
if not xgb.build_info().get("USE_CUDA", False):
    sys.exit("ERROR: --xgboost_device cuda, but this XGBoost build has no CUDA support")
print("XGBoost %s has CUDA support" % xgb.__version__)
PYEOF
fi

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
T_ALL=$SECONDS

run_one() {   # run_one <seq_filter>
  local f="$1"
  local res="$OUTPUT_DIR/tuning/${NAME}_${f}_kmeans/hyperparameters.json"
  [ "$LIST_ONLY" = 1 ] && res="$OUTPUT_DIR/tuning/${NAME}_${f}_kmeans/sequence_list.txt"
  if [ "$FORCE" != 1 ] && [ -f "$res" ]; then
    echo "[skip] ${NAME}_${f}_kmeans: $(basename "$res") exists (--force to redo)"; return 0
  fi
  local cmd=("$PYTHON" "$ROOT/scripts/tune_hyperparameters.py" --seq_filter "$f" --sceptr_variant "$VARIANT"
             --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" --xgboost_device "$DEVICE" --kmeans_device "$KM_DEVICE")
  [ -n "$SEED" ] && cmd+=(--random_seed "$SEED")
  [ -n "$N_JOBS" ] && cmd+=(--n_jobs "$N_JOBS")
  [ -n "$MAX_SAMPLES" ] && cmd+=(--max_samples "$MAX_SAMPLES")
  [ ${#EXCLUDE[@]} -gt 0 ] && cmd+=(--exclude_diseases "${EXCLUDE[@]}")
  [ -n "$N_SPLITS" ] && cmd+=(--n_splits "$N_SPLITS")
  [ ${#GRID_KM_K[@]} -gt 0 ] && cmd+=(--grid_kmeans_k "${GRID_KM_K[@]}")
  [ ${#GRID_KM_Q[@]} -gt 0 ] && cmd+=(--grid_kmeans_radius_q "${GRID_KM_Q[@]}")
  [ -n "$KM_NITER" ] && cmd+=(--kmeans_niter "$KM_NITER")
  [ -n "$KM_NREDO" ] && cmd+=(--kmeans_nredo "$KM_NREDO")
  [ -n "$KM_MAXPTS" ] && cmd+=(--kmeans_max_points_per_centroid "$KM_MAXPTS")
  [ -n "$KM_SPHERICAL" ] && cmd+=(--kmeans_spherical "$KM_SPHERICAL")
  [ -n "$MIN_CLUSTER_SIZE" ] && cmd+=(--min_cluster_size "$MIN_CLUSTER_SIZE")
  [ -n "$N_TRIALS" ] && cmd+=(--n_trials "$N_TRIALS")
  [ -n "$FISHER_SELECT" ] && cmd+=(--fisher_select "$FISHER_SELECT")
  [ -n "$FISHER_ALPHA" ] && cmd+=(--fisher_alpha "$FISHER_ALPHA")
  [ -n "$FISHER_TOP_K" ] && cmd+=(--fisher_top_k "$FISHER_TOP_K")
  [ -n "$FISHER_TOP_PERCENT" ] && cmd+=(--fisher_top_percent "$FISHER_TOP_PERCENT")
  [ ${#GRID_FISHER[@]} -gt 0 ] && cmd+=(--grid_fisher "${GRID_FISHER[@]}")
  [ "$GRID_FISHER_BARE" = 1 ] && cmd+=(--grid_fisher)
  [ -n "$MIN_SAMPLES" ] && cmd+=(--min_samples "$MIN_SAMPLES")
  [ -n "$KMER_K" ] && cmd+=(--kmer_k "$KMER_K")
  [ "$WITH_V" = 1 ] && cmd+=(--with_v)
  [ -n "$TRIM_START" ] && cmd+=(--kmer_trim_start "$TRIM_START")
  [ -n "$TRIM_END" ] && cmd+=(--kmer_trim_end "$TRIM_END")
  [ "$LIST_ONLY" = 1 ] && cmd+=(--sequence_list_only)
  [ -n "$PREV_MODE" ] && cmd+=(--prevalence_mode "$PREV_MODE")
  [ -n "$MIN_PREV" ] && cmd+=(--min_prevalence "$MIN_PREV")
  [ -n "$MIN_PREV_COUNT" ] && cmd+=(--min_prevalence_count "$MIN_PREV_COUNT")
  echo; echo "=== ${NAME} / seq_filter=${f} ==="; printf '%q ' "${cmd[@]}"; echo
  [ "$DRY_RUN" = 1 ] && return 0
  local t0=$SECONDS
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/tuning_${NAME}_${f}_${STAMP}.log"     # pipefail: a failure stops the script
  echo "[done] ${NAME}_${f}_kmeans in $((SECONDS - t0)) s"
}

for f in "${SEQ_FILTERS[@]}"; do run_one "$f"; done
echo; echo "Finished (${NAME} x ${SEQ_FILTERS[*]}) in $((SECONDS - T_ALL)) s; results under $OUTPUT_DIR/tuning"
