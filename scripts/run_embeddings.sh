#!/usr/bin/env bash
# SCEPTR embedding generation for every dataset (scripts/generate_sceptr_embeddings.py).
#
# The datasets are processed in the given order. SCEPTR is pretrained, so every dataset is just embedded.
# A dataset that already has an embedding_config.json in its output folder is skipped unless --force is given,
# so an interrupted run can be restarted with the same command.
#
# Usage:  bash scripts/run_embeddings.sh [flags]
# Flags take a value ("--device cuda" or "--device=cuda"); the ones marked (switch) take none.
#
# ----- general ---------------------------------------------------------------------------------
#   --datasets D [D ...]     datasets under <data_dir>/raw, in processing order    [train test_bulk test_sc]
#   --data_dir PATH          output root: embeddings go to <data_dir>/embeddings/sceptr/<dataset>
#                            (embeddings/sceptr_<variant>/<dataset> for another --sceptr_variant)  [<project>/data]
#   --raw_dir PATH           folder holding the raw datasets (<raw_dir>/<dataset>)           [<data_dir>/raw]
#   --device D               cpu | cuda (cuda is checked up front and fails clearly if unavailable)   [cuda]
#   --random_seed N          random seed (SCEPTR is deterministic; the seed only picks the --max_samples subset) [42]
#   --batch_size N           TCRs per forward pass                                             [512]
#   --n_jobs N               CPU threads for PyTorch                                           [all cores]
#   --max_samples N          embed only N samples per dataset (smoke test)                     [all]
#   --sequence_list FILE     text file, one cdr3_aa+V_J key per line (Fisher survivors); `train` is then
#                            embedded for these sequences only, other datasets always in full  [none]
#   --h5_dtype T             float32 | float16: storage type of the vectors in per_patient_tcr_embed.h5. float16
#                            halves the vectors (nearest centroid unchanged for 99.998 % of sequences on tutorial
#                            data, not bit-identical); do not mix types within one experiment    [float32]
#   --delete_csv (switch)    after each dataset is written, run the full checks (check_embeddings.py: values and
#                            sequences vs the raw data) and delete the per-sample CSVs only if all pass; if a check
#                            fails the CSVs are kept. The pipeline reads only the HDF5. Existing datasets:
#                            scripts/compact_embeddings.py
#   --force (switch)         redo datasets that already have an embedding_config.json
#   --dry_run (switch)       print the commands and exit without running anything
# ----- environment and logs ----------------------------------------------------------------------
#   --conda_env NAME         conda env to activate if it is not already active ('' = do not activate) [sceptr_env]
#   --python EXE             python executable to use after activation                         [python]
#   --log_dir PATH           one log file per dataset is written here                          [<project>/logs]
# ----- SCEPTR (beta chain; 64-d for the default variant) ------------------------------------------
#   --sceptr_variant V       SCEPTR model variant (default | b_sceptr | cdr3_only | finetuned | large | small |
#                            tiny | blosum | ...). Output goes to embeddings/sceptr, or
#                            embeddings/sceptr_<variant> for any other variant                [default]
#   --sceptr_invalid_v_policy P  drop | error: rows whose V gene SCEPTR rejects (non-functional or unknown
#                            names such as TRBV1, 7-5, 12-1, TRBVA-1; ~1.3 % of rows). drop logs and
#                            removes them (not needed with --sceptr_variant cdr3_only)          [drop]
#   --sceptr_chunk_size N    unique sequences per call to the model (bounds memory)            [200000]
# ----- other -------------------------------------------------------------------------------------
#   -h, --help               print this header
#
# Examples:
#   bash scripts/run_embeddings.sh --device cuda --batch_size 2048
#   bash scripts/run_embeddings.sh --device cuda --sequence_list results/run/tuning/sceptr_kmer_kmeans/sequence_list.txt
#   bash scripts/run_embeddings.sh --device cuda --h5_dtype float16 --delete_csv
#   bash scripts/run_embeddings.sh --max_samples 3 --device cpu --dry_run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS=(train test_bulk test_sc); DATA_DIR="$ROOT/data"; RAW_DIR=""; DEVICE=cuda; SEED=42
BATCH_SIZE=512; N_JOBS=""; MAX_SAMPLES=""; SEQUENCE_LIST=""; FORCE=0; DRY_RUN=0
CONDA_ENV=sceptr_env; PYTHON=python; LOG_DIR="$ROOT/logs"
SCEPTR_VARIANT=default; SCEPTR_V_POLICY=drop; SCEPTR_CHUNK=200000; H5_DTYPE=float32; DELETE_CSV=0

die() { echo "ERROR: $*" >&2; exit 2; }
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# split "--flag=value" into "--flag" "value"
ARGS=()
for a in "$@"; do
  if [[ "$a" == --*=* ]]; then ARGS+=("${a%%=*}" "${a#*=}"); else ARGS+=("$a"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

while [ $# -gt 0 ]; do
  f="$1"; shift
  val() { [ $# -gt 0 ] || die "$f needs a value"; REPLY="$1"; }
  case "$f" in
    -h|--help)  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
    --datasets) DATASETS=(); while [ $# -gt 0 ] && [[ "$1" != -* ]]; do DATASETS+=("$1"); shift; done
                [ ${#DATASETS[@]} -gt 0 ] || die "--datasets needs at least one name";;
    --data_dir) val "$@"; DATA_DIR="$REPLY"; shift;;
    --raw_dir)  val "$@"; RAW_DIR="$REPLY"; shift;;
    --device)   val "$@"; DEVICE="$REPLY"; shift;;
    --random_seed) val "$@"; SEED="$REPLY"; shift;;
    --batch_size)  val "$@"; BATCH_SIZE="$REPLY"; shift;;
    --n_jobs)      val "$@"; N_JOBS="$REPLY"; shift;;
    --max_samples) val "$@"; MAX_SAMPLES="$REPLY"; shift;;
    --sequence_list) val "$@"; SEQUENCE_LIST="$REPLY"; shift;;
    --h5_dtype) val "$@"; H5_DTYPE="$REPLY"; shift;;
    --delete_csv) DELETE_CSV=1;;
    --force)    FORCE=1;;
    --dry_run)  DRY_RUN=1;;
    --conda_env) val "$@"; CONDA_ENV="$REPLY"; shift;;
    --sceptr_variant) val "$@"; SCEPTR_VARIANT="$REPLY"; shift;;
    --sceptr_invalid_v_policy) val "$@"; SCEPTR_V_POLICY="$REPLY"; shift;;
    --sceptr_chunk_size) val "$@"; SCEPTR_CHUNK="$REPLY"; shift;;
    --python)   val "$@"; PYTHON="$REPLY"; shift;;
    --log_dir)  val "$@"; LOG_DIR="$REPLY"; shift;;
    *) die "unknown flag '$f' (see --help)";;
  esac
done

case "$SCEPTR_V_POLICY" in error|drop) ;; *) die "--sceptr_invalid_v_policy must be error or drop";; esac
case "$SCEPTR_VARIANT" in default|b_sceptr|cdr3_only|finetuned|large|small|tiny|blosum|average_pooling|left_aligned|mlm_only|cdr3_only_mlm_only|dropout_noise_only|shuffled_data|synthetic_data) ;; *) die "unknown --sceptr_variant '$SCEPTR_VARIANT' (see --help)";; esac
case "$H5_DTYPE" in float32|float16) ;; *) die "--h5_dtype must be float32 or float16";; esac
case "$DEVICE" in cuda|cpu) ;; *) die "--device must be cuda or cpu";; esac
for kv in "random_seed:$SEED" "batch_size:$BATCH_SIZE" "n_jobs:$N_JOBS" "max_samples:$MAX_SAMPLES" \
          "sceptr_chunk_size:$SCEPTR_CHUNK"; do
  [ -z "${kv#*:}" ] || is_int "${kv#*:}" || die "--${kv%%:*} must be a non-negative integer, got '${kv#*:}'"
done
RAW_DIR="${RAW_DIR:-$DATA_DIR/raw}"
for d in "${DATASETS[@]}"; do [ -d "$RAW_DIR/$d" ] || die "missing dataset folder $RAW_DIR/$d"; done
[ -z "$SEQUENCE_LIST" ] || [ -f "$SEQUENCE_LIST" ] || die "--sequence_list not found: $SEQUENCE_LIST"

if [ "$DRY_RUN" != 1 ] && [ -n "$CONDA_ENV" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# fail early and clearly when cuda is requested but PyTorch cannot use it
if [ "$DEVICE" = cuda ] && [ "$DRY_RUN" != 1 ]; then
  "$PYTHON" - <<'PYEOF'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: --device cuda, but PyTorch has no CUDA device")
print("PyTorch sees %d GPU(s)" % torch.cuda.device_count())
PYEOF
fi

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
T_ALL=$SECONDS
NAME=sceptr; [ "$SCEPTR_VARIANT" != default ] && NAME="sceptr_$SCEPTR_VARIANT"

run_one() {   # run_one <dataset>
  local ds="$1" cmd
  local out="$DATA_DIR/embeddings/$NAME/$ds"
  if [ "$FORCE" != 1 ] && [ -f "$out/embedding_config.json" ]; then
    echo "[skip] $NAME/$ds: embedding_config.json exists (--force to redo)"; return 0
  fi
  cmd=("$PYTHON" "$ROOT/scripts/generate_sceptr_embeddings.py" --input_path "$RAW_DIR/$ds"
       --output_path "$out" --variant "$SCEPTR_VARIANT" --device "$DEVICE"
       --batch_size "$BATCH_SIZE" --chunk_size "$SCEPTR_CHUNK"
       --invalid_v_policy "$SCEPTR_V_POLICY" --random_seed "$SEED" --h5_dtype "$H5_DTYPE")
  [ "$DELETE_CSV" = 1 ] && cmd+=(--delete_csv)
  [ -n "$N_JOBS" ] && cmd+=(--n_jobs "$N_JOBS")
  [ -n "$MAX_SAMPLES" ] && cmd+=(--max_samples "$MAX_SAMPLES")
  [ -n "$SEQUENCE_LIST" ] && [ "$ds" = train ] && cmd+=(--sequence_list "$SEQUENCE_LIST")
  echo; echo "=== $NAME / $ds ==="; printf '%q ' "${cmd[@]}"; echo
  [ "$DRY_RUN" = 1 ] && return 0
  local t0=$SECONDS
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/${NAME}_${ds}_${STAMP}.log"     # pipefail: a failure stops the script
  echo "[done] $NAME/$ds in $((SECONDS - t0)) s"
}

for ds in "${DATASETS[@]}"; do run_one "$ds"; done
echo; echo "Finished ($NAME) in $((SECONDS - T_ALL)) s; embeddings under $DATA_DIR/embeddings"
