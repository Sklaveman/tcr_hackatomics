#!/usr/bin/env bash
# Run the test suite gently: few threads, lowest priority, one module at a time, optional cool-down pauses.
#
# Usage:  bash scripts/run_tests.sh [flags]
#
#   --quick (default)       the fast unit tests only (feature selection, filters, clustering, k-means, evaluation,
#                           preprocessing, exclusion): about 1-2 minutes
#   --full                  everything, including the end-to-end and embedding-generator tests (many minutes;
#                           those run the CLIs as subprocesses)
#   --only M [M ...]       run just these modules, e.g.  --only test_kmeans test_feature_selection
#   --env NAME             conda env to run in (needs faiss, xgboost, sceptr: sceptr_env; tests that need a
#                           missing package skip themselves)                                                  [sceptr_env]
#   --threads N            threads per library (OpenMP, MKL, OpenBLAS, numba)                   [2]
#   --pause S              seconds to sleep between modules, to let the CPU cool down                       [0 quick, 30 full]
#   --nice N               process priority (higher = lower priority, 19 is the lowest)                       [19]
#   -h, --help             print this header
#
# Examples:
#   bash scripts/run_tests.sh
#   bash scripts/run_tests.sh --full --threads 1 --pause 60
#   bash scripts/run_tests.sh --env sceptr_env --only test_kmeans test_sceptr_micro
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=quick; ONLY=(); ENV_NAME=sceptr_env; THREADS=2; PAUSE=""; NICE=19
die() { echo "ERROR: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  f="$1"; shift
  case "$f" in
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
    --quick) MODE=quick;;
    --full)  MODE=full;;
    --only)  ONLY=(); while [ $# -gt 0 ] && [[ "$1" != -* ]]; do ONLY+=("$1"); shift; done
             [ ${#ONLY[@]} -gt 0 ] || die "--only needs at least one module name";;
    --env)     [ $# -gt 0 ] || die "--env needs a value"; ENV_NAME="$1"; shift;;
    --threads) [ $# -gt 0 ] || die "--threads needs a value"; THREADS="$1"; shift;;
    --pause)   [ $# -gt 0 ] || die "--pause needs a value"; PAUSE="$1"; shift;;
    --nice)    [ $# -gt 0 ] || die "--nice needs a value"; NICE="$1"; shift;;
    *) die "unknown flag '$f' (see --help)";;
  esac
done
for kv in "threads:$THREADS" "pause:${PAUSE:-0}" "nice:$NICE"; do
  [[ "${kv#*:}" =~ ^[0-9]+$ ]] || die "--${kv%%:*} must be a non-negative integer"
done

QUICK=(test_feature_selection test_filter_variants test_exclusion test_preprocessing test_kmeans
       test_model_evaluation test_clustering)
FULL=("${QUICK[@]}" test_sceptr_micro test_pipeline_micro)
if [ ${#ONLY[@]} -gt 0 ]; then MODULES=("${ONLY[@]}")
elif [ "$MODE" = full ]; then MODULES=("${FULL[@]}"); [ -n "$PAUSE" ] || PAUSE=30
else MODULES=("${QUICK[@]}"); fi
PAUSE="${PAUSE:-0}"

# threads: every library that would otherwise use all cores
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       VECLIB_MAXIMUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS"

if [ -n "$ENV_NAME" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi
cd "$ROOT"
echo "env=${CONDA_DEFAULT_ENV:-?} threads=$THREADS nice=$NICE pause=${PAUSE}s modules: ${MODULES[*]}"
fail=0; i=0
for m in "${MODULES[@]}"; do
  i=$((i + 1)); t0=$SECONDS
  echo; echo "=== [$i/${#MODULES[@]}] $m ==="
  log="$(mktemp)"; rc=0
  nice -n "$NICE" python -m unittest "tests.$m" > "$log" 2>&1 || rc=$?
  grep -vE "Building HDF5|Warning|warn\(|_set_array|nested_tensor" "$log" || true
  rm -f "$log"
  [ "$rc" = 0 ] || { echo "!! $m FAILED (exit $rc)"; fail=1; }
  echo "($((SECONDS - t0)) s)"
  [ "$PAUSE" -gt 0 ] && [ "$i" -lt "${#MODULES[@]}" ] && sleep "$PAUSE"
done
[ "$fail" = 0 ] && echo && echo "All selected test modules passed." || { echo; echo "Some test modules FAILED."; exit 1; }
