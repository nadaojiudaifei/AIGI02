#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [ "$#" -lt 2 ]; then
  echo 'Usage: bash aigi02_scripts/run_pipeline.sh CONFIG OUTPUT [NPROC] [--resume]' >&2
  exit 2
fi
config=$1
output=$2
nproc=${3:-1}
shift 2
if [ "$#" -gt 0 ]; then shift; fi
python -m aigi02 pipeline --config "$config" --output "$output" --nproc "$nproc" --execute "$@"
