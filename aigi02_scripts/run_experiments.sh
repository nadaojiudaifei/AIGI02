#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$#" -lt 2 ]; then
  echo 'Usage: bash aigi02_scripts/run_experiments.sh SPEC_YAML PLAN_JSON [--resume]' >&2
  exit 2
fi
python -m aigi02 matrix --spec "$1" --output "$2"
plan=$2
shift 2
python -m aigi02 run-matrix --plan "$plan" --execute "$@"
