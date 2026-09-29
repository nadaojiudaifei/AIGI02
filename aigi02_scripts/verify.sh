#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m aigi02 verify-upstream
python -m pytest aigi02_tests -q -rs
python -m aigi02 smoke --output "${1:-aigi02_runs/smoke_$(date +%Y%m%d_%H%M%S)}"
