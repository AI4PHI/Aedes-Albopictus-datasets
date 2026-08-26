#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

CONDA_BIN="${REPO_ROOT}/.conda/aiedes-data/bin"
if [[ -x "${CONDA_BIN}/python" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi

mkdir -p "${SCRIPT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib}"

# Run the Python script
python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex

# ERA5-Land historical data
python pair_ecdc_copernicus_data.py --year 2020 --climate-source era5_land

# tests
# python tests/compare_cordex_results.py
