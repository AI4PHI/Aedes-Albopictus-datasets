#!/bin/bash
set -euo pipefail

# Optional: Activate conda environment
# Uncomment and set your environment name
# conda activate your_env_name

# Ensure we always operate from the project root (counter/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

# Prefer the repo-local conda environment when it exists.
CONDA_BIN="${REPO_ROOT}/.conda/aiedes-data/bin"
if [[ -x "${CONDA_BIN}/python" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi

# Keep runtime caches and the climate download window predictable in this repo.
mkdir -p "${SCRIPT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib}"
export ERA5_START_DATE="${ERA5_START_DATE:-2019-10-01}"
export ERA5_END_DATE="${ERA5_END_DATE:-2020-12-31}"

# Complete pipeline:
python src/albopictus.py
echo "Albopictus processing summary: output_stats/albopictus_summary.json"
echo "Albopictus output: output_data/albopictus.csv.zip and output_data/albopictus.pkl"

# If you want a clean, deterministic climate rebuild:
#   REBUILD_CLIMATE=1 ./make_counter_dataset.sh
if [[ "${REBUILD_CLIMATE:-0}" == "1" ]]; then
  echo "Rebuilding climate cache (raw + processed)..."
  rm -rf "${SCRIPT_DIR}/input_data/climate/raw" "${SCRIPT_DIR}/input_data/climate/processed"
  FORCE="--force-redownload"
else
  echo "Resuming climate cache where possible..."
  echo "Climate window: ${ERA5_START_DATE} to ${ERA5_END_DATE}"
  FORCE=""
fi

python src/copernicus_data.py --enable-downloads ${FORCE}

# Result: AIMSurv_albopictus_2020_era5_land.csv.zip and AIMSurv_albopictus_2020_era5_land.pkl (final database)
