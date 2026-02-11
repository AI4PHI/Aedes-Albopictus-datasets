#!/bin/bash
set -euo pipefail

# Optional: Activate conda environment
# Uncomment and set your environment name
# conda activate your_env_name

# Ensure we always operate from the project root (counter/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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
  FORCE=""
fi

python src/copernicus_data.py --enable-downloads ${FORCE}

# Result: albopictus_with_climate_3m.csv.zip and albopictus_with_climate_3m.pkl (final database)