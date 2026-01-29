#!/bin/bash

# Optional: Activate conda environment
# Uncomment and set your environment name
# conda activate your_env_name

# Complete pipeline:
python src/albopictus.py
python src/copernicus_data.py --enable-downloads --force-redownload

# Result: albopictus_with_climate_3m.pkl (final database)