#!/bin/bash

# Optional: Activate conda environment
# Uncomment and set your environment name
# conda activate your_env_name

# Run the Python script
python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex

# ERA5-Land historical data
python pair_ecdc_copernicus_data.py --year 2020 --climate-source era5_land