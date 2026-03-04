# Environment Compatibility

## climate_env Environment

This project is fully compatible with the existing `climate_env` conda environment.

### Verified Package Versions (as of 2026-03-04)

| Package | Version in climate_env | Required | Status |
|---------|----------------------|----------|--------|
| python | 3.12.9 | ≥3.12 | ✅ |
| pandas | 2.2.3 | ≥2.2.0 | ✅ |
| numpy | 2.1.3 | ≥2.1.0 | ✅ |
| xarray | 2025.1.2 | ≥2024.1.0 | ✅ |
| geopandas | 1.0.1 | ≥1.0.0 | ✅ |
| fiona | 1.10.1 | ≥1.10.0 | ✅ |
| contextily | 1.6.2 | ≥1.6.0 | ✅ |
| netCDF4 | 1.7.2 | ≥1.7.0 | ✅ |
| cdsapi | 0.7.5 | ≥0.7.0 | ✅ |
| matplotlib | 3.10.1 | ≥3.10.0 | ✅ |
| seaborn | 0.13.2 | ≥0.13.0 | ✅ |
| tqdm | 4.67.1 | ≥4.67.0 | ✅ |
| scipy | 1.15.2 | ≥1.15.0 | ✅ |
| scikit-learn | 1.6.1 | ≥1.6.0 | ✅ |

### Usage with climate_env

Simply activate the environment and run the code:

```bash
conda activate climate_env
cd data/classifier
python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex
```

All dependencies are already satisfied - no additional installation required!

## Alternative Environments

If you prefer to create a new environment, use:
- `environment.yml` for conda
- `requirements.txt` for pip

Both are configured to match the versions in `climate_env` for consistency.