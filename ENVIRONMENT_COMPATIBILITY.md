# Environment Compatibility

## Conda Environment

This project can be installed with the repository-provided `aiedes-data` conda environment.

### Package Requirements

| Package | Required |
|---------|----------|
| python | ≥3.12 |
| pandas | ≥2.2.0 |
| numpy | ≥2.1.0 |
| xarray | ≥2024.1.0 |
| geopandas | ≥1.0.0 |
| fiona | ≥1.10.0 |
| contextily | ≥1.6.0 |
| netCDF4 | ≥1.7.0 |
| cdsapi | ≥0.7.0 |
| matplotlib | ≥3.10.0 |
| seaborn | ≥0.13.0 |
| tqdm | ≥4.67.0 |
| scipy | ≥1.15.0 |
| scikit-learn | ≥1.6.0 |

### Usage

Create and activate the environment, then run the code:

```bash
conda env create -f environment.yml
conda activate aiedes-data

# From the repository root
cd data/classifier
python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex
```

## Alternative Environments

If you prefer another environment manager, use `requirements.txt` as the package reference.
