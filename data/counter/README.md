# Albopictus Trap Data Processing with Climate Data

This directory contains a polished Python script for processing albopictus trap data by pairing it with Copernicus climate data.

## Features

- **Data Loading & Filtering**: Load trap data from pickle files and filter by date range
- **Climate Data Integration**: Extract multiple climate variables (precipitation, temperature, wind, etc.)
- **Temporal Aggregation**: Create both daily and monthly climate data summaries
- **Automatic Downloads**: Optional automatic downloading of missing climate data from Copernicus CDS
- **Error Handling**: Robust error handling with retry logic for downloads
- **Flexible Configuration**: Command-line arguments for all major parameters

## Setup

### Basic Installation

```bash
pip install -r requirements.txt
```

### For Automatic Downloads (Optional)

To enable automatic downloading of missing climate data:

1. **Install CDS API**:
   ```bash
   pip install cdsapi
   ```

2. **Register for CDS Account**:
   - Go to https://cds.climate.copernicus.eu/api-how-to
   - Create an account and get your API key

3. **Configure API Credentials**:
   Create `~/.cdsapirc` with your credentials:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_API_KEY_HERE
   ```

## Usage

### Basic Usage

```bash
# Uses default input file: ../output_data/albopictus.pkl
python src/pair_traps_and_copernicus_data_polished.py
```

### With Custom Input File

```bash
python src/pair_traps_and_copernicus_data_polished.py --input-file path/to/your/albopictus.pkl
```

### With Downloads Enabled

```bash
python src/pair_traps_and_copernicus_data_polished.py --enable-downloads
```

### Advanced Options

```bash
python src/pair_traps_and_copernicus_data_polished.py \
  --input-file /path/to/albopictus.pkl \
  --output my_results \
  --start-date 2019-01-01 \
  --end-date 2022-01-01 \
  --climate-path /path/to/climate/data \
  --enable-downloads \
  --force-redownload \
  --log-level DEBUG
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input-file` | Path to input pickle file | `../output_data/albopictus.pkl` |
| `-o, --output` | Output filename prefix | `albopictus_with_climate_3m` |
| `--climate-path` | Path to climate data directory | `../input_data/copernicus_climate_data/europe/data/` |
| `--start-date` | Start date for filtering (YYYY-MM-DD) | `2020-01-01` |
| `--end-date` | End date for filtering (YYYY-MM-DD) | `2021-01-01` |
| `--enable-downloads` | Enable automatic downloading | `False` |
| `--force-redownload` | Force redownload existing files | `False` |
| `--log-level` | Logging level | `INFO` |

## Climate Variables

The script processes the following climate variables:

- **Precipitation**: `total_precipitation`
- **Wind**: `10m_u_component_of_wind`, `10m_v_component_of_wind`
- **Temperature**: `2m_temperature`, `2m_dewpoint_temperature`
- **Soil**: `volumetric_soil_water_layer_1`

Each variable is processed to create:
- **Daily aggregates**: Over an 89-day window before each trap sampling date
- **Monthly aggregates**: 3-month averages before each sampling date

## File Structure

```
AIedes_data/counter/
├── src/
│   ├── pair_traps_and_copernicus_data_polished.py  # Main script
│   └── process_copernicus_data.py                  # Climate processing functions
├── copernicus_downloader.py                       # Download functionality
├── requirements.txt                               # Python dependencies
├── README.md                                     # This file
└── input_data/                                   # Climate data storage
    └── copernicus_climate_data/
        ├── raw/                                  # Raw downloaded files
        └── europe/data/                          # Processed files
```

## Output

The script creates two output files:
- `{output_prefix}.csv`: CSV format for easy analysis
- `{output_prefix}.pkl`: Pickle format for Python processing

## Download Process

When `--enable-downloads` is used:

1. **Check for existing files** in the expected directory structure
2. **Download missing data** from Copernicus CDS using ERA5-Land dataset
3. **Process raw data** into the expected format (daily stats/cumulative)
4. **Retry failed downloads** up to 3 times with 5-second delays
5. **Continue processing** with available data if some downloads fail

## Data Sources

- **Trap Data**: Albopictus trap data in pickle format
- **Climate Data**: ERA5-Land from Copernicus Climate Data Store
  - Spatial resolution: 0.1° × 0.1°
  - Temporal resolution: Hourly → aggregated to daily
  - Spatial coverage: Europe (33°N-72°N, 25°W-45°E)

## Error Handling

The script includes comprehensive error handling:
- **Missing files**: Automatic download or graceful skipping
- **Network issues**: Retry logic with exponential backoff
- **Data corruption**: Validation and reprocessing
- **API limits**: Proper error messages and suggestions

## Performance Notes

- **Memory usage**: Large climate datasets are processed one variable at a time
- **Download time**: Each variable/year takes 5-15 minutes to download
- **Processing time**: ~1-2 minutes per variable/year for processing
- **Storage**: Raw files are ~500MB per variable/year

## Troubleshooting

### Downloads Not Working
- Check your `~/.cdsapirc` file exists and has correct credentials
- Verify internet connection and CDS service status
- Check logs for specific error messages

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### File Not Found Errors
- Use `--climate-path` to specify correct data directory
- Enable downloads with `--enable-downloads`
- Check file permissions in data directories

### Memory Issues
- Process smaller date ranges
- Ensure sufficient disk space for downloads
- Monitor system memory during processing