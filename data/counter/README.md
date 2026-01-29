# Aedes Albopictus Counter Analysis Pipeline

## Overview

This pipeline processes **Aedes albopictus** (Asian tiger mosquito) surveillance data from the **AIM-SURV** (Aedes Invasive Mosquito Survey) initiative and pairs it with high-resolution climate data from Copernicus Climate Data Store. The goal is to create a comprehensive dataset that combines mosquito trap observations with environmental variables for epidemiological analysis and modeling.

## Data Sources

### AIM-SURV Dataset

The mosquito surveillance data comes from the **AIM-SURV** initiative, a collaborative effort coordinated by the **AIM-COST Action CA17108** (Aedes Invasive Mosquitoes). This initiative aggregates standardized mosquito surveillance data from multiple countries across Europe and beyond.

**Key Features of AIM-SURV Data**:
- **Standardized Format**: Darwin Core Archive (DwC-A) format ensuring interoperability
- **Geographic Coverage**: Pan-European scope with data from multiple countries
- **Temporal Range**: Multi-year surveillance records from various monitoring programs
- **Species Focus**: Primarily invasive Aedes species (A. albopictus, A. aegypti, A. japonicus, A. koreicus)
- **Data Types**: Trap locations, deployment periods, species identifications, abundance counts, life stages
- **Quality Control**: Coordinated data validation and standardization protocols

The dataset used in this pipeline is **AIM-SURV v2.3**, which includes both positive detections and zero-count observations (absence data), crucial for accurate ecological modeling.

**More Information**:
- Zenodo Repository: [https://doi.org/10.5281/zenodo.10985325](https://doi.org/10.5281/zenodo.10985325)
- AIM-COST Action: [https://www.aedescost.eu](https://www.aedescost.eu)

## Project Goals

### Primary Objectives

1. **Data Standardization**: Clean and standardize mosquito surveillance data from the AIM-SURV initiative
2. **Temporal Analysis**: Process trap deployment periods and calculate weekly occurrence rates
3. **Climate Integration**: Extract relevant climate variables (temperature, precipitation, humidity, etc.) for each trap location and time period
4. **Quality Control**: Validate data quality through coordinate cleaning, sampling effort verification, and duplicate detection
5. **Feature Engineering**: Create both daily and monthly climate aggregations suitable for machine learning models

### Scientific Rationale

Aedes albopictus is a vector for several arboviruses (dengue, chikungunya, Zika). Understanding the relationship between climate variables and mosquito populations is crucial for:

- **Disease Risk Prediction**: Climate affects mosquito population dynamics and disease transmission
- **Early Warning Systems**: Identifying environmental conditions favorable for mosquito proliferation
- **Control Strategy Optimization**: Targeting interventions based on climate-driven risk factors
- **Invasion Risk Assessment**: Predicting potential establishment of Aedes albopictus in new areas

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COUNTER ANALYSIS PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ AIM-SURV Data    │
│    v2.3          │
│  (dwca-aimsurv)  │
└────────┬─────────┘
         │
         v
┌──────────────────────────────────────┐
│   1. ALBOPICTUS DATA PROCESSOR       │
│   (albopictus.py)                    │
│                                      │
│   • Load event and occurrence data   │
│   • Extract Aedes albopictus records │
│   • Clean coordinates                │
│   • Create unique trap IDs           │
│   • Process temporal data            │
│   • Validate sampling effort         │
│   • Calculate weekly rates           │
│   • Filter by life stages            │
└──────────────┬───────────────────────┘
               │
               v
      ┌────────────────┐
      │ albopictus.pkl │  ← Intermediate output
      └────────┬───────┘
               │
               v
┌──────────────────────────────────────────────────────────────┐
│   2. CLIMATE DATA ACQUISITION                                 │
│   (copernicus_downloader.py + copernicus_data.py)           │
│                                                               │
│   Phase 1: Download Raw Data                                 │
│   • Query Copernicus CDS API                                 │
│   • Download ERA5-Land data (monthly chunks)                 │
│   • Extract ZIP archives                                     │
│   • Validate NetCDF files                                    │
│                                                               │
│   Phase 2: Process Climate Data                              │
│   • Subset to European region                                │
│   • Resample to daily frequency                              │
│   • Calculate statistics (min/max/mean) or cumulative sums   │
│   • Compress and save processed files                        │
│                                                               │
│   Phase 3: Extract Climate for Trap Locations                │
│   • Bilinear interpolation for each trap coordinate          │
│   • Extract daily time series (89 days before trap date)     │
│   • Aggregate to monthly averages (3 months)                 │
└──────────────┬───────────────────────────────────────────────┘
               │
               v
   ┌───────────────────────────────┐
   │ albopictus_with_climate_3m.pkl│  ← Final output
   └───────────────────────────────┘
```

## Component Details

### 1. Albopictus Data Processor (`src/albopictus.py`)

**Purpose**: Clean and standardize mosquito surveillance data.

**Key Processing Steps**:

```python
# Workflow
1. Load raw surveillance data (event.txt, occurrence.txt)
2. Extract Aedes albopictus records (including zero counts)
3. Clean coordinates (remove degree symbols, convert formats)
4. Create unique trap IDs based on coordinate pairs
5. Process temporal information (start/end dates, duration)
6. Validate sampling effort (compare reported vs calculated)
7. Calculate weekly occurrence rates
8. Filter by valid life stages (Egg, Adult, Larva)
9. Remove validation failures
```

**Output**: `../output_data/albopictus.pkl`

**Data Quality Issues Handled**:
- Mixed coordinate formats (degrees vs decimals)
- Comma vs period decimal separators
- Missing coordinates
- Zero time differences in trap deployment
- Invalid sampling effort reports
- Duplicate trap measurements

### 2. Climate Data Downloader (`src/copernicus_downloader.py`)

**Purpose**: Download and process ERA5-Land climate reanalysis data.

**Climate Variables**:
- `total_precipitation`: Daily precipitation sum (mm)
- `2m_temperature`: Air temperature at 2m height (K)
- `2m_dewpoint_temperature`: Dewpoint temperature (K)
- `10m_u_component_of_wind`: Eastward wind component (m/s)
- `10m_v_component_of_wind`: Northward wind component (m/s)
- `volumetric_soil_water_layer_1`: Soil moisture (m³/m³)
- Additional variables available (see source code)

**Processing Strategy**:
- Downloads data in monthly chunks to avoid CDS size limits
- Merges chunks into yearly files
- Subsets to European region (33°N-72°N, 25°W-45°E)
- Resamples to daily frequency
- Calculates statistics (min/max/mean) or cumulative values

**Two-Phase Approach**:
1. **Download Phase**: Download ALL required raw data first
2. **Processing Phase**: Process ALL downloaded data

This prevents partial failures and allows resuming from intermediate states.

### 3. Climate Data Processor (`src/copernicus_data.py`)

**Purpose**: Pair trap observations with climate data.

**Processing Pipeline**:

```python
# For each trap observation:
1. Filter trap data by date range (default: 2020-01-01 to 2021-01-01)
2. Determine required climate data years
3. For each climate variable:
   a. Load NetCDF files for required years
   b. Extract climate at trap location using bilinear interpolation
   c. Create daily time series (89 days before trap date)
   d. Aggregate to monthly averages (3 months)
4. Save combined dataset
```

**Interpolation Method**: Bilinear interpolation with NaN handling
- Uses 4 nearest grid points
- Handles missing values gracefully
- Weights by distance from target coordinate

**Output**: `../output_data/albopictus_with_climate_3m.pkl`

### 4. Helper Functions (`src/process_copernicus_data.py`)

**Key Functions**:
- `extract_climate_data_to_df()`: Extract climate arrays for trap locations
- `bilinear_interpolation_nan()`: Interpolate climate values with NaN handling
- `extract_climate_data()`: Get time series for specific location/date
- `aggregate_to_monthly()`: Aggregate daily values to monthly means

## Setup and Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install pandas numpy xarray netCDF4 matplotlib tqdm cdsapi
```

### Copernicus CDS API Setup (Optional, for automatic downloads)

1. **Register** at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)

2. **Get API credentials** from [API how-to page](https://cds.climate.copernicus.eu/api-how-to)

3. **Create configuration file** `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY
   ```

4. **Accept Terms & Conditions** for ERA5-Land dataset on CDS website

### Directory Structure

```
AIedes_data/data/counter/
├── README.md                          # This file
├── make_counter_dataset.sh            # Main execution script
├── src/
│   ├── albopictus.py                  # Mosquito data processor
│   ├── copernicus_downloader.py       # Climate data downloader
│   ├── copernicus_data.py             # Climate pairing script
│   └── process_copernicus_data.py     # Helper functions
├── input_data/
│   ├── dwca-aimsurv-v2.3/            # Raw AIM-SURV surveillance data
│   │   ├── event.txt                  # Trap deployment events
│   │   └── occurrence.txt             # Species occurrence records
│   └── climate/                       # Climate data (auto-created)
│       ├── raw/                       # Raw downloaded NetCDF
│       └── processed/                 # Processed climate files
│           └── europe/daily/YYYY/
└── output_data/                       # Generated outputs
    ├── albopictus.pkl                # Processed trap data
    └── albopictus_with_climate_3m.pkl # Final dataset
```

## Usage

### Quick Start

```bash
# Complete pipeline with automatic downloads
cd /home/biazzin/git/AIedes_data/data/counter
bash make_counter_dataset.sh
```

### Step-by-Step Execution

#### Step 1: Process Mosquito Data

```bash
cd src
python albopictus.py
```

**Output**: `../output_data/albopictus.pkl`

#### Step 2: Download and Process Climate Data

```bash
# With automatic downloads enabled
python copernicus_data.py --enable-downloads --force-redownload

# Or manually specify input file
python copernicus_data.py --input-file ../output_data/albopictus.pkl \
    --enable-downloads
```

**Output**: `../output_data/albopictus_with_climate_3m.pkl`

### Advanced Usage

#### Custom Date Range

```bash
python copernicus_data.py \
    --start-date 2019-01-01 \
    --end-date 2022-12-31 \
    --enable-downloads
```

#### Custom Climate Data Path

```bash
python copernicus_data.py \
    --climate-path /path/to/climate/data \
    --enable-downloads
```

#### Download Specific Variables/Years

```python
from src.copernicus_downloader import download_missing_data

variables = ["total_precipitation", "2m_temperature"]
years = [2019, 2020, 2021]

results = download_missing_data(
    variables=variables,
    years=years,
    base_output_dir="../input_data/climate",
    freq="daily"
)
```

## Output Data Format

### Final Dataset Structure

```python
import pandas as pd

df = pd.read_pickle("output_data/albopictus_with_climate_3m.pkl")

# Trap identification
df['id_trap']           # Unique trap ID
df['decimalLatitude']   # Latitude
df['decimalLongitude']  # Longitude

# Temporal information
df['start_date']        # Trap deployment start
df['end_date']          # Trap collection date
df['time_diff']         # Deployment duration (days)

# Mosquito observations
df['individualCount']   # Number of specimens
df['lifeStage']         # Egg/Adult/Larva
df['weeklyRate']        # Weekly occurrence rate

# Climate data (for each variable)
df['variable_name']              # Daily time series (89 days)
df['variable_name_monthly']      # Monthly averages (3 months)
df['climate_nan']                # Data quality flag
```

### Climate Variable Naming Convention

- **Daily data**: `{variable_name}` (e.g., `total_precipitation_sum`)
- **Monthly data**: `{variable_name}_monthly` (e.g., `total_precipitation_sum_monthly`)
- **Statistics variables**: Have `_min`, `_max`, `_mean` suffixes
- **Cumulative variables**: Have `_sum` suffix

## Troubleshooting

### Common Issues

#### 1. CDS API Errors

```
Problem: "Client not authorized" or "Invalid API key"
Solution: 
- Check ~/.cdsapirc configuration
- Verify API key on CDS website
- Accept Terms & Conditions for ERA5-Land dataset
```

#### 2. File Format Errors

```
Problem: "NetCDF file validation failed"
Solution:
- CDS sometimes returns ZIP files instead of NetCDF
- The downloader automatically extracts them
- If issues persist, delete and redownload:
  rm -rf input_data/climate/raw/YYYY/
```

#### 3. Memory Issues

```
Problem: "MemoryError" during climate processing
Solution:
- Process fewer years at once
- Increase system swap space
- Use a machine with more RAM (16GB+ recommended)
```

#### 4. Missing Climate Data

```
Problem: "File does not exist" for climate data
Solution:
- Run with --enable-downloads flag
- Or manually download from Copernicus CDS:
  python src/copernicus_downloader.py
```

### Validation

#### Check Trap Data

```python
import pandas as pd

df = pd.read_pickle("output_data/albopictus.pkl")

print(f"Total records: {len(df)}")
print(f"Unique traps: {df['id_trap'].nunique()}")
print(f"Date range: {df['end_date'].min()} to {df['end_date'].max()}")
print(f"Life stages: {df['lifeStage'].value_counts()}")
```

#### Check Climate Data

```python
df = pd.read_pickle("output_data/albopictus_with_climate_3m.pkl")

# Check for missing climate data
nan_count = (df['climate_nan'] == 'yes').sum()
print(f"Records with NaN climate data: {nan_count}/{len(df)}")

# Check climate array shapes
sample_climate = df['total_precipitation_sum'].iloc[0]
print(f"Daily climate array shape: {sample_climate.shape}")  # Should be (89,)

sample_monthly = df['total_precipitation_sum_monthly'].iloc[0]
print(f"Monthly climate array shape: {sample_monthly.shape}")  # Should be (3,)
```

## Performance Tips

1. **Download Climate Data Once**: Downloaded files are cached in `input_data/climate/raw/`
2. **Use Processed Files**: Processed files in `input_data/climate/processed/` are reused automatically
3. **Parallel Downloads**: The downloader processes variables sequentially; consider running multiple instances for different variables
4. **Disk Space**: Each year of climate data requires ~500MB-2GB depending on variables

## Citation

If using this pipeline or the resulting datasets, please cite:

### AIM-SURV Data

**Primary Citation (Dataset)**:
```
Metz, M., Palmer, J. R. B., Florian, M., Kríž, B., Dekoninck, W., Selmi, S., 
Merdić, E., Veronesi, E., Lilja, T., Montagner, P., Šebesta, O., Balenghien, T., 
Werner, D., Schaffner, F., & Eritja, R. (2024). 
AIM-SURV Dataset (v2.3) [Data set]. Zenodo. 
https://doi.org/10.5281/zenodo.10985325
```

**Scientific Paper**:
```
Metz, M., Palmer, J. R. B., Florian, M., Kríž, B., Dekoninck, W., Selmi, S., 
Merdić, E., Veronesi, E., Lilja, T., Montagner, P., Šebesta, O., Balenghien, T., 
Werner, D., Schaffner, F., & Eritja, R. (2024). 
Linking a citizen science programme and a federal surveillance programme reveals 
large-scale effects of landscape and climate on invasive Aedes mosquitoes. 
Scientific Reports, 14, Article 10456. 
https://doi.org/10.1038/s41598-024-60967-1
```

**AIM-COST Action**:
```
AIM-COST Action CA17108 (2017-2021). Aedes Invasive Mosquitoes. 
Website: https://www.aedescost.eu
```

### Climate Data

**ERA5-Land Dataset**:
```
Muñoz Sabater, J. (2019). ERA5-Land hourly data from 1950 to present. 
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 
DOI: 10.24381/cds.e2161bac
```

**Access**:
```
Accessed via Copernicus Climate Data Store: https://cds.climate.copernicus.eu
```

### This Pipeline

```
[Your Name/Institution] (2024). Aedes Albopictus Counter Analysis Pipeline. 
GitHub: [repository URL if applicable]
```

## Acknowledgments

This work builds upon data and efforts from:

- **AIM-COST Action CA17108** for coordinating and standardizing the AIM-SURV dataset
- **European Cooperation in Science and Technology (COST)** for funding the AIM network
- All **national surveillance programs and researchers** who contributed data to AIM-SURV
- **Copernicus Climate Change Service (C3S)** and **ECMWF** for ERA5-Land climate data
- **Mosquito Alert** and other citizen science initiatives contributing to surveillance efforts

## Data Use and Restrictions

### AIM-SURV Data

The AIM-SURV dataset is made available for research and public health purposes under the Creative Commons Attribution 4.0 International License (CC BY 4.0). When using this data:

1. **Cite appropriately**: Always cite both the dataset (Zenodo) and the scientific paper
2. **Respect contributors**: Individual country data may have specific acknowledgment requirements
3. **Data sharing**: Follow FAIR principles (Findable, Accessible, Interoperable, Reusable)
4. **Contact for access**: Data available on Zenodo at [https://doi.org/10.5281/zenodo.10985325](https://doi.org/10.5281/zenodo.10985325)

### Climate Data

ERA5-Land data from Copernicus CDS:
- **License**: Available under the Copernicus License
- **Attribution**: Always cite the Copernicus C3S and the DOI
- **Registration**: Free registration required for API access

## Contributing

For issues, improvements, or questions about this pipeline:
- Open an issue on the project repository
- Contact: biazzin@example.com

For questions about the AIM-SURV dataset:
- Zenodo: [https://doi.org/10.5281/zenodo.10985325](https://doi.org/10.5281/zenodo.10985325)
- Paper: [https://doi.org/10.1038/s41598-024-60967-1](https://doi.org/10.1038/s41598-024-60967-1)
- AIM-COST Action: [https://www.aedescost.eu](https://www.aedescost.eu)

## License

**Pipeline Code**: [Add your license here, e.g., MIT, GPL-3.0]

**Data**:
- AIM-SURV data: Creative Commons Attribution 4.0 International License (CC BY 4.0)
- ERA5-Land data: Copernicus License
