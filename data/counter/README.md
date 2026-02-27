# Dataset 2: AIMSurv Trap-Level Climate Dataset

This pipeline constructs the **AIMSurv trap-level climate dataset** described in the accompanying paper (*Harmonized European surveillance–climate datasets for Aedes albopictus*). It transforms raw mosquito surveillance records from the AIMSurv Darwin Core Archive (v2.3) into an analysis-ready dataset where each trap observation is paired with time-resolved ERA5-Land climate variables.

## Motivation

Systematic comparison of mosquito distribution and abundance models is hindered by the lack of harmonised datasets that consistently link surveillance observations with climate covariates in a standardised and reusable format. This pipeline addresses that gap by producing a publicly available reference dataset that supports benchmarking, methodological comparison, and reuse across ecological and epidemiological studies focused on climate-informed modelling of *Aedes albopictus*.

## Created Dataset

Each row in the output represents a single sampling event at a mosquito trap — either a positive *Aedes albopictus* detection or a confirmed zero-count (negative observation). Trap records are paired with ERA5-Land daily and monthly climate variables aligned to the sampling period, suitable for time-resolved abundance modelling.

| File | Format | Description |
|------|--------|-------------|
| `output_data/AIMSurv_albopictus_2020_era5_land.csv.zip` | Compressed CSV | **Final dataset** — trap observations + climate variables |
| `output_data/AIMSurv_albopictus_2020_era5_land.pkl` | Python pickle | Same data, preserving numpy arrays for direct use in Python |
| `output_data/albopictus.csv.zip` | Compressed CSV | Intermediate — cleaned trap data before climate linkage |
| `output_data/albopictus.pkl` | Python pickle | Same intermediate data |
| `output_stats/albopictus_summary.json` | JSON | Pipeline diagnostics and QA counts |
| `output_stats/plots/*.png` | PNG | 10 exploratory charts |

**What each record contains:**

- **Trap identity & location**: unique trap ID, latitude, longitude
- **Observation**: species, life stage (Egg / Adult / Larva), individual count, weekly occurrence rate
- **Trap deployment**: start date, end date, duration in days
- **Climate history** (per trap, per collection date):
  - **Daily arrays** (89 days before collection) for precipitation, temperature, dewpoint, wind, and soil moisture — each statistic (min/max/mean or sum) stored as a separate column
  - **Monthly summaries** (3 × ~30-day means) of the same variables, capturing month-to-month climatic trends preceding the observation
  - A `climate_nan` flag indicating whether any extracted value contained missing data

The dataset is designed for abundance modelling, temporal analysis, and epidemiological studies requiring trap-level observations paired with time-resolved climate information.

## Data Sources

| Source | Description | Link |
|--------|-------------|------|
| **AIMSurv v2.3** | Standardised mosquito surveillance (Darwin Core Archive); 19,743 occurrence records across 18 taxa | [GBIF IPT](https://ipt.gbif.es/resource?r=aimsurv) · [Zenodo](https://doi.org/10.5281/zenodo.10985325) |
| **ERA5-Land** | Hourly climate reanalysis at 0.1° resolution | [CDS](https://doi.org/10.24381/cds.e2161bac) |

## Directory Structure

```
counter/
├── make_counter_dataset.sh              # Run the full pipeline
├── src/
│   ├── albopictus.py                    # Mosquito data processor
│   ├── copernicus_downloader.py         # ERA5-Land download & processing
│   ├── copernicus_data.py               # Pair traps with climate data
│   ├── process_copernicus_data.py       # Climate extraction helpers
│   └── plot_stats.py                    # Generate analysis plots
├── input_data/
│   ├── dwca-aimsurv-v2.3/              # Raw AIM-SURV data (event.txt, occurrence.txt)
│   └── climate/                         # Climate data (auto-created)
│       ├── raw/{year}/                  # Raw hourly NetCDF downloads
│       └── processed/europe/daily/{year}/  # Daily aggregated NetCDF
├── output_data/                         # ← created datasets live here
│   ├── albopictus.csv.zip
│   ├── albopictus.pkl
│   ├── AIMSurv_albopictus_2020_era5_land.csv.zip
│   └── AIMSurv_albopictus_2020_era5_land.pkl
└── output_stats/
    ├── albopictus_summary.json
    └── plots/
```

## Pipeline

The dataset is constructed in three stages: surveillance data cleaning, climate data acquisition, and climate linkage.

```
AIM-SURV v2.3 ──► albopictus.py ──► albopictus.csv.zip / .pkl
                                          │
ERA5-Land (CDS) ──► copernicus_downloader.py ──┐
                                               ▼
                                     copernicus_data.py
                                               │
                                               ▼
                              AIMSurv_albopictus_2020_era5_land.csv.zip / .pkl
```

### Step 1 — Surveillance data processing (`src/albopictus.py`)

Transforms raw AIMSurv Darwin Core Archive files into a clean *Aedes albopictus* dataset through six stages (see paper, Figure 2):

1. **Extraction**: select *Aedes albopictus* records (3,775) plus all zero-count records (11,955 negative observations, predominantly family-level *Culicidae*), yielding 15,730 records
2. **Coordinate cleaning**: remove `°` symbols, replace `,` → `.`, convert to numeric, drop NaN (no records lost)
3. **Trap identification**: assign unique IDs via `ngroup()` on `(decimalLatitude, decimalLongitude)` → 1,648 traps
4. **Temporal processing**: parse `eventDate` ranges → `start_date` / `end_date`; compute `time_diff` (days); fix 7 zero-duration records to 1 day
5. **Effort validation**: cross-check reported `samplingEffort` against computed duration; discard 356 records (2.3%) with discrepancies > 2 days
6. **Rate normalisation & filtering**: compute `weeklyRate = 7 × individualCount / time_diff`; restrict to life stages {Egg, Adult, Larva} and validated records → **15,374 final records** (76.5% zero-count)

**Outputs:** `output_data/albopictus.csv.zip`, `output_data/albopictus.pkl`, `output_stats/albopictus_summary.json`

### Step 2 — Climate data acquisition & linkage (`src/copernicus_downloader.py` + `src/copernicus_data.py`)

ERA5-Land hourly fields are downloaded from the Copernicus CDS at 0.1° resolution and aggregated to daily frequency:

| Variable | Aggregation |
|----------|-------------|
| `total_precipitation` | daily sum |
| `2m_temperature`, `2m_dewpoint_temperature` | daily min / max / mean |
| `10m_u_component_of_wind`, `10m_v_component_of_wind` | daily min / max / mean |
| `volumetric_soil_water_layer_1` | daily min / max / mean |

Downloads are chunked by month, merged per year, subset to a European domain (25–75°N, 25°W–45°E), and stored as compressed NetCDF-4.

**Climate extraction** for each trap observation uses bilinear interpolation over the four surrounding grid cells (weights normalised over non-NaN neighbours for coastal/border traps):

- **Daily window**: 89-day time series ending on `end_date` → vector of length 89 per variable
- **Three-month summary**: the same window reshaped into 3 × ~30-day blocks, mean per block → vector of length 3

This produces 16 climate feature columns per observation (daily vectors) plus their monthly aggregates.

**Outputs:** `output_data/AIMSurv_albopictus_2020_era5_land.csv.zip`, `output_data/AIMSurv_albopictus_2020_era5_land.pkl`

### Step 3 — Exploratory plots (`src/plot_stats.py`)

Generates 10 charts into `output_stats/plots/` (species breakdown, pipeline funnel, life stages, geographic scatter, time-diff distribution, etc.). Several of these figures appear in the accompanying paper.

## Quick Start

```bash
cd /home/biazzin/git/AIedes_data/data/counter

# Full pipeline (downloads climate data automatically)
bash make_counter_dataset.sh

# Or step-by-step:
python src/albopictus.py
python src/copernicus_data.py --enable-downloads
python src/plot_stats.py
```

### Prerequisites

```bash
pip install pandas numpy xarray netCDF4 matplotlib tqdm cdsapi seaborn
# Optional (for basemap in geographic plot):
pip install contextily geopandas pyproj
```

For automatic climate downloads, configure `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

### Advanced Options

```bash
# Custom date range
python src/copernicus_data.py --start-date 2019-01-01 --end-date 2022-12-31 --enable-downloads

# Force re-download of climate data
REBUILD_CLIMATE=1 bash make_counter_dataset.sh

# Custom input file
python src/copernicus_data.py --input-file path/to/data.pkl --enable-downloads
```

## Output Dataset Columns

| Column | Description |
|--------|-------------|
| `id_trap` | Unique trap ID (coordinate-based) |
| `decimalLatitude`, `decimalLongitude` | Trap coordinates |
| `start_date`, `end_date` | Trap deployment period |
| `time_diff` | Deployment duration (days) |
| `individualCount` | Specimen count |
| `lifeStage` | Egg / Adult / Larva |
| `weeklyRate` | `7 × individualCount / time_diff` |
| `{variable}` | Daily climate array (89 values) |
| `{variable}_monthly` | Monthly climate array (3 values) |
| `climate_nan` | `"yes"` if any climate extraction contained NaN |

## Run Summary (`output_stats/albopictus_summary.json`)

Records pipeline diagnostics: raw record counts, species breakdown, coordinate cleaning stats, trap counts, temporal fixes, sampling effort validation, and final filtering results. Use this for QA or to populate manuscript tables without re-running the pipeline.

## Citation

### AIMSurv Data

> Miranda Chueca, M. Á. & Barceló Seguí, C. (2022). AIMSurv Aedes Invasive Mosquito species harmonized surveillance in Europe. AIM-COST Action. GBIF. https://doi.org/10.15470/vs3677

### ERA5-Land

> Muñoz Sabater, J. et al. (2021). ERA5-Land: A state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383. https://doi.org/10.5194/essd-13-4349-2021

### This Dataset

> [Authors] (2025). Harmonized European surveillance–climate datasets for *Aedes albopictus*. [Journal/Repository]. [DOI]

## License

- **Pipeline code**: [Add your license]
- **AIMSurv data**: CC0 1.0 Public Domain Dedication
- **ERA5-Land data**: Copernicus License
