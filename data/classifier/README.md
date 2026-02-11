# Dataset 1: ECDC Polygon-to-Grid Climate Dataset

This pipeline constructs the **ECDC polygon-to-grid climate dataset** described in the accompanying paper (*Harmonized European surveillance–climate datasets for Aedes albopictus*). It spatially joins polygon-level mosquito surveillance records from the European Centre for Disease Prevention and Control (ECDC), reported at the NUTS-3 regional level, to individual climate grid points derived from CORDEX regional climate simulations or ERA5-Land reanalysis, producing a grid-resolved dataset suitable for species distribution modelling and climate impact assessments.

## Motivation

Polygon-level surveillance data — where an entire administrative region is labelled as "present" or "absent" — cannot be used directly in grid-based modelling frameworks without spatial downscaling. At the same time, climate conditions can vary substantially within a single NUTS-3 polygon (e.g., coastal lowlands vs. mountain tops), so naively assigning a single climate value per polygon discards information relevant to ecological modelling.

This pipeline addresses both issues by inverting the resolution hierarchy: instead of aggregating climate to polygon level, each climate grid point inherits the surveillance label of the polygon it falls within. The result is a dataset where every row is a georeferenced point with its own climate profile and the ECDC surveillance status of its surrounding region, ready for grid-based classifiers and species distribution models.

## Created Dataset

Each row in the output represents **one climate grid point** located within an ECDC surveillance polygon. Grid points inherit all ECDC metadata (region identifiers, species name, presence/absence status) from the containing polygon. Climate values are computed as **decadal averages**: the `year` field denotes the final year of the averaging window (e.g., `year = 2020` corresponds to the period 2011–2020).

| File | Format | Description |
|------|--------|-------------|
| `data/outputs/ecdc_albopictus_cordex_{year}.zip` | Compressed CSV | Final dataset using CORDEX climate (~12 km) |
| `data/outputs/ecdc_albopictus_era5_land_{year}.zip` | Compressed CSV | Final dataset using ERA5-Land climate (~9 km) |

**What each record contains:**

- **Grid point location**: latitude, longitude (WGS84), unique `location_id`
- **Surveillance label**: ECDC presence/absence status (`presence_numeric` ∈ {0, 1, 2, 3}), human-readable `status`, NUTS `LocationCode`
- **Monthly climatology**: 12-element arrays of mean temperature (°C) and total precipitation (mm), plus expanded month-specific columns
- **Quality-control flags**: binary indicators for temperature suitability, precipitation suitability, and combined suitability — conservative filters to mitigate labelling artefacts from climatic heterogeneity within surveillance polygons (see [Suitability Filters](#suitability-filters))

The dataset is designed for binary classification of *Aedes albopictus* presence/absence at the grid-point scale, benchmarking of species distribution models, and assessment of climate suitability under current or projected conditions.

## Data Sources

| Source | Description | Link |
|--------|-------------|------|
| **ECDC** | Polygon-level mosquito surveillance (NUTS-3), species presence/absence | [ECDC mosquito maps](https://www.ecdc.europa.eu/en/disease-vectors/surveillance-and-disease-data/mosquito-maps) |
| **CORDEX** | Regional climate projections at 0.11° (~12 km); MPI-ESM-LR → SMHI-RCA4, RCP 4.5 | [cordex.org](https://cordex.org/) |
| **ERA5-Land** | Climate reanalysis at 0.1° (~9 km) | [CDS](https://doi.org/10.24381/cds.e2161bac) |

## Directory Structure

```
data/classifier/
├── pair_ecdc_copernicus_data.py       # Main pipeline script
├── make_classifier_database.sh        # Shell wrapper
├── src/
│   ├── aedes_suitability.py           # Temperature & precipitation QC filters
│   ├── copernicus.py                  # CORDEX download & processing
│   ├── era5_land_downloader.py        # ERA5-Land download & processing
│   ├── unified_climate_downloader.py  # Unified interface for both sources
│   └── check_similarity_df.py        # Utility: compare two output CSVs
├── data/
│   ├── inputs/                        # ECDC GDB + downloaded climate NetCDFs
│   ├── outputs/                       # ← created datasets live here
│   └── img/                           # Generated plots
└── README.md                          # This file
```

## Pipeline

```
ECDC GDB polygons ──┐
                     ├─ spatial join (point-in-polygon) ─► paired CSV (zipped)
Climate NetCDF grid ─┘                                     └─ data/outputs/
```

**Steps** (implemented in `pair_ecdc_copernicus_data.py`):

1. Load ECDC geodatabase → filter for *Aedes albopictus*
2. Download / load 10-year monthly climatology (CORDEX or ERA5-Land)
3. Build a DataFrame with one row per grid point (lat, lon, 12 monthly temps, 12 monthly precips)
4. Spatial join: each grid point inherits the ECDC polygon attributes it falls within
5. Compute climate-suitability QC flags (see [Suitability Filters](#suitability-filters))
6. Filter to continental Europe (lat 34–75°N)
7. Save to `data/outputs/`

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--year` | `2020` | End year of 10-year climatology |
| `--climate-source` | `cordex` | `cordex` or `era5_land` |
| `--parent-dir` | `./data/inputs/` | Directory containing ECDC GDB |
| `--ecdc-file` | `20230828_VectorFlatFileGDB.gdb.zip` | ECDC geodatabase filename |
| `--output-dir` | `./data/outputs/` | Output directory |

## Quick Start

```bash
cd /home/biazzin/git/AIedes_data/data/classifier

# Generate dataset with CORDEX projections (2011–2020 climatology)
python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex

# Generate dataset with ERA5-Land reanalysis
python pair_ecdc_copernicus_data.py --year 2020 --climate-source era5_land
```

Load in Python:

```python
import pandas as pd
df = pd.read_csv('data/outputs/ecdc_albopictus_cordex_2020.zip', compression='zip')
```

## Output Variables

### Target & Status

| Variable | Type | Description |
|----------|------|-------------|
| `presence_numeric` | int | **0** = absent, **1** = present/established, 2 = introduced, 3 = no data/unknown |
| `status` | str | Human-readable label (`Absent`, `Established`, `Present`, `Introduced`, `No data`, `Unknown`) |
| `LocationCode` | str | NUTS administrative code of the ECDC polygon |

> **For ML training use only rows where `presence_numeric ∈ {0, 1}`** and optionally `Suitable == 1`.

### Climate Features

| Variable | Unit | Description |
|----------|------|-------------|
| `temperature_2m_monthly` | °C | List of 12 monthly mean temperatures [Jan … Dec] |
| `precipitation_monthly` | mm | List of 12 monthly precipitation totals [Jan … Dec] |
| `temp_Jan_C` … `temp_Dec_C` | °C | Expanded individual-month temperature columns |
| `precip_Jan_mm` … `precip_Dec_mm` | mm | Expanded individual-month precipitation columns |

### Geography & Metadata

| Variable | Description |
|----------|-------------|
| `latitude`, `longitude` | Grid point coordinates (WGS84) |
| `location_id` | Unique key: `lat_XX.XXX_lon_YY.YYY` |
| `year` / `analysis_year` | End year of the 10-year climatology period |
| `climate_data_source` | `cordex` or `era5_land` |

### QC Flags

| Variable | Values | Purpose |
|----------|--------|---------|
| `Temperature Suitable` | 0 / 1 | Passes temperature thresholds |
| `Precipitation Suitable` | 0 / 1 | Annual precip > 200 mm |
| `Suitable` | 0 / 1 | Both flags combined |

## Suitability Filters

### Why?

ECDC labels are at polygon scale (~10–50 km). When downscaling to climate grid points (~9–12 km), some points within a "present" polygon may have extreme climates (e.g., mountain tops) that are biologically implausible for *Aedes albopictus*. The QC flags help remove these artefacts from training data. As noted in the paper, these filters are intended as **conservative, transparent controls** to mitigate false positive presence labels arising from spatial heterogeneity within surveillance polygons, rather than as definitive ecological constraints.

### Temperature Filter

Implemented in `src/aedes_suitability.py :: aedes_temperature_suitability()`. A location passes if **all** conditions hold:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Coldest month mean | ≥ −3 °C | Egg overwintering survival |
| Annual mean | ≥ 10 °C | Sustained population development |

### Precipitation Filter

Implemented in `src/aedes_suitability.py :: aedes_precipitation_suitability()`.

A location passes if `annual_precip > 200 mm`.

### Recommended Usage

```python
# Clean training set: definitive labels + climatically suitable points
df = pd.read_csv('data/outputs/ecdc_albopictus_cordex_2020.zip', compression='zip')
df_train = df[(df['presence_numeric'].isin([0, 1])) & (df['Suitable'] == 1)].copy()
```

### Limitations

- Thresholds are intentionally **conservative** (permissive) to avoid false negatives.
- No upper-temperature or flood filters.
- Microhabitat effects (urban heat islands, irrigation) are not captured.
- Treat these flags as QC indicators, not as an ecological niche model.

## ECDC Status Mapping

| ECDC Code | `status` | `presence_numeric` |
|-----------|----------|-------------------|
| INV001A | Established | 1 |
| INV002A | Introduced | 2 |
| INV003A | Absent | 0 |
| INV004A / INV999A | No data / Unknown | 3 |
| NAT001A | Present | 1 |
| NAT002A / NAT003A | Absent | 0 |
| NAT004A / NAT999A | No data / Unknown | 3 |
| NAT005A | Introduced | 2 |

## Climate Data Sources

### CORDEX

| Property | Value |
|----------|-------|
| Resolution | 0.11° (~12 km) |
| Model chain | MPI-ESM-LR → SMHI-RCA4, RCP 4.5, r1i1p1 |
| Type | Future projections (decadal climatology) |
| Conversion | tas [K] → °C; pr [kg m⁻² s⁻¹] → mm/month |

### ERA5-Land

| Property | Value |
|----------|-------|
| Resolution | 0.1° (~9 km) |
| Extent | Europe [33–72°N, 25°W–45°E] |
| Type | Historical reanalysis (decadal climatology) |
| Conversion | t2m [K] → °C; tp [m/day rate] → mm/month |

## Important Notes

- **Multiple rows per NUTS region** — by design (one row per climate grid point within each polygon).
- **`year` = end of 10-year window** — e.g., `year=2020` means 2011–2020 average.
- **NaN in ECDC columns** — grid points outside any ECDC polygon; drop with `df.dropna(subset=['LocationCode'])`.

## Citation

### ECDC Data

> European Centre for Disease Prevention and Control (2024). Mosquito maps. https://www.ecdc.europa.eu/en/disease-vectors/surveillance-and-disease-data/mosquito-maps

### CORDEX

> Giorgi, F., Jones, C., & Asrar, G. R. (2009). Addressing climate information needs at the regional level: the CORDEX framework. *WMO Bulletin*, 58(3), 175–183.

### ERA5-Land

> Muñoz Sabater, J. et al. (2021). ERA5-Land: A state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383. https://doi.org/10.5194/essd-13-4349-2021

### This Dataset

> [Authors] (2025). Harmonized European surveillance–climate datasets for *Aedes albopictus*. [Journal/Repository]. [DOI]

## License

- **Pipeline code**: [Add your license]
- **ECDC data**: ECDC data use terms ([legal notice](https://www.ecdc.europa.eu/en/legal-notice))
- **CORDEX data**: Subject to data use terms of respective modelling centres
- **ERA5-Land data**: Copernicus License