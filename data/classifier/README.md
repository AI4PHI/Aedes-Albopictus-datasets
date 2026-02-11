# ECDC–Climate Data Pairing for *Aedes albopictus* Analysis

## Overview

This module pairs **mosquito presence/absence observations** from [ECDC](https://www.ecdc.europa.eu/en/disease-vectors/surveillance-and-disease-data/mosquito-maps) with **gridded climate data** (CORDEX or ERA5-Land) to produce a single tabular dataset suitable for machine learning and species distribution modelling.

Each row in the output represents **one climate grid point** annotated with the ECDC surveillance status of the containing NUTS-3 polygon.

**Output location:** `data/outputs/`  
**Naming convention:** `ecdc_albopictus_{climate_source}_{year}.zip`  
Examples: `ecdc_albopictus_cordex_2020.zip`, `ecdc_albopictus_era5_land_2020.zip`

---

## Quick Start

```bash
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

---

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
5. Compute simple climate-suitability QC flags (see [Suitability Filters](#suitability-filters))
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

---

## Output Variables

### Target & Status

| Variable | Type | Description |
|----------|------|-------------|
| `presence_numeric` | int | **0** = absent, **1** = present/established, 2 = introduced, 3 = no data/unknown |
| `status` | str | Human-readable label (`Absent`, `Established`, `Present`, `Introduced`, `No data`, `Unknown`) |
| `LocationCode` | str | NUTS administrative code of the ECDC polygon |

> **For ML training use only rows where `presence_numeric ∈ {0, 1}`.**

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
| `Temperature Suitable` | 0 / 1 | Passes ECDC temperature thresholds |
| `Precipitation Suitable` | 0 / 1 | Annual precip > 200 mm |
| `Suitable` | 0 / 1 | Both flags combined |

These are **data-quality filters**, not ecological suitability scores (see below).

---

## Suitability Filters

### Why?

ECDC labels are at polygon scale (~10–50 km). When downscaling to climate grid points (~9–12 km), some points within a "present" polygon may have extreme climates (e.g., mountain tops) that are biologically implausible. The QC flags help remove these artefacts from training data.

### Temperature Filter

Implemented in `src/aedes_suitability.py :: aedes_temperature_suitability()`. A location passes if **all** conditions hold:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Coldest month mean | ≥ −3 °C | Egg overwintering survival (ECDC) |
| Annual mean | ≥ 10 °C | Sustained population development (ECDC) |
| Months ≥ 10 °C | ≥ 0 (permissive) | Kept for future extensibility |

Source: [ECDC *Aedes albopictus* factsheet – Establishment thresholds](https://www.ecdc.europa.eu/en/disease-vectors/facts/mosquito-factsheets/aedes-albopictus)

### Precipitation Filter

Implemented in `src/aedes_suitability.py :: aedes_precipitation_suitability()`.

A location passes if `annual_precip > 200 mm`.

### Recommended Usage

```python
# Clean training set
df = pd.read_csv('data/outputs/ecdc_albopictus_cordex_2020.zip', compression='zip')
df_train = df[(df['presence_numeric'].isin([0, 1])) & (df['Suitable'] == 1)].copy()
```

### Limitations

- Thresholds are intentionally **conservative** (permissive) to avoid false negatives.
- No upper-temperature or flood filters.
- Microhabitat effects (urban heat islands, irrigation) are not captured.
- Treat these flags as QC indicators, not as an ecological niche model.

---

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

---

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

---

## Important Notes

- **Multiple rows per NUTS region** — by design (one row per climate grid point).
- **`year` = end of 10-year window** — e.g., `year=2020` means 2011–2020 average.
- **NaN in ECDC columns** — grid points outside any ECDC polygon; drop with `df.dropna(subset=['LocationCode'])`.

---

## Repository Structure

```
data/classifier/
├── pair_ecdc_copernicus_data.py   # Main pipeline script
├── make_classifier_database.sh    # Shell wrapper
├── src/
│   ├── aedes_suitability.py       # Temperature & precipitation QC filters
│   ├── copernicus.py              # CORDEX download & processing
│   ├── era5_land_downloader.py    # ERA5-Land download & processing
│   ├── unified_climate_downloader.py  # Unified interface for both sources
│   └── check_similarity_df.py    # Utility: compare two output CSVs
├── data/
│   ├── inputs/                    # ECDC GDB + downloaded climate NetCDFs
│   ├── outputs/                   # ⬅ Generated datasets (ZIP/CSV)
│   └── img/                       # Generated plots
└── README.md                      # This file
```

---

## References

- **ECDC** — *Aedes albopictus* factsheet & vector maps: [ecdc.europa.eu](https://www.ecdc.europa.eu/en/disease-vectors/facts/mosquito-factsheets/aedes-albopictus)
- **CORDEX**: [cordex.org](https://cordex.org/)
- **ERA5-Land**: [CDS catalogue](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means)
- Cunze et al. (2016). *Aedes albopictus and Its Environmental Limits in Europe.* PLoS ONE. [doi](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162116)
- Caminade et al. (2012). *Suitability of European climate for the Asian tiger mosquito.* J. R. Soc. Interface 9(75). [doi](https://royalsocietypublishing.org/rsif/article/9/75/2708)
- Kraemer et al. (2019). *Past and future spread of Aedes aegypti and Aedes albopictus.* Nature Microbiology. [doi](https://www.nature.com/articles/s41564-019-0376-y)

---

**Last Updated**: 2025 · **Pipeline**: Python 3.8+