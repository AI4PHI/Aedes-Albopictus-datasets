# ECDC-Climate Data Pairing for Aedes Mosquito Analysis

## Overview

This database pairs **mosquito presence/absence observations** from ECDC (European Centre for Disease Prevention and Control) with **high-resolution climate data** from multiple sources (CORDEX, ERA5-Land). The primary goal is to create a spatially-explicit dataset that combines vector surveillance data with environmental variables for machine learning, species distribution modeling, and climate-driven risk assessment.

Each ECDC observation polygon (typically NUTS-3 regions) is spatially matched with climate grid points, creating a detailed dataset where each row represents a geographic location with both mosquito status and monthly climate conditions.

**Important Note on Data Quality:** During the spatial downscaling process (matching coarse ECDC polygons to fine-resolution climate grids), some climate grid points within a "presence" polygon may have extreme climatic conditions that are biologically unsuitable for mosquito survival. To prevent these false positives from contaminating the training dataset, we apply **climate suitability filters** based on established temperature and precipitation thresholds (see [Suitability Filtering](#suitability-filtering-for-data-quality) section). These filters flag locations with climatically impossible conditions, allowing users to exclude them from machine learning models if desired.

---

## Database Purpose

### Primary Use Cases

1. **Machine Learning Training Data**: Supervised learning for habitat suitability prediction
2. **Species Distribution Modeling**: Statistical modeling of mosquito presence/absence
3. **Climate Change Impact Assessment**: Projecting mosquito distributions under future scenarios
4. **Spatial Risk Mapping**: High-resolution risk assessment at sub-regional scales
5. **Comparative Analysis**: Evaluating different climate datasets (CORDEX vs ERA5-Land)

### Data Flow

```
ECDC Polygons          Climate Grid
(NUTS regions)    →    (0.11° / 0.1° resolution)
                  ↓
          Spatial Join (within)
                  ↓
       Paired Dataset (this database)
    (1 row = 1 climate grid point with ECDC status)
```

---

## Database Structure

### Input Data Sources

#### 1. ECDC Vector Surveillance Data

**File:** `20230828_VectorFlatFileGDB.gdb.zip`

| Property | Value |
|----------|-------|
| Format | ESRI Geodatabase (GDB) |
| Geometry | Polygons (NUTS-3 regions primarily) |
| Spatial Coverage | Europe |
| Temporal Coverage | Variable by region (surveillance year) |
| Species | Multiple Aedes species |
| Coordinate System | WGS84 (EPSG:4326) |

**Key Fields from ECDC:**
- `VectorSpeciesName`: Scientific name (e.g., "Aedes albopictus")
- `AssessedDistributionStatus`: Coded status (INV001A, NAT001A, etc.)
- `LocationCode`: NUTS administrative code
- `geometry`: Polygon geometry

#### 2. Climate Data Sources

##### CORDEX (Coordinated Regional Climate Downscaling Experiment)

| Property | Value |
|----------|-------|
| Resolution | 0.11° (~12 km) |
| Domain | EUR-11 (Europe) |
| GCM Model | MPI-M-MPI-ESM-LR |
| RCM Model | SMHI-RCA4 |
| Scenario | RCP 4.5 |
| Ensemble | r1i1p1 |
| Temporal Type | Future projections (decadal climatology) |
| Variables | 2m air temperature, mean precipitation flux |

##### ERA5-Land (ECMWF Reanalysis)

| Property | Value |
|----------|-------|
| Resolution | 0.1° (~9 km) |
| Coverage | Historical reanalysis (1950-present) |
| Product Type | Monthly averaged reanalysis |
| Spatial Extent | Europe [33°N-72°N, -25°W-45°E] |
| Temporal Type | Historical observations (decadal climatology) |
| Variables | 2m temperature, total precipitation |

---

## Database Variables

### Geographic and Spatial Variables

| Variable | Type | Unit | Description | Source |
|----------|------|------|-------------|--------|
| `latitude` | float | degrees | Grid point latitude (WGS84) | Climate grid |
| `longitude` | float | degrees | Grid point longitude (WGS84) | Climate grid |
| `location_id` | string | - | Unique ID: `lat_XX.XXX_lon_YY.YYY` | Derived |
| `geometry` | geometry | - | Point geometry (only in GeoDataFrame, dropped in CSV) | Climate grid |

**Note:** Each row represents one climate grid point, NOT an ECDC polygon. Multiple grid points fall within each ECDC region.

### ECDC Observation Variables

| Variable | Type | Values | Description | Source |
|----------|------|--------|-------------|--------|
| `LocationCode` | string | e.g., "DE123" | NUTS-3 or similar administrative code | ECDC |
| `NUTS_ID` | string | Various | NUTS region identifier (when available) | ECDC |
| `NUTS_NAME` | string | - | Human-readable region name | ECDC |
| `VectorSpeciesName` | string | - | Scientific species name (e.g., "Aedes albopictus") | ECDC |
| `AssessedDistributionStatus` | string | See below | Original ECDC status code | ECDC |
| `status` | string | See mapping | Human-readable status | Derived |
| `presence_numeric` | integer | 0, 1, 2, 3 | Numeric encoding for ML | Derived |

**Status Mapping:**

| ECDC Code | `status` | `presence_numeric` | Interpretation for ML |
|-----------|----------|-------------------|----------------------|
| INV001A | Established | 1 | **Present** (positive class) |
| INV002A | Introduced | 2 | Recently introduced (use cautiously) |
| INV003A | Absent | 0 | **Absent** (negative class) |
| INV004A | No data | 3 | **Exclude** from training |
| INV999A | Unknown | 3 | **Exclude** from training |
| NAT001A | Present | 1 | **Present** (native, positive class) |
| NAT002A | Absent | 0 | **Absent** (negative class) |
| NAT003A | Absent | 0 | **Absent** (negative class) |
| NAT004A | No data | 3 | **Exclude** from training |
| NAT005A | Introduced | 2 | Introduced native (use cautiously) |
| NAT999A | Unknown | 3 | **Exclude** from training |

**For Machine Learning:**
- **Training data**: Use only `presence_numeric` ∈ {0, 1}
- **Positive class (Present)**: `presence_numeric == 1`
- **Negative class (Absent)**: `presence_numeric == 0`
- **Exclude**: `presence_numeric` ∈ {2, 3}

### Climate Variables - Monthly Time Series

#### Core Monthly Data (12-element arrays)

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `temperature_2m_monthly` | list[float] | °C | 12 monthly mean temperatures [Jan, Feb, ..., Dec] |
| `precipitation_monthly` | list[float] | mm | 12 monthly precipitation totals [Jan, Feb, ..., Dec] |
| `months` | list[string] | - | Month names: `['Jan','Feb','Mar',...,'Dec']` |

**Array Structure:**
```python
temperature_2m_monthly = [T_Jan, T_Feb, T_Mar, ..., T_Dec]  # 12 floats
precipitation_monthly = [P_Jan, P_Feb, P_Mar, ..., P_Dec]   # 12 floats
```

**Unit Conversions Applied:**
- **Temperature**: Kelvin → Celsius (T_C = T_K - 273.15)
- **Precipitation (CORDEX)**: kg m⁻² s⁻¹ → mm/month
- **Precipitation (ERA5-Land)**: m/day (mean rate) → mm/month

#### Expanded Monthly Columns (Optional)

When processed with `add_monthly_columns=True` (default in current code):

| Variable Pattern | Type | Unit | Description |
|-----------------|------|------|-------------|
| `temp_Jan_C` through `temp_Dec_C` | float | °C | Individual month temperature |
| `precip_Jan_mm` through `precip_Dec_mm` | float | mm | Individual month precipitation |

**Examples:**
- `temp_Jan_C`: Mean temperature for January (in °C)
- `temp_Jul_C`: Mean temperature for July (in °C)
- `precip_Jan_mm`: Total precipitation for January (in mm)
- `precip_Aug_mm`: Total precipitation for August (in mm)

### Temporal Metadata

| Variable | Type | Description |
|----------|------|-------------|
| `year` | integer | End year of 10-year climatology period used |
| `analysis_year` | integer | Analysis reference year (same as `year`) |

**Important:** The year represents the END of a 10-year climatology period. For example:
- `year=2020` means climatology averaged over 2011-2020
- `year=2050` means climatology averaged over 2041-2050 (for CORDEX projections)

### Auxiliary Suitability Variables (Optional)

These variables are computed during processing but are **NOT the primary purpose** of the database:

| Variable | Type | Values | Description | Use |
|----------|------|--------|-------------|-----|
| `Temperature Suitable` | integer | 0, 1 | Basic temperature threshold check | Quality filter for downscaling artifacts |
| `Precipitation Suitable` | integer | 0, 1 | Basic precipitation threshold check | Quality filter for downscaling artifacts |
| `Suitable` | integer | 0, 1 | Combined binary suitability | Quality filter for downscaling artifacts |

**Note:** These are simple threshold-based flags used primarily to identify and filter out climatically impossible locations that may arise during spatial downscaling. They are NOT sophisticated suitability scores for ecological modeling. For machine learning, use the raw climate variables (`temperature_2m_monthly`, `precipitation_monthly`) along with `presence_numeric` as your target variable, and consider filtering training data using the suitability flags to remove extreme outliers.

---

## Suitability Filtering for Data Quality

### Purpose and Rationale

When downscaling ECDC presence data (typically 10–50 km NUTS-3 polygons) to high-resolution climate grids (~9–12 km), **spatial heterogeneity** can create problems:

1. **The Issue**: A NUTS region marked as "Aedes present" may contain hundreds of climate grid points
2. **The Problem**: Some grid points may represent extreme local climates (e.g., high elevation, very cold winters) that are inconsistent with *established* mosquito populations
3. **The Risk**: These extreme points inherit the polygon label ("present") and become **false positives** at grid scale
4. **The Consequence**: ML/SDM models trained on these artifacts can learn spurious associations and generalize poorly

**Solution**: we compute **simple climate suitability flags** to identify grid points that are *implausible under ECDC establishment criteria*. These flags are intended as **data-quality filters** (QC), not as a full ecological niche model.

---

### Source of Thresholds

The numeric thresholds used below (e.g., winter temperature and annual mean temperature criteria) follow **ECDC mosquito factsheets / establishment guidance** for Europe. Peer-reviewed modelling papers are referenced in the bibliography to support the broader climate–distribution relationship, but the *operational cutoffs* used in the QC flags come from ECDC guidance.

Reference:
- ECDC mosquito factsheet: *Aedes albopictus* (establishment criteria / thresholds)
  https://www.ecdc.europa.eu/en/disease-vectors/facts/mosquito-factsheets/aedes-albopictus

---

### Temperature Suitability Filter

**Function:** `aedes_temperature_suitability()` in `src/aedes_suitability.py`

This filter flags locations where temperature conditions are inconsistent with establishment under ECDC criteria.

#### Algorithm (for *Aedes albopictus*)

A location passes the temperature filter if **ALL** conditions are met:

1. **Winter Survival (ECDC criterion)**: `min(monthly_temps) ≥ -3°C`  
   - Rationale: proxy for egg overwintering survival / establishment constraints in Europe

2. **Annual Development (ECDC criterion)**: `mean(monthly_temps) ≥ 10°C`  
   - Rationale: proxy for sustained population development

3. **Warm Season Length (currently permissive)**: `count(monthly_temps ≥ 10°C) ≥ 0`  
   - Rationale: currently set to 0 (no additional constraint)
   - Note: this is kept for extensibility (can be raised for stricter filtering)

**Mathematical Expression:**
```python
Temperature_Suitable = (min(T_monthly) >= -3.0) and \
                       (mean(T_monthly) >= 10.0) and \
                       (count(T_monthly >= 10.0) >= 0)
```

## Species Differences

- **Aedes albopictus**: thresholds above (ECDC-based)
- **Aedes aegypti**: currently uses the same thresholds as a conservative placeholder; may be updated with species-specific criteria

**Reference:**
- ECDC Technical Report: *Aedes albopictus* factsheet - Establishment thresholds
- URL: https://www.ecdc.europa.eu/en/disease-vectors/facts/mosquito-factsheets/aedes-albopictus

---

---

### Precipitation Suitability Filter

**Function:** `aedes_precipitation_suitability()` in `src/aedes_suitability.py`

This filter flags locations with extremely low annual precipitation where sustained breeding habitat availability is unlikely.

#### Algorithm

A location passes the precipitation filter if:

**Annual Precipitation**: `sum(monthly_precip) > 200 mm/year`

**Rationale:**
- Conservative minimum threshold to avoid classifying very arid regions as plausible due to polygon-to-grid downscaling
- Not intended to model container availability, irrigation, or urban water storage

**Mathematical Expression:**
```python
Precipitation_Suitable = sum(P_monthly) > 200.0
```

**Note:** This is a simple minimum threshold. More sophisticated approaches could consider:
- Seasonal distribution patterns
- Maximum thresholds (flooding disrupts breeding)
- Humidity interactions

---

### Combined Suitability Filter

The overall suitability flag combines both filters:

```python
Suitable = Temperature_Suitable AND Precipitation_Suitable
```

A location is flagged as `Suitable = 1` only if BOTH temperature and precipitation conditions allow survival.

---

### Usage Recommendations

#### For Machine Learning

**Option 1: Filter Training Data (Recommended)**
```python
# Remove climatically impossible locations from training
df_clean = df[df['presence_numeric'].isin([0, 1])].copy()
df_train = df_clean[df_clean['Suitable'] == 1].copy()

# Now use for ML
y = df_train['presence_numeric'].values
X = df_train[feature_columns].values
```

**Option 2: Use as Additional Feature**
```python
# Keep all data but add suitability as feature
features = ['temp_Jan_C', ..., 'precip_Dec_mm', 'Suitable']
X = df_train[features].values
```

**Option 3: Analyze Outliers**
```python
# Identify suspicious "presence" records in unsuitable climate
suspicious = df[(df['presence_numeric'] == 1) & (df['Suitable'] == 0)]
print(f"Found {len(suspicious)} presence records in climatically unsuitable locations")
# These may be data quality issues or microhabitat effects
```

#### For Species Distribution Modeling

```python
# Use suitability to identify potential downscaling artifacts
# Compare observed presence with climate suitability
confusion_matrix = pd.crosstab(
    df['presence_numeric'], 
    df['Suitable'], 
    rownames=['Observed'], 
    colnames=['Climate Suitable']
)
```

#### For Risk Mapping

```python
# Combine observed presence with climate suitability
df['Risk_Category'] = 'Unknown'
df.loc[(df['presence_numeric'] == 1) & (df['Suitable'] == 1), 'Risk_Category'] = 'Established Risk'
df.loc[(df['presence_numeric'] == 0) & (df['Suitable'] == 1), 'Risk_Category'] = 'Potential Risk'
df.loc[(df['presence_numeric'] == 1) & (df['Suitable'] == 0), 'Risk_Category'] = 'Data Quality Issue'
df.loc[(df['presence_numeric'] == 0) & (df['Suitable'] == 0), 'Risk_Category'] = 'Climatically Unsuitable'
```

---

### Limitations and Caveats

1. **Conservative Thresholds**: Filters are intentionally permissive to avoid false negatives
2. **No Upper Limits**: Current implementation doesn't filter extreme heat (>40°C) or excessive precipitation
3. **Binary Classification**: Real suitability exists on a continuum
4. **Microhabitat Effects**: Urban heat islands and sheltered sites may support populations outside filtered ranges
5. **Species-Specific**: *Aedes aegypti* may need different thresholds (currently uses *albopictus* values)

**Recommendation**: Treat suitability flags as **data quality indicators** rather than definitive ecological limits. For ecological modeling, use the raw climate variables with more sophisticated suitability functions.

---

## Data Characteristics

### Spatial Resolution

- **CORDEX**: ~12 km (0.11° grid)
- **ERA5-Land**: ~9 km (0.1° grid)
- **ECDC polygons**: NUTS-3 regions (~10-50 km typical size)

**Result:** Each ECDC polygon contains multiple climate grid points.

### Record Count

Typical database contains **~200,000 to 500,000 rows** depending on:
- Climate data source (grid resolution)
- Geographic extent (filtering)
- ECDC polygon coverage

### Data Duplication Pattern

**Important:** Multiple rows share the same ECDC information:

```
LocationCode  latitude  longitude  status       temp_Jan_C  precip_Jan_mm
DE123         48.100    11.500     Established  2.1         45.2
DE123         48.110    11.510     Established  2.3         46.1
DE123         48.120    11.520     Established  2.2         44.8
```

This is **by design** - each climate grid point gets paired with its containing ECDC polygon.

---

## Usage Examples

### Load Data

```python
import pandas as pd
from pathlib import Path

# Load from ZIP
df = pd.read_csv('data/outputs/ecdc_albopictus_cordex_2020.zip', compression='zip')

# Basic info
print(f"Records: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Climate source: {df['climate_data_source'].iloc[0]}")
```

### Filter for Clean Training Data

```python
import pandas as pd

# Load data
df = pd.read_csv('data/outputs/ecdc_albopictus_cordex_2020.zip', compression='zip')

# Filter 1: Remove uncertain status (No data, Unknown)
df_certain = df[df['presence_numeric'].isin([0, 1])].copy()

# Filter 2: Remove climatically impossible locations
df_train = df_certain[df_certain['Suitable'] == 1].copy()

# Check data loss
print(f"Original: {len(df):,} records")
print(f"After status filter: {len(df_certain):,} ({100*len(df_certain)/len(df):.1f}%)")
print(f"After climate filter: {len(df_train):,} ({100*len(df_train)/len(df_certain):.1f}%)")

# Analyze filtered-out presence records
false_positives = df_certain[(df_certain['presence_numeric'] == 1) & 
                              (df_certain['Suitable'] == 0)]
print(f"\n⚠️  Filtered {len(false_positives)} 'presence' records in unsuitable climate")
print("Temperature range:", false_positives['temp_Jan_C'].describe())
```

### Prepare for Machine Learning

```python
import numpy as np

# Filter for training data (only certain presence/absence)
df_train = df[df['presence_numeric'].isin([0, 1])].copy()

# Extract target variable
y = df_train['presence_numeric'].values  # 0 = absent, 1 = present

# Extract features: convert monthly lists to array
temp_array = np.stack(df_train['temperature_2m_monthly'].values)  # shape: (n, 12)
precip_array = np.stack(df_train['precipitation_monthly'].values)  # shape: (n, 12)

# Option 1: Use all 12 months as features
X = np.hstack([temp_array, precip_array])  # shape: (n, 24)

# Option 2: Use individual monthly columns (if available)
feature_cols = [f'temp_{m}_C' for m in ['Jan','Feb','Mar','Apr','May','Jun',
                                          'Jul','Aug','Sep','Oct','Nov','Dec']]
feature_cols += [f'precip_{m}_mm' for m in ['Jan','Feb','Mar','Apr','May','Jun',
                                              'Jul','Aug','Sep','Oct','Nov','Dec']]
X = df_train[feature_cols].values  # shape: (n, 24)

# Add geographic features
X_with_geo = np.column_stack([
    X,
    df_train['latitude'].values,
    df_train['longitude'].values
])  # shape: (n, 26)

print(f"Training samples: {len(y):,}")
print(f"Positive class: {(y==1).sum():,} ({100*(y==1).mean():.1f}%)")
print(f"Negative class: {(y==0).sum():,} ({100*(y==0).mean():.1f}%)")
```

### Statistical Analysis

```python
# Summary statistics by presence/absence
summary = df[df['presence_numeric'].isin([0,1])].groupby('presence_numeric').agg({
    'temp_Jan_C': ['mean', 'std', 'min', 'max'],
    'temp_Jul_C': ['mean', 'std', 'min', 'max'],
    'precip_Jan_mm': ['mean', 'std', 'min', 'max'],
    'precip_Aug_mm': ['mean', 'std', 'min', 'max'],
})

print(summary)
```

### Spatial Aggregation

```python
# Aggregate to NUTS regions (remove spatial duplication)
df_nuts = df.groupby('LocationCode').agg({
    'latitude': 'mean',
    'longitude': 'mean',
    'presence_numeric': 'first',  # Same for all points in region
    'status': 'first',
    'temperature_2m_monthly': 'first',  # Or use mean across grid points
    'precipitation_monthly': 'first',
    'climate_data_source': 'first',
})

print(f"Unique NUTS regions: {len(df_nuts)}")
```

### Time Series Extraction

```python
# Extract seasonal patterns
def get_seasonal_means(row):
    temps = row['temperature_2m_monthly']
    precips = row['precipitation_monthly']
    return pd.Series({
        'winter_temp': np.mean([temps[11], temps[0], temps[1]]),  # DJF
        'spring_temp': np.mean(temps[2:5]),    # MAM
        'summer_temp': np.mean(temps[5:8]),    # JJA
        'autumn_temp': np.mean(temps[8:11]),   # SON
        'winter_precip': np.sum([precips[11], precips[0], precips[1]]),
        'summer_precip': np.sum(precips[5:8]),
    })

df_seasonal = df.apply(get_seasonal_means, axis=1)
```

---

## Output Files

### File Naming Convention

```
ecdc_{species}_{climate_source}_{year}.zip
```

**Examples:**
- `ecdc_albopictus_cordex_2020.zip` - CORDEX data, 2011-2020 climatology
- `ecdc_albopictus_era5_land_2020.zip` - ERA5-Land data, 2011-2020 climatology
- `ecdc_aegypti_cordex_2050.zip` - CORDEX projection, 2041-2050 climatology

### File Format

| Property | Value |
|----------|-------|
| Container | ZIP archive |
| Content | Single CSV file |
| Encoding | UTF-8 |
| Delimiter | Comma (`,`) |
| Header | Yes (column names in first row) |
| Index | No (`index=False` in pandas) |
| Missing values | Empty string or `NaN` |
| Compression | ZIP (deflate) |

**Load with pandas:**
```python
df = pd.read_csv('path/to/file.zip', compression='zip')
```

---

## Processing Pipeline

### Main Script: `pair_ecdc_copernicus_data.py`

**Command Line Usage:**

```bash
# CORDEX future projection (default)
python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex

# ERA5-Land historical reanalysis
python pair_ecdc_copernicus_data.py --year 2020 --climate-source era5_land

# Full parameter specification
python pair_ecdc_copernicus_data.py \
    --year 2050 \
    --climate-source cordex \
    --parent-dir ./data/inputs/ \
    --ecdc-file 20230828_VectorFlatFileGDB.gdb.zip \
    --output-dir ./data/outputs/
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--year` | string | '2020' | End year of 10-year climatology |
| `--climate-source` | string | 'cordex' | Climate source: 'cordex' or 'era5_land' |
| `--parent-dir` | string | './data/inputs/' | Directory with ECDC GDB file |
| `--ecdc-file` | string | '20230828_VectorFlatFileGDB.gdb.zip' | ECDC geodatabase filename |
| `--output-dir` | string | './data/outputs/' | Output directory |

### Processing Steps

1. **Load ECDC Vector Data**
   ```python
   # Reads GDB, filters for Aedes species
   gdf_albo = gdf[gdf['VectorSpeciesName'].str.contains('albo', case=False)]
   ```

2. **Download/Load Climate Data**
   ```python
   # 10-year monthly climatology
   p_month, t_month = load_climate_data(year=2020, climate_source='cordex')
   ```

3. **Create Climate DataFrame**
   ```python
   # Stack spatial grid to rows, create monthly arrays
   climate_df = create_climate_dataframe(p_month, t_month, year=2020)
   # Each row = one lat/lon point with 12-month arrays
   ```

4. **Spatial Join**
   ```python
   # Join climate points with ECDC polygons
   gdf_merged = gpd.sjoin(gdf_climate_points, gdf_ecdc_polygons, 
                          how='left', predicate='within')
   # Result: climate points inherit ECDC attributes
   ```

5. **Optional: Calculate Simple Suitability Flags**
   ```python
   # Apply climate filters to identify downscaling artifacts
   # These flags help remove climatically impossible training points
   df['Temperature Suitable'] = aedes_temperature_suitability(temps)
   df['Precipitation Suitable'] = aedes_precipitation_suitability(precip)
   df['Suitable'] = np.logical_and(df['Temperature Suitable'], 
                                    df['Precipitation Suitable'])
   ```

6. **Filter for Europe**
   ```python
   # Keep continental Europe, latitude 34-75°N
   df_eur = df[df['LocationCode'].str[:2].isin(european_codes)]
   ```

7. **Save to Database**
   ```python
   # Compress to ZIP, add metadata
   df.to_csv('ecdc_albopictus_cordex_2020.zip', compression='zip', index=False)
   ```

---

## Data Quality Notes

### Spatial Join Behavior

- **Method**: `predicate='within'` (point-in-polygon)
- **Direction**: `how='left'` (keep all climate points)
- **Result**: Climate points outside any ECDC polygon have `NaN` for ECDC variables

### Training Data Filtering

**Recommended filter for ML:**
```python
df_clean = df[df['presence_numeric'].isin([0, 1])].copy()
```

This excludes:
- `presence_numeric == 2` (Introduced - uncertain status)
- `presence_numeric == 3` (No data / Unknown)

### Geographic Coverage

**European filter:**
- NUTS-2 country codes: AT, BE, BG, CH, CY, CZ, DE, DK, EE, ES, FI, FR, GB, GR, HR, HU, IE, IS, IT, LT, LU, LV, MT, NL, NO, PL, PT, RO, SE, SI, SK, etc.
- Latitude: 34°N to 75°N (excludes Arctic islands, overseas territories)

### Missing Data Patterns

1. **Climate points outside ECDC coverage**: ECDC variables are `NaN`
2. **Uncertain mosquito status**: `presence_numeric` ∈ {2, 3}
3. **No surveillance data**: `status == 'No data'`

---

## Climate Data Specifications

### CORDEX Technical Details

**Download Parameters:**
```python
{
    "domain": "europe",
    "horizontal_resolution": "0_11_degree_x_0_11_degree",
    "temporal_resolution": "monthly_mean",
    "experiment": "rcp_4_5",
    "gcm_model": "mpi_m_mpi_esm_lr",
    "rcm_model": "smhi_rca4",
    "ensemble_member": "r1i1p1",
}
```

**Processing:**
- Temperature: tas [K] → t2m [°C]
- Precipitation: pr [kg m⁻² s⁻¹] → tp [mm/month]
  - Formula: `mm = flux * 86400 * days_in_month / 1000`

### ERA5-Land Technical Details

**Download Parameters:**
```python
{
    "product_type": "monthly_averaged_reanalysis",
    "area": [72, -25, 33, 45],  # N, W, S, E
    "variable": ["2m_temperature", "total_precipitation"],
}
```

**Processing:**
- Temperature: t2m [K] → [°C]
- Precipitation: tp [m/day mean rate] → [mm/month]
  - Formula: `mm = m_per_day * 1000 * days_in_month`
  - `days_in_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]`

---

## Troubleshooting

### Common Issues

**1. Large file size**
- Typical: 50-200 MB compressed
- Solution: This is expected given ~200k-500k rows × 50+ columns

**2. Duplicate ECDC information**
- This is by design (multiple climate points per region)
- For region-level analysis, aggregate by `LocationCode`

**3. NaN in ECDC columns**
- Climate points outside any ECDC polygon
- Filter: `df = df.dropna(subset=['LocationCode'])`

**4. Slow spatial join**
- Large datasets take time (~5-10 minutes typical)
- Progress logged to console

**5. "Presence" records in climatically unsuitable locations**
- **Cause**: Spatial downscaling from coarse ECDC polygons to fine climate grids
- **Solution**: Filter using `Suitable == 1` for training data
- **Example**: A mountain region within a "presence" NUTS-3 area may have freezing temperatures

---

## References

### Data Sources

- **ECDC Vector Maps**: https://www.ecdc.europa.eu/en/disease-vectors/surveillance-and-disease-data/mosquito-maps
- **CORDEX**: https://cordex.org/
- **ERA5-Land**: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means

### Scientific Background

### Scientific Background

**ECDC operational guidance (threshold source):**
- ECDC. *Aedes albopictus* mosquito factsheet (establishment criteria / thresholds for Europe).  
  https://www.ecdc.europa.eu/en/disease-vectors/facts/mosquito-factsheets/aedes-albopictus


**Climate suitability and distribution modelling (context / supporting literature):**
- Cunze, S., Kochmann, J., Koch, L. K., & Klimpel, S. (2016). *Aedes albopictus and Its Environmental Limits in Europe.* **PLoS ONE**.  
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162116
- Caminade, C., Medlock, J. M., et al. (2012). *Suitability of European climate for the Asian tiger mosquito Aedes albopictus: recent trends and future scenarios.* **Journal of the Royal Society Interface**, 9(75), 2708–2717.  
  https://royalsocietypublishing.org/rsif/article/9/75/2708/251/Suitability-of-European-climate-for-the-Asian
- Kraemer, M. U. G., Reiner Jr, R. C., Brady, O. J., et al. (2019). *Past and future spread of the arbovirus vectors Aedes aegypti and Aedes albopictus.* **Nature Microbiology**.  
  https://www.nature.com/articles/s41564-019-0376-y

**Spatial / surveillance context:**
- Brady, O. J., & Hay, S. I. (2020). *The Global Expansion of Dengue: How Aedes aegypti Mosquitoes Enabled the First Pandemic Arbovirus.* **Annual Review of Entomology**, 65, 191–208.  
  https://www.annualreviews.org/doi/10.1146/annurev-ento-011019-024918

---

## Contact & Maintenance

For questions, issues, or contributions:
- Open an issue in the repository
- Contact the AIedes project team

**Last Updated**: 2024
**Database Version**: 1.0
**Pipeline Version**: Compatible with Python 3.8+