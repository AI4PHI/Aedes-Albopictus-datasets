from __future__ import annotations
import os
from pathlib import Path
import zipfile
from typing import Iterable, Dict, Optional

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1) DOWNLOAD / LOAD, MERGE, CONVERT
# ──────────────────────────────────────────────────────────────────────────────

def _scenario_text(experiment: str) -> str:
    mapping = {
        "rcp_2_6": "rcp26",
        "rcp_4_5": "rcp45",
        "rcp_8_5": "rcp85",
        "historical": "historical",
    }
    return mapping.get(experiment, experiment.replace("_", ""))

def _short_name(variable: str) -> str:
    mapping = {
        "2m_air_temperature": "tas",
        "mean_precipitation_flux": "pr",
    }
    if variable not in mapping:
        raise ValueError(f"Unsupported variable '{variable}'")
    return mapping[variable]

def _expected_raw_filenames(
    gcm_model: str,
    rcm_model: str,
    experiment_text: str,
    ensemble: str,
    year_start: str,
    year_end: str,
    suffix: str = "v1a_mon",
) -> Dict[str, str]:
    gcm = gcm_model.upper().replace("_", "-")
    rcm = rcm_model.upper().replace("_", "-")
    period = f"{year_start}01-{year_end}12.nc"
    return {
        "tas": f"tas_EUR-11_{gcm}_{experiment_text}_{ensemble}_{rcm}_{suffix}_{period}",
        "pr":  f"pr_EUR-11_{gcm}_{experiment_text}_{ensemble}_{rcm}_{suffix}_{period}",
    }

def _download_variable_zip(
    out_dir: Path,
    domain: str,
    resolution: str,
    experiment: str,
    gcm_model: str,
    rcm_model: str,
    ensemble: str,
    year_start: str,
    year_end: str,
    variable: str,
) -> None:
    months = [f"{m:02d}" for m in range(1, 13)]
    tmp_zip = out_dir / f"tmp_cordex_{variable}.zip"
    print(f"🔄 Downloading {variable} {year_start}-{year_end} …")
    c = cdsapi.Client()
    c.retrieve(
        "projections-cordex-domains-single-levels",
        {
            "download_format":       "zip",
            "data_format":           "netcdf_legacy",
            "domain":                domain,
            "horizontal_resolution": resolution,
            "temporal_resolution":   "monthly_mean",
            "start_year":            [year_start],
            "end_year":              [year_end],
            "month":                 months,
            "experiment":            experiment,
            "gcm_model":             gcm_model,
            "rcm_model":             rcm_model,
            "ensemble_member":       ensemble,
            "variable":              variable,
        },
        str(tmp_zip),
    )
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(out_dir)
    tmp_zip.unlink(missing_ok=True)
    print(f"✅ Downloaded & extracted {variable}")

def _ensure_raw_files(
    out_dir: Path,
    domain: str,
    resolution: str,
    experiment: str,
    gcm_model: str,
    rcm_model: str,
    ensemble: str,
    year_start: str,
    year_end: str,
    variables: Iterable[str],
    expected_names: Dict[str, str],
) -> Dict[str, Path]:
    paths = {k: out_dir / v for k, v in expected_names.items()}
    missing = [
        v for v in variables
        if not (out_dir / expected_names[_short_name(v)]).exists()
    ]
    if missing:
        for var in missing:
            _download_variable_zip(
                out_dir, domain, resolution, experiment,
                gcm_model, rcm_model, ensemble, year_start, year_end, var
            )
    else:
        print("✅ Raw CORDEX files already present—skipping download.")
    for short, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")
    return paths

def _process_merge_to_ds(tas_path: Path, pr_path: Path) -> xr.Dataset:
    ds_t2m = xr.open_dataset(tas_path)
    ds_pr  = xr.open_dataset(pr_path)
    ds = xr.merge([ds_t2m, ds_pr])
    ds = ds.rename({"tas": "t2m"})  # Kelvin
    # pr flux (kg m-2 s-1) → monthly total (m)
    flux = ds["pr"]
    sec_per_day = 24 * 3600
    days_in_month = ds["time"].dt.days_in_month
    tp_m = flux * (sec_per_day * days_in_month) / 1000.0
    tp_m.name = "tp"
    ds = ds.drop_vars("pr")
    ds["tp"] = tp_m
    return ds

def load_eurocordex_monthly(
    out_dir: str | Path = "./data/inputs/",
    *,
    domain: str = "europe",
    resolution: str = "0_11_degree_x_0_11_degree",
    experiment: str = "rcp_4_5",
    gcm_model: str = "mpi_m_mpi_esm_lr",
    rcm_model: str = "smhi_rca4",
    ensemble: str = "r1i1p1",
    year_start: str = "2041",
    year_end: str = "2050",
    variables: Iterable[str] = ("2m_air_temperature", "mean_precipitation_flux"),
    expected_suffix: str = "v1a_mon",
    force_redownload: bool = False,
) -> xr.Dataset:
    """
    Returns merged Dataset with:
      - t2m [K]
      - tp  [m per month]
    Downloads from CDS only if needed.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_text = _scenario_text(experiment)
    expected = _expected_raw_filenames(
        gcm_model=gcm_model, rcm_model=rcm_model,
        experiment_text=exp_text, ensemble=ensemble,
        year_start=year_start, year_end=year_end, suffix=expected_suffix
    )

    if force_redownload:
        for p in (out_dir / expected["tas"], out_dir / expected["pr"]):
            if p.exists():
                p.unlink()
        print("♻️ Forced redownload: removed existing raw files.")

    paths = _ensure_raw_files(
        out_dir, domain, resolution, experiment, gcm_model, rcm_model, ensemble,
        year_start, year_end, variables, expected
    )
    ds = _process_merge_to_ds(paths["tas"], paths["pr"])
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# 2) BUILD CLIMATE DATAFRAME (VECTORIZED)
# ──────────────────────────────────────────────────────────────────────────────

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def climate_climatology(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    From full-period ds, return monthly climatologies:
      p_month = tp (m) monthly mean, dims: (month, rlat, rlon)
      t_month = t2m (K) monthly mean, dims: (month, rlat, rlon)
    """
    p_month = ds["tp"].groupby("time.month").mean("time")
    t_month = ds["t2m"].groupby("time.month").mean("time")
    # rename month→time only if you need to save; for processing we keep 'month'
    return p_month, t_month

def create_climate_dataframe_fast(p_month: xr.DataArray, t_month: xr.DataArray, *, year: str) -> pd.DataFrame:
    """
    Vectorized: stack 2D grid to 1D 'cell' dim, then build rows per cell
    with 12-month lists for temperature (°C) and precipitation (mm).
    Expects p_month dims like ('month', <y>, <x>) and coords 'lat','lon'.
    """
    # 1) Identify spatial dims (anything not time/month)
    spatial_dims = [d for d in p_month.dims if d not in ("time", "month")]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected exactly 2 spatial dims, got {spatial_dims}")

    ydim, xdim = spatial_dims

    # 2) Ensure lat/lon exist and are aligned/broadcastable to the spatial dims
    if ("lat" not in p_month.coords) or ("lon" not in p_month.coords):
        raise KeyError("Expected 2D coordinates 'lat' and 'lon' in DataArray")

    lat = p_month["lat"]
    lon = p_month["lon"]

    # Handle different shapes for lat/lon:
    # - 2D lat/lon on (ydim, xdim)
    # - OR 1D lat on (ydim,) and 1D lon on (xdim,) -> broadcast to 2D
    if lat.ndim == 1 and lon.ndim == 1 and lat.dims == (ydim,) and lon.dims == (xdim,):
        lat, lon = xr.broadcast(lat, lon)  # make them 2D (ydim, xdim)
    elif lat.dims != (ydim, xdim) or lon.dims != (ydim, xdim):
        # Try to broadcast to the grid if possible
        try:
            lat, lon = xr.broadcast(lat, lon, p_month.isel(month=0, drop=True))
            lat = lat.transpose(ydim, xdim)
            lon = lon.transpose(ydim, xdim)
        except Exception as e:
            raise ValueError(
                f"'lat'/'lon' not aligned with spatial dims {spatial_dims}. "
                f"lat.dims={lat.dims}, lon.dims={lon.dims}"
            ) from e

    # 3) Stack grid → cell for data and coords using the SAME spatial dims
    p_stacked = p_month.stack(cell=spatial_dims)            # dims: ('month','cell')
    t_stacked = t_month.stack(cell=spatial_dims)            # dims: ('month','cell')
    lat_stk   = lat.stack(cell=spatial_dims)                # dims: ('cell',)
    lon_stk   = lon.stack(cell=spatial_dims)                # dims: ('cell',)

    # 4) Convert units & move to shape (cells, 12)
    precip_mm = (p_stacked * 1000.0).transpose("cell", "month").values
    temp_c    = (t_stacked - 273.15).transpose("cell", "month").values

    lat_vals = lat_stk.values
    lon_vals = lon_stk.values

    # 5) Filter invalid cells
    valid = ~(np.isnan(lat_vals) | np.isnan(lon_vals))
    lat_vals = lat_vals[valid].astype(float)
    lon_vals = lon_vals[valid].astype(float)
    precip_mm = precip_mm[valid, :]
    temp_c    = temp_c[valid, :]

    # 6) Build DataFrame
    MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df = pd.DataFrame({
        "latitude":  lat_vals,
        "longitude": lon_vals,
        "year": year,
        "temperature_2m_monthly": [row.tolist() for row in temp_c],
        "precipitation_monthly":  [row.tolist() for row in precip_mm],
        "months": [MONTH_NAMES] * len(lat_vals),
    })
    df["location_id"] = (
        "lat_" + df["latitude"].round(3).astype(str) + "_lon_" + df["longitude"].round(3).astype(str)
    )
    return df


def add_detailed_monthly_columns(df: pd.DataFrame) -> pd.DataFrame:
    for i, m in enumerate(MONTH_NAMES):
        df[f"temp_{m}_C"]   = df["temperature_2m_monthly"].str[i]
        df[f"precip_{m}_mm"] = df["precipitation_monthly"].str[i]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3) EXAMPLE USAGE (EDIT ONLY THIS BLOCK)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # — USER PARAMS —
    year = "2050"
    ds = load_eurocordex_monthly(
        out_dir="./data/inputs/",
        domain="europe",
        resolution="0_11_degree_x_0_11_degree",
        experiment="rcp_4_5",
        gcm_model="mpi_m_mpi_esm_lr",
        rcm_model="smhi_rca4",
        ensemble="r1i1p1",
        year_start="2041",
        year_end="2050",
        variables=("2m_air_temperature", "mean_precipitation_flux"),
        expected_suffix="v1a_mon",
        force_redownload=False,  # set True to refresh files
    )

    # Monthly climatology (12 months) from the whole 2041–2050 span
    p_month, t_month = climate_climatology(ds)

    print("Dataset shapes:")
    print("Precipitation:", p_month.shape)
    print("Temperature:",   t_month.shape)
    print("Precipitation coordinates:", list(p_month.coords))
    print("Temperature coordinates:",   list(t_month.coords))
    print("Precipitation dims:",        list(p_month.dims))

    # Build climate dataframe (vectorized)
    climate_df = create_climate_dataframe_fast(p_month, t_month, year=year)
    print("\n📊 DataFrame:", climate_df.shape, "columns:", list(climate_df.columns))

    # Optionally add per-month columns
    climate_df = add_detailed_monthly_columns(climate_df)
    print("\n✅ Added detailed monthly columns.")
    print("\n📋 First 3 rows:")
    print(climate_df.head(3))

    # Quick stats
    print("\n📈 Summary:")
    print(f"Latitude range:  {climate_df['latitude'].min():.2f} → {climate_df['latitude'].max():.2f}")
    print(f"Longitude range: {climate_df['longitude'].min():.2f} → {climate_df['longitude'].max():.2f}")

    # Example access
    first = climate_df.iloc[0]
    print("\n🔍 Example — first location:")
    print(f"Location: {first['latitude']:.2f}°N, {first['longitude']:.2f}°E")
    print("Temperature (°C):", [f"{v:.1f}" for v in first["temperature_2m_monthly"]])
    print("Precipitation (mm):", [f"{v:.1f}" for v in first["precipitation_monthly"]])

    # # Optional save
    # out_csv = f"./data/europe_climate_{year}_monthly.csv"
    # Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    # climate_df.to_csv(out_csv, index=False)
    # print(f"\n💾 Saved to {out_csv}")
