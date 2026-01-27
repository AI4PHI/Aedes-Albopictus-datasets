"""
Unified climate data downloader supporting both CORDEX and ERA5-Land.
Provides a consistent interface for both climate data sources.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import xarray as xr
import pandas as pd
import numpy as np


def load_climate_data_unified(
    year: int,
    climate_source: str = 'cordex',
    out_dir: str = "./data/inputs/"
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Unified function to load climate data from either CORDEX or ERA5-Land.
    Returns monthly climatology in consistent format.
    
    Parameters:
        year: End year for climatology (uses 10-year period ending at this year)
        climate_source: 'cordex' or 'era5_land'
        out_dir: Output directory for downloaded files
        
    Returns:
        Tuple of (p_month, t_month) as xarray DataArrays
        - p_month: Precipitation in mm/month (12 months)
        - t_month: Temperature in °C (12 months)
    """
    if climate_source == 'cordex':
        return _load_cordex_climatology(year, out_dir)
    elif climate_source == 'era5_land':
        return _load_era5_climatology(year, out_dir)
    else:
        raise ValueError(f"Unknown climate_source: {climate_source}. Use 'cordex' or 'era5_land'")


def _load_cordex_climatology(year: int, out_dir: str) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load CORDEX 10-year climatology (original implementation)."""
    import importlib
    from src import copernicus
    importlib.reload(copernicus)
    
    print(f"📡 Loading CORDEX projection data for {year-9}-{year}...")
    
    ds = copernicus.load_eurocordex_monthly(
        out_dir=out_dir,
        domain="europe",
        resolution="0_11_degree_x_0_11_degree",
        experiment="rcp_4_5",
        gcm_model="mpi_m_mpi_esm_lr",
        rcm_model="smhi_rca4",
        ensemble="r1i1p1",
        year_start=str(year - 9),
        year_end=str(year),
        variables=("2m_air_temperature", "mean_precipitation_flux"),
        expected_suffix="v1a_mon",
        force_redownload=False,
    )
    
    # Compute monthly climatology
    p_month, t_month = copernicus.climate_climatology(ds)
    
    return p_month, t_month


def _load_era5_climatology(year: int, out_dir: str) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load ERA5-Land 10-year climatology (matching CORDEX pattern)."""
    from src.era5_land_downloader import ERA5LandDownloader
    
    print(f"📡 Loading ERA5-Land reanalysis data for {year-9}-{year}...")
    
    year_start = year - 9
    year_end = year
    
    out_dir_path = Path(out_dir) / "era5_land"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    downloader = ERA5LandDownloader(base_output_dir=str(out_dir_path))
    
    # Download monthly climatology for both variables
    print("  Downloading temperature...")
    temp_file = downloader.download_monthly_climatology("2m_temperature", year_start, year_end, str(out_dir_path))
    
    print("  Downloading precipitation...")
    precip_file = downloader.download_monthly_climatology("total_precipitation", year_start, year_end, str(out_dir_path))
    
    # Load datasets
    ds_temp = xr.open_dataset(temp_file)
    ds_precip = xr.open_dataset(precip_file)
    
    # Get variable names (ERA5 uses different internal names)
    temp_var = "t2m" if "t2m" in ds_temp else "2m_temperature"
    precip_var = "tp" if "tp" in ds_precip else "total_precipitation"
    
    # Rename valid_time to time for compatibility with groupby operations
    if "valid_time" in ds_temp.dims:
        ds_temp = ds_temp.rename({"valid_time": "time"})
    if "valid_time" in ds_precip.dims:
        ds_precip = ds_precip.rename({"valid_time": "time"})
    
    # Compute 10-year monthly climatology (group by month and average)
    print("  📊 Computing 10-year monthly climatology...")
    t_month = ds_temp[temp_var].groupby("time.month").mean()
    p_month = ds_precip[precip_var].groupby("time.month").mean()
    
    # Convert temperature from K to °C
    t_month = t_month - 273.15
    t_month.attrs["units"] = "°C"
    
    # ✅ FIX: Convert precipitation from m/day to mm/month
    # ERA5-Land monthly means give mean daily rate in m/day
    days_per_month = xr.DataArray(
        [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], 
        dims=['month'], 
        coords={'month': range(1, 13)}
    )
    p_month = p_month * 1000.0 * days_per_month  # m/day → mm/month
    p_month.attrs["units"] = "mm/month"
    
    print(f"  📊 Precipitation range: {float(p_month.min()):.1f} to {float(p_month.max()):.1f} mm/month")
    print(f"  📊 Temperature range: {float(t_month.min()):.1f} to {float(t_month.max()):.1f} °C")
    
    # Standardize coordinate names to match CORDEX (latitude/longitude -> lat/lon)
    coord_rename = {}
    if "latitude" in t_month.dims:
        coord_rename["latitude"] = "lat"
    if "longitude" in t_month.dims:
        coord_rename["longitude"] = "lon"
    
    if coord_rename:
        t_month = t_month.rename(coord_rename)
        p_month = p_month.rename(coord_rename)
    
    # Close datasets
    ds_temp.close()
    ds_precip.close()
    
    print(f"  ✅ ERA5-Land climatology ready: {t_month.shape}")
    
    return p_month, t_month


def create_climate_dataframe(
    p_month: xr.DataArray,
    t_month: xr.DataArray,
    year: int,
    add_monthly_columns: bool = True
) -> pd.DataFrame:
    """
    Vectorized: stack 2D grid to 1D 'cell' dim, then build rows per cell
    with 12-month lists for temperature (°C) and precipitation (mm).
    Expects p_month dims like ('month', <y>, <x>) and coords 'lat','lon'.
    
    Copied from copernicus.py create_climate_dataframe_fast()
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
    # Check if temperature is already in Celsius (for ERA5-Land)
    temp_units = t_month.attrs.get("units", "")
    if temp_units in ("°C", "C", "celsius"):
        # Already in Celsius, no conversion needed
        temp_c = t_stacked.transpose("cell", "month").values
        print("   Temperature already in °C")
    else:
        # Assume Kelvin, convert to Celsius
        temp_c = (t_stacked - 273.15).transpose("cell", "month").values
        print("   Converted temperature from K to °C")
    
    # Check if precipitation is already in mm (for ERA5-Land)
    precip_units = p_month.attrs.get("units", "")
    if precip_units in ("mm", "mm/month"):
        # Already in mm, no conversion needed
        precip_mm = p_stacked.transpose("cell", "month").values
        print("   Precipitation already in mm/month")
    else:
        # Assume meters, convert to mm
        precip_mm = (p_stacked * 1000.0).transpose("cell", "month").values
        print("   Converted precipitation from m to mm")

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
        "year": int(year),
        "temperature_2m_monthly": [row.tolist() for row in temp_c],
        "precipitation_monthly":  [row.tolist() for row in precip_mm],
        "months": [MONTH_NAMES] * len(lat_vals),
    })
    df["location_id"] = (
        "lat_" + df["latitude"].round(3).astype(str) + "_lon_" + df["longitude"].round(3).astype(str)
    )
    
    # Optionally add individual monthly columns
    if add_monthly_columns:
        df = add_detailed_monthly_columns(df)
    
    return df


def add_detailed_monthly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add individual monthly columns to the DataFrame."""
    MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for i, m in enumerate(MONTH_NAMES):
        df[f"temp_{m}_C"]   = df["temperature_2m_monthly"].str[i]
        df[f"precip_{m}_mm"] = df["precipitation_monthly"].str[i]
    return df