"""
ERA5-Land downloader that extends the existing Copernicus CDS infrastructure.
Builds on the existing CDS setup in the classifier module.
"""

from __future__ import annotations
import os
import cdsapi
import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import time
import zipfile


class ERA5LandDownloader:
    """
    Downloads and processes ERA5-Land reanalysis data from Copernicus CDS.
    Designed to work with the existing CDS API setup.
    """

    def __init__(self, base_output_dir: str = "/eos/jeodpp/data/projects/ETOHA/DATA/ClimateData/Copernicus_data"):
        self.base_output_dir = Path(base_output_dir)
        self.client = None

        # Variable mapping for ERA5-Land
        self.era5_variable_mapping = {
            "total_precipitation": "total_precipitation",
            "10m_u_component_of_wind": "10m_u_component_of_wind",
            "10m_v_component_of_wind": "10m_v_component_of_wind",
            "2m_dewpoint_temperature": "2m_dewpoint_temperature",
            "2m_temperature": "2m_temperature",
            "surface_net_thermal_radiation": "surface_net_thermal_radiation",
            "surface_net_solar_radiation": "surface_net_solar_radiation",
            "surface_pressure": "surface_pressure",
            "skin_temperature": "skin_temperature",
            "surface_sensible_heat_flux": "surface_sensible_heat_flux",
            "surface_latent_heat_flux": "surface_latent_heat_flux",
            "surface_thermal_radiation_downwards": "surface_thermal_radiation_downwards",
            "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1",
            "potential_evaporation": "potential_evaporation",
            "evaporation_from_vegetation_transpiration": "evaporation_from_vegetation_transpiration",
            "total_evaporation": "total_evaporation"
        }

        # Variables that use statistics processing (min/max/mean)
        self.stats_variables = {
            "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature",
            "2m_temperature", "surface_net_thermal_radiation", "surface_net_solar_radiation",
            "surface_pressure", "skin_temperature", "surface_sensible_heat_flux",
            "surface_latent_heat_flux", "surface_thermal_radiation_downwards",
            "volumetric_soil_water_layer_1"
        }

        # Internal variable name mapping (ERA5-Land uses different names internally)
        self.internal_mapping = {
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "2m_temperature": "t2m",
            "2m_dewpoint_temperature": "d2m",
            "volumetric_soil_water_layer_1": "swvl1",
            "total_precipitation": "tp",
            "surface_pressure": "sp",
            "skin_temperature": "skt"
        }

    def _get_cds_client(self) -> cdsapi.Client:
        """Initialize CDS API client (reuses existing setup)."""
        if self.client is None:
            try:
                self.client = cdsapi.Client()
                print("✅ Using existing CDS API configuration")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CDS API client: {e}")
        return self.client

    def _get_expected_file_path(self, variable: str, year: int, freq: str = "daily") -> Path:
        """Get expected processed file path."""
        year_dir = self.base_output_dir / "europe" / "data" / str(year)

        if variable in self.stats_variables:
            filename = f"{variable}_{freq}_stats_{year}.nc"
        else:
            filename = f"{variable}_{freq}_cum_{year}.nc"

        return year_dir / filename

    def _get_raw_file_path(self, variable: str, year: int) -> Path:
        """Get raw download file path."""
        raw_dir = self.base_output_dir / "raw" / str(year)
        raw_dir.mkdir(parents=True, exist_ok=True)
        return raw_dir / f"cds_era5_land_{variable}_{year}.nc"

    def _extract_zip_if_needed(self, file_path: Path) -> Path:
        """Extract ZIP file if the downloaded file is a ZIP archive."""
        if not file_path.exists():
            return file_path

        # Check if file is a ZIP archive
        try:
            if zipfile.is_zipfile(file_path):
                print(f"  📦 Extracting ZIP archive: {file_path.name}")

                # Create extraction directory
                extract_dir = file_path.parent / f"extracted_{file_path.stem}"
                extract_dir.mkdir(exist_ok=True)

                # Extract ZIP contents
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Find the NetCDF file in extracted content
                nc_files = list(extract_dir.glob("*.nc"))
                if nc_files:
                    extracted_nc = nc_files[0]  # Use first NetCDF file found

                    # Use a temporary name first to avoid conflicts
                    temp_nc = file_path.parent / f"{file_path.stem}_temp.nc"
                    if temp_nc.exists():
                        temp_nc.unlink()

                    # Move extracted file to temp location
                    extracted_nc.rename(temp_nc)

                    # Clean up extraction directory
                    import shutil
                    shutil.rmtree(extract_dir)

                    # Now safely replace the ZIP with the NetCDF (ZIP file is already closed)
                    file_path.unlink()  # Remove ZIP file
                    temp_nc.rename(file_path)  # Move temp to original location

                    # Small delay to ensure file system has settled
                    time.sleep(1)

                    print(f"  ✅ Extracted NetCDF: {file_path.name}")
                    print(f"      File size: {file_path.stat().st_size:,} bytes")
                    return file_path
                else:
                    print(f"  ❌ No NetCDF files found in ZIP archive")
                    return file_path
            else:
                return file_path

        except Exception as e:
            print(f"  ⚠️ Error processing potential ZIP file {file_path}: {e}")
            return file_path

    def _validate_and_extract_netcdf_file(self, file_path: Path) -> tuple[bool, Path]:
        """
        Validate that a netCDF file can be opened properly.
        Returns (is_valid, actual_file_path)
        """
        if not file_path.exists():
            return False, file_path

        # First, check if it's a ZIP file and extract if needed
        actual_file = self._extract_zip_if_needed(file_path)

        if not actual_file.exists():
            return False, actual_file

        try:
            print(f"  🔍 Validating file: {actual_file.name} ({actual_file.stat().st_size:,} bytes)")

            # Try to open with explicit engine specification
            with xr.open_dataset(actual_file, engine='netcdf4') as ds:
                # Check if file has data
                if len(ds.data_vars) == 0:
                    print(f"  ⚠️ File {actual_file} has no data variables")
                    return False, actual_file

                print(f"  📊 Found variables: {list(ds.data_vars.keys())}")
                print(f"  📏 Dimensions: {dict(ds.dims)}")

                # Try to access the data to ensure it's not corrupted
                for var in list(ds.data_vars.keys())[:1]:  # Just test first variable
                    _ = ds[var].dims
                    print(f"  ✅ Variable '{var}' accessible")

            print(f"  ✅ Validation successful for {actual_file.name}")
            return True, actual_file

        except Exception as e:
            print(f"  ❌ File validation failed for {actual_file.name}")
            print(f"      Error: {type(e).__name__}: {e}")
            print(f"      File exists: {actual_file.exists()}")
            if actual_file.exists():
                print(f"      File size: {actual_file.stat().st_size:,} bytes")
            return False, actual_file

    def _validate_netcdf_file(self, file_path: Path) -> bool:
        """Backward compatibility wrapper."""
        is_valid, _ = self._validate_and_extract_netcdf_file(file_path)
        return is_valid

    def _download_monthly_chunk(self, variable: str, year: int, month: int, client: cdsapi.Client, max_retries: int = 3) -> Path:
        """Download data for a single month to avoid size limits."""
        chunk_file = self._get_raw_file_path(variable, year).parent / f"chunk_{variable}_{year}_{month:02d}.nc"

        # European bounding box
        area = [75, -25, 25, 45]  # North, West, South, East

        for attempt in range(max_retries):
            try:
                print(f"  📥 Downloading {variable} for {year}-{month:02d} (attempt {attempt+1}/{max_retries})...")

                # Remove existing file if it exists
                if chunk_file.exists():
                    chunk_file.unlink()

                client.retrieve(
                    'reanalysis-era5-land',
                    {
                        'variable': self.era5_variable_mapping[variable],
                        'year': str(year),
                        'month': f"{month:02d}",
                        'day': [f"{d:02d}" for d in range(1, 32)],  # All days in month
                        'time': ['00:00', '06:00', '12:00', '18:00'],  # 4 times per day to reduce size
                        'area': area,
                        'grid': [0.1, 0.1],
                        'format': 'netcdf',
                    },
                    str(chunk_file)
                )

                # Wait a moment for file to be fully written
                time.sleep(2)

                # Validate the downloaded file and handle ZIP extraction
                is_valid, actual_file = self._validate_and_extract_netcdf_file(chunk_file)
                if is_valid:
                    print(f"  ✅ Successfully downloaded and validated {year}-{month:02d}")
                    return actual_file
                else:
                    print(f"  ❌ Downloaded file failed validation (attempt {attempt+1})")
                    # Clean up both original and extracted files
                    if chunk_file.exists():
                        chunk_file.unlink()
                    if actual_file != chunk_file and actual_file.exists():
                        actual_file.unlink()

            except Exception as e:
                print(f"  ❌ Download attempt {attempt+1} failed: {e}")
                if chunk_file.exists():
                    chunk_file.unlink()

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Exponential backoff
                    print(f"  ⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        raise RuntimeError(f"Failed to download valid chunk for {variable} {year}-{month:02d} after {max_retries} attempts")

    def download_era5_land_data(self, variable: str, year: int, overwrite: bool = False) -> Path:
        """
        Download ERA5-Land reanalysis data for a specific variable and year.
        Downloads in monthly chunks to avoid CDS size limits.

        Args:
            variable: ERA5-Land variable name
            year: Year to download
            overwrite: Whether to overwrite existing files

        Returns:
            Path to downloaded raw file
        """
        if variable not in self.era5_variable_mapping:
            raise ValueError(f"Unsupported variable: {variable}")

        raw_file = self._get_raw_file_path(variable, year)

        if raw_file.exists() and not overwrite:
            print(f"✅ Raw file already exists: {raw_file}")
            return raw_file

        print(f"🔄 Downloading ERA5-Land {variable} for {year} in monthly chunks...")

        client = self._get_cds_client()
        chunk_files = []

        try:
            # Download each month separately
            for month in range(1, 13):
                try:
                    chunk_file = self._download_monthly_chunk(variable, year, month, client)
                    chunk_files.append(chunk_file)
                    print(f"  ✅ Downloaded chunk for {year}-{month:02d}")
                except Exception as e:
                    print(f"  ⚠️ Failed to download {year}-{month:02d}: {e}")
                    # Continue with other months

            # Merge monthly chunks into single file
            if chunk_files:
                print(f"🔗 Merging {len(chunk_files)} monthly chunks...")
                datasets = []
                valid_chunk_files = []
                for chunk_file in chunk_files:
                    if chunk_file.exists():
                        is_valid, actual_file = self._validate_and_extract_netcdf_file(chunk_file)
                        if is_valid:
                            try:
                                ds = xr.open_dataset(actual_file, engine='netcdf4')
                                datasets.append(ds)
                                valid_chunk_files.append(actual_file)
                                print(f"  ✅ Loaded chunk: {actual_file.name}")
                            except Exception as e:
                                print(f"  ⚠️ Failed to load chunk {actual_file}: {e}")
                        else:
                            print(f"  ⚠️ Invalid chunk file: {chunk_file.name}")

                if datasets:
                    print(f"  🔗 Concatenating {len(datasets)} valid datasets...")
                    # Use valid_time dimension for ERA5-Land data
                    merged_ds = xr.concat(datasets, dim="valid_time")

                    # Sort by time to ensure proper ordering
                    merged_ds = merged_ds.sortby("valid_time")

                    # Save with explicit engine and compression
                    merged_ds.to_netcdf(
                        raw_file,
                        engine='netcdf4',
                        encoding={var: {'zlib': True, 'complevel': 4} for var in merged_ds.data_vars}
                    )

                    # Close datasets to free memory
                    for ds in datasets:
                        ds.close()
                    merged_ds.close()

                    print(f"✅ Merged and saved: {raw_file}")

                    # Validate the merged file
                    if self._validate_netcdf_file(raw_file):
                        # Clean up chunk files only if merge was successful
                        for chunk_file in valid_chunk_files:
                            if chunk_file.exists():
                                chunk_file.unlink()
                        print(f"  🧹 Cleaned up {len(valid_chunk_files)} chunk files")
                    else:
                        raise RuntimeError("Merged file failed validation")
                else:
                    raise RuntimeError("No valid chunks to merge")
            else:
                raise RuntimeError("No chunks downloaded successfully")

            return raw_file

        except Exception as e:
            # Clean up on failure
            if raw_file.exists():
                raw_file.unlink()
            for chunk_file in chunk_files:
                if chunk_file.exists():
                    chunk_file.unlink()
            raise RuntimeError(f"Download failed for {variable} {year}: {e}")

    def subset_to_europe(self, ds: xr.Dataset, lat_min: float = 33, lat_max: float = 72,
                        lon_min: float = -25, lon_max: float = 45) -> xr.Dataset:
        """Subset dataset to European region."""
        # Handle longitude coordinate conversion
        if 'longitude' in ds.coords:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds = ds.sortby("longitude")

        # Subset to European bounds
        ds = ds.sel(longitude=slice(lon_min, lon_max))

        if ds.latitude[0] > ds.latitude[-1]:
            ds = ds.sel(latitude=slice(lat_max, lat_min))
        else:
            ds = ds.sel(latitude=slice(lat_min, lat_max))

        return ds

    def get_internal_variable_name(self, ds: xr.Dataset, expected_var: str) -> str:
        """Get the actual variable name from the dataset."""
        # First try the expected name
        if expected_var in ds.data_vars:
            return expected_var

        # Try the internal mapping
        if expected_var in self.internal_mapping:
            internal_var = self.internal_mapping[expected_var]
            if internal_var in ds.data_vars:
                return internal_var

        # If only one variable, use it
        if len(ds.data_vars) == 1:
            return list(ds.data_vars.keys())[0]

        # Fallback
        print(f"Warning: Could not find variable {expected_var}, using {list(ds.data_vars.keys())[0]}")
        return list(ds.data_vars.keys())[0]

    def process_era5_data(self, variable: str, year: int, freq: str = "daily",
                         raw_file: Optional[Path] = None) -> Path:
        """
        Process raw ERA5-Land data into the expected format.

        Args:
            variable: Variable name
            year: Year
            freq: Frequency ("daily", "weekly", "monthly")
            raw_file: Path to raw file

        Returns:
            Path to processed file
        """
        if raw_file is None:
            raw_file = self._get_raw_file_path(variable, year)

        if not raw_file.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_file}")

        # Set resampling frequency
        resample_map = {"daily": "1D", "weekly": "1W", "monthly": "1M"}
        if freq not in resample_map:
            raise ValueError(f"Invalid frequency: {freq}")
        resample_str = resample_map[freq]

        # Output directory
        output_dir = self.base_output_dir / "europe" / "data" / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Processing {variable} for {year} at {freq} frequency...")

        # Validate raw file before processing
        if not self._validate_netcdf_file(raw_file):
            raise ValueError(f"Raw file failed validation: {raw_file}")

        # Load and process with explicit engine
        ds = xr.open_dataset(raw_file, engine='netcdf4')
        internal_var = self.get_internal_variable_name(ds, variable)
        print(f"  Using internal variable: '{internal_var}'")

        # Subset to Europe
        ds_europe = self.subset_to_europe(ds)

        if internal_var not in ds_europe:
            raise ValueError(f"Variable '{internal_var}' not found in European subset")

        # Process based on variable type
        # Determine the correct time dimension name
        time_dim = "valid_time" if "valid_time" in ds_europe.dims else "time"
        print(f"  🕐 Using time dimension: '{time_dim}'")

        if variable in self.stats_variables:
            # Statistics processing
            da = ds_europe[internal_var].resample({time_dim: resample_str})
            ds_out = xr.Dataset({
                f"{variable}_min": da.min(),
                f"{variable}_max": da.max(),
                f"{variable}_mean": da.mean()
            })
            outfile = output_dir / f"{variable}_{freq}_stats_{year}.nc"
        else:
            # Cumulative processing
            ds_out = xr.Dataset({
                f"{variable}_sum": ds_europe[internal_var].resample({time_dim: resample_str}).sum()
            })
            outfile = output_dir / f"{variable}_{freq}_cum_{year}.nc"

        # Save processed data with explicit engine
        ds_out.to_netcdf(outfile, engine='netcdf4')

        # Close datasets to free memory
        ds.close()
        ds_out.close()

        print(f"✅ Saved processed data: {outfile}")

        # Validate processed file
        if not self._validate_netcdf_file(outfile):
            raise RuntimeError(f"Processed file failed validation: {outfile}")

        return outfile

    def ensure_data_available(self, variable: str, year: int, freq: str = "daily",
                            force_redownload: bool = False) -> Path:
        """
        Ensure processed ERA5-Land data is available for a variable/year.
        Downloads and processes if missing.

        Args:
            variable: Variable name
            year: Year
            freq: Frequency
            force_redownload: Force redownload even if exists

        Returns:
            Path to processed file
        """
        expected_file = self._get_expected_file_path(variable, year, freq)

        if expected_file.exists() and not force_redownload:
            print(f"✅ Data already available: {expected_file}")
            return expected_file

        print(f"📥 Ensuring data availability for {variable} {year}...")

        # Download raw data
        raw_file = self.download_era5_land_data(variable, year, overwrite=force_redownload)

        # Process to final format
        processed_file = self.process_era5_data(variable, year, freq, raw_file)

        return processed_file

    def ensure_multiple_years(self, variable: str, years: List[int], freq: str = "daily",
                            force_redownload: bool = False) -> List[Path]:
        """Ensure data is available for multiple years."""
        processed_files = []

        for year in years:
            try:
                processed_file = self.ensure_data_available(variable, year, freq, force_redownload)
                processed_files.append(processed_file)
            except Exception as e:
                print(f"❌ Failed to ensure data for {variable} {year}: {e}")

        return processed_files


# Convenience function that matches your existing usage pattern
def ensure_era5_data_available(variable: str, year: int,
                             base_dir: str = "/eos/jeodpp/data/projects/ETOHA/DATA/ClimateData/Copernicus_data",
                             freq: str = "daily") -> Path:
    """Convenience function to ensure a single variable/year is available."""
    downloader = ERA5LandDownloader(base_dir)
    return downloader.ensure_data_available(variable, year, freq)