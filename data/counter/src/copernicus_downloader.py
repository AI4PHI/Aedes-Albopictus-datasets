#!/usr/bin/env python3
"""
Copernicus Climate Data Store (CDS) downloader for the counter analysis.
Downloads and processes ERA5-Land data with proper formatting for trap data analysis.
"""

import os
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

import cdsapi
import xarray as xr
import numpy as np


class CopernicusDownloader:
    """
    Downloads and processes ERA5-Land data from Copernicus Climate Data Store.
    Specifically designed for the albopictus trap data analysis pipeline.
    """

    def __init__(self, base_output_dir: str = "../input_data/climate"):
        """
        Initialize the downloader.

        Args:
            base_output_dir: Base directory for storing climate data
                           Default: ../input_data/climate (relative to src/)
                           Structure:
                           - raw/{year}/ : Raw downloaded NetCDF files
                           - processed/europe/daily/{year}/ : Processed regional data
        """
        self.base_output_dir = Path(base_output_dir)
        self.client = None
        
        # DEBUG: Print the actual paths being used
        print(f"🔍 DEBUG: Initializing CopernicusDownloader")
        print(f"🔍 DEBUG: base_output_dir (input): '{base_output_dir}'")
        print(f"🔍 DEBUG: self.base_output_dir (Path): '{self.base_output_dir}'")
        print(f"🔍 DEBUG: Absolute path: '{self.base_output_dir.absolute()}'")
        
        # Define subdirectories
        self.raw_dir = self.base_output_dir / "raw"
        self.processed_dir = self.base_output_dir / "processed" / "europe" / "daily"
        
        print(f"🔍 DEBUG: Raw data dir: '{self.raw_dir.absolute()}'")
        print(f"🔍 DEBUG: Processed data dir: '{self.processed_dir.absolute()}'")

        # Variable mapping from expected names to CDS API names
        self.variable_mapping = {
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
        """Initialize CDS API client if not already done."""
        if self.client is None:
            try:
                self.client = cdsapi.Client()
                print("✅ CDS API client initialized successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CDS API client. Make sure you have configured your CDS API key: {e}")
        return self.client

    def _get_expected_file_path(self, variable: str, year: int, freq: str = "daily") -> Path:
        """Get the expected file path for a processed variable."""
        year_dir = self.processed_dir / str(year)

        if variable in self.stats_variables:
            filename = f"{variable}_{freq}_stats_{year}.nc"
        else:
            filename = f"{variable}_{freq}_cum_{year}.nc"

        return year_dir / filename

    def _get_raw_file_path(self, variable: str, year: int) -> Path:
        """Get the raw downloaded file path."""
        raw_year_dir = self.raw_dir / str(year)
        raw_year_dir.mkdir(parents=True, exist_ok=True)
        return raw_year_dir / f"cds_era5_land_{variable}_{year}.nc"

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
                    print(f"    Found NetCDF: {extracted_nc.name}")

                    # Create new filename based on original ZIP name
                    new_nc_path = file_path.parent / f"{file_path.stem}.nc"

                    # Remove the old ZIP file first
                    file_path.unlink()

                    # Move extracted file
                    extracted_nc.rename(new_nc_path)

                    # Clean up extraction directory
                    import shutil
                    shutil.rmtree(extract_dir)

                    # Give file system more time to settle (increased from 0.5s)
                    time.sleep(2.0)

                    print(f"  ✅ Extracted to: {new_nc_path.name}")
                    return new_nc_path
                else:
                    print(f"  ❌ No NetCDF files found in ZIP archive")
                    return file_path
            else:
                return file_path

        except Exception as e:
            print(f"  ⚠️ Error processing potential ZIP file {file_path}: {e}")
            return file_path

    def _validate_netcdf_file(self, file_path: Path) -> bool:
        """Validate that a NetCDF file can be opened properly."""
        if not file_path.exists():
            print(f"  ❌ File does not exist: {file_path}")
            return False

        try:
            # Give more time for file system to settle
            time.sleep(1.5)
            
            print(f"  🔍 Validating: {file_path.name} ({file_path.stat().st_size:,} bytes)")

            # Check if the file is still a ZIP (shouldn't be, but check anyway)
            if zipfile.is_zipfile(file_path):
                print(f"  ⚠️  File is still in ZIP format, attempting extraction...")
                actual_file = self._extract_zip_if_needed(file_path)
                
                # Give extra time after extraction
                time.sleep(2.0)
                
                # Update the file_path to point to the extracted file
                if actual_file != file_path:
                    file_path = actual_file
                    print(f"  ✅ Using extracted file: {file_path.name}")
                    
                    # Verify the extracted file exists
                    if not file_path.exists():
                        print(f"  ❌ Extracted file does not exist: {file_path}")
                        return False
                else:
                    print(f"  ❌ Extraction failed - file path unchanged")
                    return False

            with xr.open_dataset(file_path, engine='netcdf4') as ds:
                # Check if file has data variables
                if len(ds.data_vars) == 0:
                    print(f"  ⚠️  File {file_path} has no data variables")
                    return False

                print(f"  📊 Found variables: {list(ds.data_vars.keys())}")
                print(f"  📏 Dimensions: {dict(ds.dims)}")
                print(f"  🗺️  Coordinates: {list(ds.coords.keys())}")

                # Try to access a variable to ensure data is readable
                for var in list(ds.data_vars.keys())[:1]:
                    var_dims = ds[var].dims
                    var_shape = ds[var].shape
                    print(f"  📐 Variable '{var}': dims={var_dims}, shape={var_shape}")

                    # Try to access actual data
                    try:
                        sample_data = ds[var].isel({dim: 0 for dim in var_dims}).values
                        print(f"  ✅ Variable '{var}' data accessible")
                    except Exception as data_err:
                        print(f"  ⚠️  Variable '{var}' data access failed: {data_err}")
                        return False

            print(f"✅ Validated: {file_path.name}")
            return True

        except Exception as e:
            print(f"❌ Validation failed for {file_path.name}")
            print(f"  🔍 Error details: {type(e).__name__}: {e}")
            
            # Provide diagnostic information
            try:
                if file_path.exists():
                    print(f"  📏 File size: {file_path.stat().st_size:,} bytes")
                    with open(file_path, 'rb') as f:
                        header = f.read(16)
                        if b'CDF' in header:
                            print("  📄 File appears to be NetCDF (classic format)")
                        elif b'HDF' in header or header.startswith(b'\x89HDF'):
                            print("  📄 File appears to be NetCDF4/HDF5 format")
                        else:
                            print(f"  ⚠️  Unexpected file format. Header: {header}")
            except Exception as detail_err:
                print(f"  ❌ Could not get diagnostic info: {detail_err}")

            return False

    def _download_monthly_chunk(self, variable: str, year: int, month: int,
                               client: cdsapi.Client, max_retries: int = 3) -> Path:
        """Download data for a single month to avoid size limits."""
        chunk_file = self._get_raw_file_path(variable, year).parent / f"chunk_{variable}_{year}_{month:02d}.nc"

        # European bounding box (North, West, South, East)
        area = [75, -25, 25, 45]

        for attempt in range(max_retries):
            try:
                print(f"  📥 Downloading {variable} for {year}-{month:02d} (attempt {attempt+1}/{max_retries})...")

                # Remove existing file if it exists
                if chunk_file.exists():
                    chunk_file.unlink()

                client.retrieve(
                    'reanalysis-era5-land',
                    {
                        'variable': self.variable_mapping[variable],
                        'year': str(year),
                        'month': f"{month:02d}",
                        'day': [f"{d:02d}" for d in range(1, 32)],  # All days in month
                        'time': [f"{h:02d}:00" for h in range(0, 24)],  # All 24 hours (FIXED)
                        'area': area,
                        'grid': [0.1, 0.1],  # 0.1° resolution
                        'format': 'netcdf',
                    },
                    str(chunk_file)
                )

                # Wait for file to be fully written
                time.sleep(2)

                # Check what type of file was downloaded
                if chunk_file.exists():
                    file_size = chunk_file.stat().st_size
                    print(f"  📁 Downloaded file: {chunk_file.name} ({file_size:,} bytes)")

                    # Check if it's a ZIP file
                    if zipfile.is_zipfile(chunk_file):
                        print(f"  📦 File is ZIP format - extracting...")
                        # Extract immediately instead of waiting for validation
                        actual_file = self._extract_zip_if_needed(chunk_file)
                        # Update chunk_file to point to the extracted file
                        chunk_file = actual_file
                        print(f"  ✅ Using extracted file: {chunk_file.name}")
                    else:
                        print(f"  📄 File is NetCDF format")

                # Validate the file (now should be NetCDF)
                if self._validate_netcdf_file(chunk_file):
                    print(f"  ✅ Successfully downloaded and validated {year}-{month:02d}")
                    return chunk_file
                else:
                    print(f"  ❌ Downloaded file failed validation (attempt {attempt+1})")
                    # Clean up the file
                    if chunk_file.exists():
                        chunk_file.unlink()

            except Exception as e:
                print(f"  ❌ Download attempt {attempt+1} failed: {e}")
                if chunk_file.exists():
                    chunk_file.unlink()

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Exponential backoff
                    print(f"  ⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        raise RuntimeError(f"Failed to download valid chunk for {variable} {year}-{month:02d} after {max_retries} attempts")

    def download_raw_data(self, variable: str, year: int, overwrite: bool = False) -> Path:
        """
        Download raw ERA5-Land data for a specific variable and year.
        Downloads in monthly chunks to avoid CDS size limits.

        Args:
            variable: Variable name (e.g., "total_precipitation")
            year: Year to download
            overwrite: Whether to overwrite existing files

        Returns:
            Path to downloaded raw file
        """
        if variable not in self.variable_mapping:
            raise ValueError(f"Unsupported variable: {variable}. Supported variables: {list(self.variable_mapping.keys())}")

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
                    print(f"  ⚠️  Failed to download {year}-{month:02d}: {e}")
                    continue

            # Merge monthly chunks into single file
            if chunk_files:
                print(f"🔗 Merging {len(chunk_files)} monthly chunks...")
                datasets = []

                for chunk_file in chunk_files:
                    if chunk_file.exists():
                        try:
                            # chunk_file might be the actual .nc file already (post-extraction)
                            ds = xr.open_dataset(chunk_file, engine='netcdf4')
                            datasets.append(ds)
                            print(f"  ✅ Loaded chunk: {chunk_file.name}")
                        except Exception as e:
                            print(f"  ⚠️  Failed to load chunk {chunk_file}: {e}")

                if datasets:
                    print(f"  🔗 Concatenating {len(datasets)} valid datasets...")

                    # Use the correct time dimension for concatenation
                    time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
                    merged_ds = xr.concat(datasets, dim=time_dim)
                    merged_ds = merged_ds.sortby(time_dim)

                    # Save with compression
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
                        # Clean up chunk files
                        for chunk_file in chunk_files:
                            if chunk_file.exists():
                                chunk_file.unlink()
                        print(f"  🧹 Cleaned up {len(chunk_files)} chunk files")
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
        # Handle longitude coordinate conversion if needed
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

        # Fallback to first available
        print(f"Warning: Could not find variable {expected_var}, using {list(ds.data_vars.keys())[0]}")
        return list(ds.data_vars.keys())[0]

    def process_raw_data(self, variable: str, year: int, freq: str = "daily",
                        raw_file: Optional[Path] = None) -> Path:
        """
        Process raw ERA5-Land data into the expected format.

        Args:
            variable: Variable name
            year: Year of data
            freq: Frequency ("daily", "weekly", "monthly")
            raw_file: Path to raw file (if None, will use default path)

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
        output_dir = self.processed_dir / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Processing {variable} for {year} at {freq} frequency...")

        # Extract ZIP if needed and validate
        actual_file = self._extract_zip_if_needed(raw_file)
        if not self._validate_netcdf_file(actual_file):
            raise ValueError(f"Raw file failed validation: {actual_file}")

        # Load with chunking for memory efficiency
        print(f"  📂 Loading data with chunking...")
        ds = xr.open_dataset(
            actual_file, 
            engine='netcdf4',
            chunks={'latitude': 100, 'longitude': 100}  # Larger chunks for better performance
        )
        
        internal_var = self.get_internal_variable_name(ds, variable)
        print(f"  ✓ Using internal variable: '{internal_var}'")

        # Subset to Europe
        print(f"  ✂️  Subsetting to European region...")
        ds_europe = self.subset_to_europe(ds)
        print(f"  ✓ Subset complete. Shape: {ds_europe[internal_var].shape}")

        if internal_var not in ds_europe:
            raise ValueError(f"Variable '{internal_var}' not found in European subset")

        # Determine correct time dimension for resampling
        time_dim = "valid_time" if "valid_time" in ds_europe.dims else "time"
        print(f"  🕐 Using time dimension: '{time_dim}' with {len(ds_europe[time_dim])} timesteps")

        # Process based on variable type
        print(f"  🔄 Resampling to {freq} frequency...")
        
        if variable in self.stats_variables:
            # Statistics processing (min/max/mean)
            print(f"  📊 Computing statistics (min/max/mean)...")
            
            # Create resampled object once
            resampled = ds_europe[internal_var].resample({time_dim: resample_str})
            
            # Compute statistics with progress tracking
            import warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
            
            print(f"    ⏱️  Computing min... (this may take a few minutes)")
            import time
            start = time.time()
            da_min = resampled.min().compute()
            print(f"    ✓ Min computed in {time.time() - start:.1f}s")
            
            print(f"    ⏱️  Computing max...")
            start = time.time()
            da_max = resampled.max().compute()
            print(f"    ✓ Max computed in {time.time() - start:.1f}s")
            
            print(f"    ⏱️  Computing mean...")
            start = time.time()
            da_mean = resampled.mean().compute()
            print(f"    ✓ Mean computed in {time.time() - start:.1f}s")
            
            ds_out = xr.Dataset({
                f"{variable}_min": da_min,
                f"{variable}_max": da_max,
                f"{variable}_mean": da_mean
            })
            outfile = output_dir / f"{variable}_{freq}_stats_{year}.nc"
        else:
            # Cumulative processing (sum)
            print(f"  📊 Computing cumulative sum...")
            import time
            start = time.time()
            da_sum = ds_europe[internal_var].resample({time_dim: resample_str}).sum().compute()
            print(f"  ✓ Sum computed in {time.time() - start:.1f}s")
            
            ds_out = xr.Dataset({
                f"{variable}_sum": da_sum
            })
            outfile = output_dir / f"{variable}_{freq}_cum_{year}.nc"

        # Standardize time dimension naming
        for var_name in ds_out.data_vars:
            if 'valid_time' in ds_out[var_name].dims:
                ds_out = ds_out.rename({'valid_time': 'time'})
                break

        if 'valid_time' in ds_out.coords:
            ds_out = ds_out.rename({'valid_time': 'time'})

        # Save processed data with compression
        print(f"  💾 Saving to {outfile.name}...")
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_out.data_vars}
        ds_out.to_netcdf(outfile, engine='netcdf4', encoding=encoding)

        # Close datasets to free memory
        ds.close()
        ds_out.close()

        print(f"✅ Processed data saved: {outfile}")
        print(f"  📏 Output file size: {outfile.stat().st_size / (1024*1024):.1f} MB")

        # Validate processed file
        if not self._validate_netcdf_file(outfile):
            raise RuntimeError(f"Processed file failed validation: {outfile}")

        return outfile

    def ensure_data_available(self, variable: str, year: int, freq: str = "daily",
                            force_redownload: bool = False, download_only: bool = False) -> Path:
        """
        Ensure that processed data is available for a variable and year.
        Downloads and processes if missing.

        Args:
            variable: Variable name
            year: Year of data
            freq: Frequency
            force_redownload: Whether to force redownload even if file exists
            download_only: If True, only download raw data without processing

        Returns:
            Path to processed file (or raw file if download_only=True)
        """
        expected_file = self._get_expected_file_path(variable, year, freq)
        raw_file = self._get_raw_file_path(variable, year)

        # Check if processed file exists
        if expected_file.exists() and not force_redownload:
            print(f"✅ Processed data already available: {expected_file.name}")
            return expected_file

        # Check if raw file exists
        if raw_file.exists() and not force_redownload:
            print(f"✅ Raw data already downloaded: {raw_file.name}")
            if download_only:
                return raw_file
            # Process existing raw data
            print(f"🔄 Processing existing raw data...")
            processed_file = self.process_raw_data(variable, year, freq, raw_file)
            return processed_file

        # Need to download
        print(f"📥 Data not found for {variable} {year}. Downloading...")
        raw_file = self.download_raw_data(variable, year, overwrite=force_redownload)

        if download_only:
            print(f"✅ Download complete (processing skipped): {raw_file.name}")
            return raw_file

        # Process the downloaded data
        print(f"🔄 Processing downloaded data...")
        processed_file = self.process_raw_data(variable, year, freq, raw_file)
        return processed_file

    def download_all_required_data(self, variables: List[str], years: List[int], 
                                   force_redownload: bool = False) -> Dict[str, Dict[int, Path]]:
        """
        Download ALL required raw data before processing.
        
        Args:
            variables: List of variable names
            years: List of years
            force_redownload: Whether to force redownload existing files
            
        Returns:
            Dictionary mapping {variable: {year: raw_file_path}}
        """
        print("\n" + "="*80)
        print("📥 PHASE 1: DOWNLOADING RAW DATA")
        print("="*80)
        
        downloaded_data = {}
        total_downloads = len(variables) * len(years)
        current = 0
        
        for variable in variables:
            downloaded_data[variable] = {}
            
            for year in years:
                current += 1
                print(f"\n[{current}/{total_downloads}] Checking {variable} for {year}...")
                
                try:
                    raw_file = self.ensure_data_available(
                        variable, year, 
                        download_only=True,
                        force_redownload=force_redownload
                    )
                    downloaded_data[variable][year] = raw_file
                    print(f"✅ [{current}/{total_downloads}] {variable} {year} ready")
                    
                except Exception as e:
                    print(f"❌ [{current}/{total_downloads}] Failed to download {variable} {year}: {e}")
                    downloaded_data[variable][year] = None
        
        # Summary of downloads
        print("\n" + "="*80)
        print("📊 DOWNLOAD PHASE SUMMARY")
        print("="*80)
        
        success_count = 0
        fail_count = 0
        
        for variable in variables:
            for year in years:
                if downloaded_data[variable][year] is not None:
                    success_count += 1
                    print(f"  ✅ {variable} {year}")
                else:
                    fail_count += 1
                    print(f"  ❌ {variable} {year}")
        
        print(f"\n✅ Successfully downloaded: {success_count}/{total_downloads}")
        if fail_count > 0:
            print(f"❌ Failed downloads: {fail_count}/{total_downloads}")
            response = input("\nContinue with processing available data? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("User cancelled due to download failures")
        
        return downloaded_data

    def process_all_downloaded_data(self, downloaded_data: Dict[str, Dict[int, Path]], 
                                   freq: str = "daily") -> Dict[str, Dict[int, Path]]:
        """
        Process all downloaded raw data into final format.
        
        Args:
            downloaded_data: Dictionary from download_all_required_data()
            freq: Frequency for processing
            
        Returns:
            Dictionary mapping {variable: {year: processed_file_path}}
        """
        print("\n" + "="*80)
        print("🔄 PHASE 2: PROCESSING DATA")
        print("="*80)
        
        processed_data = {}
        
        # Count total items to process
        total_items = sum(
            1 for var_data in downloaded_data.values() 
            for raw_file in var_data.values() 
            if raw_file is not None
        )
        current = 0
        
        for variable, year_data in downloaded_data.items():
            processed_data[variable] = {}
            
            for year, raw_file in year_data.items():
                if raw_file is None:
                    print(f"\n⏭️  Skipping {variable} {year} (no raw data)")
                    processed_data[variable][year] = None
                    continue
                
                current += 1
                print(f"\n[{current}/{total_items}] Processing {variable} for {year}...")
                print(f"  📁 Raw file: {raw_file.name} ({raw_file.stat().st_size / (1024**2):.1f} MB)")
                
                try:
                    processed_file = self.process_raw_data(variable, year, freq, raw_file)
                    processed_data[variable][year] = processed_file
                    print(f"✅ [{current}/{total_items}] {variable} {year} processed successfully")
                    
                except Exception as e:
                    print(f"❌ [{current}/{total_items}] Failed to process {variable} {year}: {e}")
                    processed_data[variable][year] = None
        
        # Summary of processing
        print("\n" + "="*80)
        print("📊 PROCESSING PHASE SUMMARY")
        print("="*80)
        
        success_count = 0
        fail_count = 0
        
        for variable, year_data in processed_data.items():
            for year, processed_file in year_data.items():
                if processed_file is not None:
                    success_count += 1
                    print(f"  ✅ {variable} {year}: {processed_file.name}")
                else:
                    fail_count += 1
                    print(f"  ❌ {variable} {year}")
        
        print(f"\n✅ Successfully processed: {success_count}/{total_items}")
        if fail_count > 0:
            print(f"❌ Failed processing: {fail_count}/{total_items}")
        
        return processed_data

    def ensure_multiple_years(self, variable: str, years: List[int], freq: str = "daily",
                            force_redownload: bool = False) -> List[Path]:
        """
        Ensure data is available for multiple years of a variable.
        Uses two-phase approach: download all, then process all.

        Args:
            variable: Variable name
            years: List of years
            freq: Frequency
            force_redownload: Whether to force redownload

        Returns:
            List of paths to processed files
        """
        print(f"\n🎯 Ensuring data for {variable} across {len(years)} years")
        
        # Phase 1: Download all
        downloaded_data = self.download_all_required_data([variable], years, force_redownload)
        
        # Phase 2: Process all
        processed_data = self.process_all_downloaded_data(downloaded_data, freq)
        
        # Extract results for this variable
        processed_files = [
            processed_data[variable][year] 
            for year in years 
            if processed_data[variable][year] is not None
        ]
        
        return processed_files


def download_missing_data(variables: List[str], years: List[int],
                         base_output_dir: str = "../input_data/copernicus_climate_data",
                         freq: str = "daily", force_redownload: bool = False) -> Dict[str, List[Path]]:
    """
    Convenience function to download and process data for multiple variables and years.
    Uses two-phase approach: downloads all data first, then processes.

    Args:
        variables: List of variable names
        years: List of years
        base_output_dir: Base output directory
        freq: Frequency ("daily", "weekly", "monthly")
        force_redownload: Whether to force redownload existing files

    Returns:
        Dictionary mapping variable names to lists of processed file paths
    """
    downloader = CopernicusDownloader(base_output_dir)
    
    print("="*80)
    print(f"🚀 STARTING DATA PIPELINE")
    print(f"📋 Variables: {', '.join(variables)}")
    print(f"📅 Years: {', '.join(map(str, years))}")
    print(f"📊 Frequency: {freq}")
    print("="*80)
    
    # Phase 1: Download ALL raw data
    downloaded_data = downloader.download_all_required_data(variables, years, force_redownload)
    
    # Phase 2: Process ALL downloaded data
    processed_data = downloader.process_all_downloaded_data(downloaded_data, freq)
    
    # Convert to the expected return format
    results = {}
    for variable in variables:
        results[variable] = [
            processed_data[variable][year]
            for year in years
            if processed_data[variable][year] is not None
        ]
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE")
    print("="*80)
    for variable, files in results.items():
        print(f"  {variable}: {len(files)}/{len(years)} years processed")
    
    return results


def debug_file_format(file_path: str):
    """Debug utility to check what format a downloaded file is in."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File does not exist: {path}")
        return

    file_size = path.stat().st_size
    print(f"📁 File: {path.name} ({file_size:,} bytes)")

    # Check if it's a ZIP file
    if zipfile.is_zipfile(path):
        print("📦 Format: ZIP archive")
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                files = zf.namelist()
                print(f"   Contents: {len(files)} files")
                for f in files[:5]:  # Show first 5 files
                    print(f"   - {f}")
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more files")
        except Exception as e:
            print(f"   ❌ Error reading ZIP: {e}")
    else:
        print("📄 Format: Direct file (likely NetCDF)")
        # Try to read as NetCDF
        try:
            import xarray as xr
            with xr.open_dataset(path) as ds:
                print(f"   ✅ NetCDF - Variables: {list(ds.data_vars.keys())}")
                print(f"   ✅ NetCDF - Dimensions: {dict(ds.dims)}")
        except Exception as e:
            print(f"   ❌ Not a valid NetCDF: {e}")
            # Check first few bytes to identify format
            with open(path, 'rb') as f:
                header = f.read(16)
                print(f"   🔍 File header: {header}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        if len(sys.argv) > 2:
            debug_file_format(sys.argv[2])
        else:
            print("Usage: python copernicus_downloader.py debug <file_path>")
    else:
        # Example usage for testing
        variables = [
            "total_precipitation",
            "2m_temperature",
            "10m_u_component_of_wind"
        ]
        years = [2020]

        print("🧪 Testing CopernicusDownloader...")

        try:
            downloader = CopernicusDownloader()
            results = downloader.ensure_multiple_years("total_precipitation", [2020])
            print(f"✅ Test successful: {len(results)} files processed")
        except Exception as e:
            print(f"❌ Test failed: {e}")