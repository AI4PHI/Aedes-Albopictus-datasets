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

    def download_monthly_climatology(self, variable: str, year_start: int, year_end: int, 
                                    out_dir: str = "./data/inputs/") -> Path:
        """
        Download ERA5-Land monthly averaged climatology (simple, like CORDEX).
        
        Args:
            variable: '2m_temperature' or 'total_precipitation'
            year_start: Start year
            year_end: End year
            out_dir: Output directory
            
        Returns:
            Path to downloaded NetCDF file with monthly climatology
        """
        output_dir = Path(out_dir) / "era5_land"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{variable}_{year_start}_{year_end}_monthly.nc"
        
        if output_file.exists():
            print(f"  ✅ Using cached file: {output_file.name}")
            return output_file
        
        print(f"📥 Downloading ERA5-Land {variable} monthly data ({year_start}-{year_end})...")
        
        # Map variable names
        cds_variable_map = {
            "2m_temperature": "2m_temperature",
            "total_precipitation": "total_precipitation"
        }
        
        if variable not in cds_variable_map:
            raise ValueError(f"Unsupported variable: {variable}. Use '2m_temperature' or 'total_precipitation'")
        
        cds_variable = cds_variable_map[variable]
        
        # European bounding box
        area = [72, -25, 33, 45]  # North, West, South, East
        
        # Build request for monthly averaged reanalysis
        request = {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': cds_variable,
            'year': [str(y) for y in range(year_start, year_end + 1)],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'time': '00:00',
            'area': area,
            'format': 'netcdf',
        }
        
        client = self._get_cds_client()
        
        try:
            # Download directly - no chunks needed for monthly data
            client.retrieve('reanalysis-era5-land-monthly-means', request, str(output_file))
            print(f"  ✅ Downloaded: {output_file.name}")
            
            # Validate
            if not self._validate_netcdf_file(output_file):
                raise ValueError(f"Downloaded file failed validation: {output_file}")
            
            return output_file
            
        except Exception as e:
            if output_file.exists():
                output_file.unlink()
            raise RuntimeError(f"Failed to download {variable} climatology: {e}")

    def load_era5_monthly_climatology(self, datasets_by_var: Dict[str, xr.Dataset]) -> xr.Dataset:
        """
        Process ERA5-Land monthly climatology datasets.

        Args:
            datasets_by_var: Dictionary of datasets by variable name.

        Returns:
            Processed xarray Dataset with climatology.
        """
        # Process precipitation
        ds_precip = datasets_by_var["total_precipitation"]
        precip_var_name = "tp"

        # ERA5-Land monthly means: precipitation is MEAN RATE in m/day
        # Compute climatology (average across years for each month)
        p_month = ds_precip[precip_var_name].groupby("time.month").mean(dim="time")

        # Convert from m/day to mm/month:
        # 1. m/day → mm/day: × 1000
        # 2. mm/day → mm/month: × days_in_month
        days_per_month = xr.DataArray([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], 
                                       dims=['month'], 
                                       coords={'month': range(1, 13)})

        p_month = p_month * 1000.0 * days_per_month  # m/day → mm/month
        p_month.attrs["units"] = "mm/month"

        # Additional processing can be added here for other variables

        return p_month