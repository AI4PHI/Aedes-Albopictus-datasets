#!/usr/bin/env python3
"""
Copernicus Climate Data Store (CDS) downloader for ERA5-Land data.
Automatically downloads and processes missing climate data files.
"""

import os
import cdsapi
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import time


class CopernicusDownloader:
    """
    Downloads and processes ERA5-Land data from Copernicus Climate Data Store.
    """

    def __init__(self, base_output_dir: str = "/eos/jeodpp/data/projects/ETOHA/DATA/ClimateData/Copernicus_data"):
        self.base_output_dir = Path(base_output_dir)
        self.client = None

        # Variable name mapping from expected names to CDS API names
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

        # Variables that use stats processing (min/max/mean)
        self.stats_variables = {
            "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature",
            "2m_temperature", "surface_net_thermal_radiation", "surface_net_solar_radiation",
            "surface_pressure", "skin_temperature", "surface_sensible_heat_flux",
            "surface_latent_heat_flux", "surface_thermal_radiation_downwards",
            "volumetric_soil_water_layer_1"
        }

        # Variable name mapping from CDS to internal names
        self.internal_variable_mapping = {
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "2m_temperature": "t2m",
            "2m_dewpoint_temperature": "d2m",
            "volumetric_soil_water_layer_1": "swvl1",
            "total_precipitation": "tp"
        }

    def _get_cds_client(self) -> cdsapi.Client:
        """Initialize CDS API client if not already done."""
        if self.client is None:
            try:
                self.client = cdsapi.Client()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CDS API client. Make sure you have configured your CDS API key. Error: {e}")
        return self.client

    def _get_expected_file_path(self, variable: str, year: int, freq: str = "daily") -> Path:
        """Get the expected file path for a processed variable."""
        year_dir = self.base_output_dir / "europe" / "data" / str(year)

        if variable in self.stats_variables:
            filename = f"{variable}_{freq}_stats_{year}.nc"
        else:
            filename = f"{variable}_{freq}_cum_{year}.nc"

        return year_dir / filename

    def _get_raw_file_path(self, variable: str, year: int) -> Path:
        """Get the raw downloaded file path."""
        raw_dir = self.base_output_dir / "raw" / str(year)
        raw_dir.mkdir(parents=True, exist_ok=True)
        return raw_dir / f"cds_era5_land_{variable}_{year}.nc"

    def download_raw_data(self, variable: str, year: int, overwrite: bool = False) -> Path:
        """
        Download raw ERA5-Land data from CDS for a specific variable and year.

        Args:
            variable: Variable name (e.g., "total_precipitation")
            year: Year to download
            overwrite: Whether to overwrite existing files

        Returns:
            Path to downloaded file
        """
        if variable not in self.variable_mapping:
            raise ValueError(f"Unsupported variable: {variable}. Supported variables: {list(self.variable_mapping.keys())}")

        raw_file = self._get_raw_file_path(variable, year)

        if raw_file.exists() and not overwrite:
            print(f"✅ Raw file already exists: {raw_file}")
            return raw_file

        print(f"🔄 Downloading {variable} for {year} from Copernicus CDS...")

        client = self._get_cds_client()
        cds_variable = self.variable_mapping[variable]

        # European bounding box
        area = [75, -25, 25, 45]  # North, West, South, East

        # Generate all months for the year
        months = [f"{m:02d}" for m in range(1, 13)]

        try:
            client.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': cds_variable,
                    'year': str(year),
                    'month': months,
                    'day': [f"{d:02d}" for d in range(1, 32)],  # All days
                    'time': [f"{h:02d}:00" for h in range(0, 24)],  # All hours
                    'area': area,
                    'grid': [0.1, 0.1],  # 0.1° x 0.1° resolution
                    'format': 'netcdf',
                },
                str(raw_file)
            )
            print(f"✅ Successfully downloaded: {raw_file}")
            return raw_file

        except Exception as e:
            if raw_file.exists():
                raw_file.unlink()  # Remove incomplete file
            raise RuntimeError(f"Failed to download {variable} for {year}: {e}")

    def subset_to_europe(self, ds: xr.Dataset, lat_min: float = 33, lat_max: float = 72,
                        lon_min: float = -25, lon_max: float = 45) -> xr.Dataset:
        """Subset dataset to European bounds."""
        # Handle longitude coordinate conversion if needed
        if 'longitude' in ds.coords:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds = ds.sortby("longitude")

        # Select European region
        ds = ds.sel(longitude=slice(lon_min, lon_max))

        if ds.latitude[0] > ds.latitude[-1]:
            ds = ds.sel(latitude=slice(lat_max, lat_min))
        else:
            ds = ds.sel(latitude=slice(lat_min, lat_max))

        return ds

    def get_internal_variable(self, ds: xr.Dataset, expected_var: str) -> str:
        """Get the internal variable name from the dataset."""
        if expected_var in ds.data_vars:
            return expected_var
        if len(ds.data_vars) == 1:
            return list(ds.data_vars.keys())[0]
        if expected_var in self.internal_variable_mapping:
            internal_var = self.internal_variable_mapping[expected_var]
            if internal_var in ds.data_vars:
                return internal_var

        print(f"Warning: Could not determine internal variable name for '{expected_var}'. Using '{list(ds.data_vars.keys())[0]}'")
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

        # Determine resampling frequency
        if freq == "daily":
            resample_str = "1D"
        elif freq == "weekly":
            resample_str = "1W"
        elif freq == "monthly":
            resample_str = "1M"
        else:
            raise ValueError("Invalid frequency: choose daily, weekly or monthly")

        # Set up output paths
        output_dir = self.base_output_dir / "europe" / "data" / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Processing {variable} for {year} at {freq} frequency...")

        # Load and process data
        ds = xr.open_dataset(raw_file)
        internal_var = self.get_internal_variable(ds, variable)
        print(f"  Found internal variable: '{internal_var}'")

        # Subset to Europe
        ds_europe = self.subset_to_europe(ds)

        if internal_var not in ds_europe:
            raise ValueError(f"Internal variable '{internal_var}' not found in European subset for '{variable}'")

        # Determine correct time dimension for resampling
        time_dim_for_resample = "valid_time" if "valid_time" in ds_europe.dims else "time"

        # Process based on variable type
        if variable in self.stats_variables:
            # Statistics processing (min/max/mean)
            da = ds_europe[internal_var].resample({time_dim_for_resample: resample_str})
            agg_min = da.min()
            agg_max = da.max()
            agg_mean = da.mean()

            ds_out = xr.Dataset({
                f"{variable}_min": agg_min,
                f"{variable}_max": agg_max,
                f"{variable}_mean": agg_mean
            })
            outfile = output_dir / f"{variable}_{freq}_stats_{year}.nc"
        else:
            # Cumulative processing (sum)
            agg_sum = ds_europe[internal_var].resample({time_dim_for_resample: resample_str}).sum()
            ds_out = xr.Dataset({f"{variable}_sum": agg_sum})
            outfile = output_dir / f"{variable}_{freq}_cum_{year}.nc"

        # Ensure consistent time dimension naming - rename to 'time' for compatibility
        for var_name in ds_out.data_vars:
            if 'valid_time' in ds_out[var_name].dims:
                ds_out = ds_out.rename({'valid_time': 'time'})
                break

        # Ensure coordinate names are correct
        if 'valid_time' in ds_out.coords:
            ds_out = ds_out.rename({'valid_time': 'time'})

        # Save processed data
        ds_out.to_netcdf(outfile)
        print(f"✅ Processed data saved: {outfile}")

        # Clean up raw file to save space (optional)
        # raw_file.unlink()

        return outfile

    def ensure_data_available(self, variable: str, year: int, freq: str = "daily",
                            force_redownload: bool = False) -> Path:
        """
        Ensure that processed data is available for a variable and year.
        Downloads and processes if missing.

        Args:
            variable: Variable name
            year: Year of data
            freq: Frequency
            force_redownload: Whether to force redownload even if file exists

        Returns:
            Path to processed file
        """
        expected_file = self._get_expected_file_path(variable, year, freq)

        if expected_file.exists() and not force_redownload:
            print(f"✅ Data already available: {expected_file}")
            return expected_file

        print(f"📥 Data not found for {variable} {year}. Downloading and processing...")

        # Download raw data
        raw_file = self.download_raw_data(variable, year, overwrite=force_redownload)

        # Process raw data
        processed_file = self.process_raw_data(variable, year, freq, raw_file)

        return processed_file

    def ensure_multiple_years(self, variable: str, years: List[int], freq: str = "daily",
                            force_redownload: bool = False) -> List[Path]:
        """
        Ensure data is available for multiple years of a variable.

        Args:
            variable: Variable name
            years: List of years
            freq: Frequency
            force_redownload: Whether to force redownload

        Returns:
            List of paths to processed files
        """
        processed_files = []

        for year in years:
            try:
                processed_file = self.ensure_data_available(variable, year, freq, force_redownload)
                processed_files.append(processed_file)
            except Exception as e:
                print(f"❌ Failed to ensure data for {variable} {year}: {e}")
                # Continue with other years

        return processed_files


def download_missing_data(variables: List[str], years: List[int],
                         base_output_dir: str = "/eos/jeodpp/data/projects/ETOHA/DATA/ClimateData/Copernicus_data",
                         freq: str = "daily", force_redownload: bool = False) -> Dict[str, List[Path]]:
    """
    Convenience function to download missing data for multiple variables and years.

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
    results = {}

    for variable in variables:
        print(f"\n🎯 Processing variable: {variable}")
        try:
            processed_files = downloader.ensure_multiple_years(variable, years, freq, force_redownload)
            results[variable] = processed_files
            print(f"✅ Completed {variable}: {len(processed_files)} files processed")
        except Exception as e:
            print(f"❌ Failed to process {variable}: {e}")
            results[variable] = []

    return results


if __name__ == "__main__":
    # Example usage
    variables = [
        "total_precipitation",
        "2m_temperature",
        "10m_u_component_of_wind",
        "volumetric_soil_water_layer_1"
    ]
    years = [2019, 2020, 2021]

    results = download_missing_data(variables, years)

    print("\n📊 Download Summary:")
    for var, files in results.items():
        print(f"  {var}: {len(files)} files")