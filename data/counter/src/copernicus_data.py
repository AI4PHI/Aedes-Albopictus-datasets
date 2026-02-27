#!/usr/bin/env python3
"""
Albopictus Trap Data Processing with Copernicus Climate Data

This script processes albopictus trap data by pairing it with Copernicus climate data.
It loads trap data, filters it for the specified year, extracts climate variables
from NetCDF files, and creates both daily and monthly aggregated climate variables.

Features:
- Automatic filtering of trap data by date range
- Extraction of multiple climate variables (precipitation, temperature, wind, etc.)
- Both daily and monthly climate data aggregation
- Optional automatic downloading of missing climate data from Copernicus CDS

Usage:
    # Basic usage with default input file (../output_data/albopictus.pkl)
    python pair_traps_and_copernicus_data_polished.py

    # With custom input file
    python pair_traps_and_copernicus_data_polished.py --input-file path/to/data.pkl

    # With automatic downloads enabled
    python pair_traps_and_copernicus_data_polished.py --enable-downloads

Setup for automatic downloads:
1. Install cdsapi: pip install cdsapi
2. Register at https://cds.climate.copernicus.eu/api-how-to
3. Create ~/.cdsapirc with your API credentials:
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_API_KEY

Author: Generated from notebook pair_traps_and_copernicus_data.ipynb
"""

import os
import sys
import argparse
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import pandas as pd
import xarray as xr

# Import local climate processing functions
try:
    from process_copernicus_data import extract_climate_data_to_df, aggregate_to_monthly
except ImportError as e:
    logging.error(f"Failed to import process_copernicus_data: {e}")
    sys.exit(1)

# Try to import the downloader
try:
    from copernicus_downloader import CopernicusDownloader
    DOWNLOADER_AVAILABLE = True
    logging.info("Copernicus downloader available")
except ImportError as e:
    logging.warning("Copernicus downloader not available. Missing data will cause errors.")
    DOWNLOADER_AVAILABLE = False
    CopernicusDownloader = None


class TrapClimateProcessor:
    """Processes trap data and pairs it with climate data."""

    def __init__(self, config: Dict):
        """Initialize processor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize downloader if available and enabled
        self.downloader = None
        if config.get('enable_downloads', False) and DOWNLOADER_AVAILABLE:
            self.downloader = CopernicusDownloader(
                base_output_dir=config.get('climate_data_dir', '../input_data/copernicus_climate_data')
            )
            self.logger.info("Copernicus downloader initialized")
        elif config.get('enable_downloads', False) and not DOWNLOADER_AVAILABLE:
            self.logger.warning("Downloads enabled but downloader not available")

    def ensure_all_climate_data(self, variables: List[str], first_year: int, last_year: int):
        """
        Ensure all required climate data is downloaded and processed.
        
        Args:
            variables: List of variables to download
            first_year: First year needed
            last_year: Last year needed
        """
        if self.downloader is None:
            self.logger.warning("Downloader not available, skipping data check")
            return
        
        years = list(range(first_year, last_year + 1))
        
        self.logger.info(f"Checking climate data for {len(variables)} variables, {len(years)} years")
        
        try:
            # Use the two-phase download approach
            from copernicus_downloader import download_missing_data
            results = download_missing_data(
                variables=variables,
                years=years,
                base_output_dir=self.config.get('climate_data_dir', '../input_data/copernicus_climate_data'),
                freq=self.config.get('freq', 'daily'),
                force_redownload=False
            )
            
            # Check results
            for variable, files in results.items():
                if len(files) < len(years):
                    self.logger.warning(
                        f"Variable {variable}: only {len(files)}/{len(years)} years available"
                    )
                else:
                    self.logger.info(f"Variable {variable}: all {len(files)} years available")
                    
        except Exception as e:
            self.logger.error(f"Failed to ensure climate data: {e}")
            raise

    def load_and_filter_trap_data(self, input_file: str) -> pd.DataFrame:
        """
        Load trap data from pickle file and apply temporal filtering.

        Args:
            input_file: Path to the input pickle file

        Returns:
            Filtered DataFrame with trap data
        """
        self.logger.info(f"Loading trap data from {input_file}")

        try:
            df = pd.read_pickle(input_file)
            self.logger.info(f"Loaded {len(df)} records")
        except Exception as e:
            self.logger.error(f"Failed to load data from {input_file}: {e}")
            raise

        # Apply date filtering
        start_date = pd.Timestamp(self.config['filter_start_date'])
        end_date = pd.Timestamp(self.config['filter_end_date'])

        self.logger.info(f"Filtering data between {start_date} and {end_date}")
        initial_count = len(df)

        df = df[df["end_date"] < end_date]
        df = df[df["end_date"] >= start_date]

        filtered_count = len(df)
        self.logger.info(f"Filtered from {initial_count} to {filtered_count} records")

        # Ensure date columns are datetime
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])

        return df

    def determine_required_years(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Determine the range of years needed for climate data based on trap data dates.

        Args:
            df: Trap data DataFrame

        Returns:
            Tuple of (first_year_needed, last_year_needed)
        """
        time_col = self.config['time_col']
        time_window_avg = self.config['time_window_avg']

        df[time_col] = pd.to_datetime(df[time_col])

        first_date_needed = df[time_col].min() - pd.Timedelta(time_window_avg)
        last_date_needed = df[time_col].max()

        first_year = first_date_needed.year
        last_year = last_date_needed.year

        self.logger.info(f"Climate data required from {first_year} to {last_year}")
        return first_year, last_year

    def get_climate_file_path(self, var: str, year: int) -> str:
        """
        Generate the appropriate file path for a climate variable and year.

        Args:
            var: Climate variable name
            year: Year for the data

        Returns:
            Path to the NetCDF file in format: {path_dir}{year}/{var}_{freq}_{suffix}_{year}.nc
        """
        freq = self.config['freq']
        path_dir = self.config['path_dir']

        # Variables that use 'stats' suffix
        stats_vars = {
            "10m_u_component_of_wind", "10m_v_component_of_wind",
            "2m_dewpoint_temperature", "2m_temperature",
            "surface_net_thermal_radiation", "surface_net_solar_radiation",
            "surface_pressure", "skin_temperature",
            "surface_sensible_heat_flux", "surface_latent_heat_flux",
            "surface_thermal_radiation_downwards",
            "volumetric_soil_water_layer_1"
        }

        if var in stats_vars:
            suffix = "stats"
        else:
            suffix = "cum"

        # path_dir already ends with "/" from config
        return f"{path_dir}{year}/{var}_{freq}_{suffix}_{year}.nc"

    def load_climate_datasets(self, var: str, first_year: int, last_year: int) -> Optional[xr.Dataset]:
        """
        Load and merge climate datasets for a variable across multiple years.

        Args:
            var: Climate variable name
            first_year: First year to load
            last_year: Last year to load

        Returns:
            Merged xarray Dataset or None if no data available
        """
        datasets = []

        for year in range(first_year, last_year + 1):
            input_file = self.get_climate_file_path(var, year)

            if os.path.exists(input_file):
                self.logger.info(f"Loading dataset: {input_file}")
                try:
                    ds = xr.open_dataset(input_file)
                    datasets.append(ds)
                except Exception as e:
                    self.logger.warning(f"Failed to load {input_file}: {e}")
            else:
                self.logger.warning(f"File does not exist: {input_file}")

                # Try to download missing data if downloader is available
                if self.downloader is not None:
                    max_retries = 3
                    retry_delay = 5  # seconds

                    for attempt in range(max_retries):
                        try:
                            self.logger.info(f"Attempting to download {var} for {year} (attempt {attempt + 1}/{max_retries})")
                            processed_file = self.downloader.ensure_data_available(
                                var, year, self.config['freq'],
                                force_redownload=self.config.get('force_redownload', False)
                            )

                            # Try to load the newly downloaded file
                            if processed_file.exists():
                                self.logger.info(f"Loading downloaded dataset: {processed_file}")
                                ds = xr.open_dataset(str(processed_file))
                                datasets.append(ds)
                                break  # Success - exit retry loop
                            else:
                                self.logger.error(f"Downloaded file not found: {processed_file}")
                                if attempt < max_retries - 1:
                                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                                    import time
                                    time.sleep(retry_delay)

                        except Exception as e:
                            self.logger.error(f"Download attempt {attempt + 1} failed for {var} {year}: {e}")
                            if attempt < max_retries - 1:
                                self.logger.info(f"Retrying in {retry_delay} seconds...")
                                import time
                                time.sleep(retry_delay)
                            else:
                                self.logger.error(f"All download attempts failed for {var} {year}")
                                continue
                else:
                    if self.config.get('enable_downloads', False):
                        self.logger.warning("Downloads enabled but downloader not available - install cdsapi and configure API key")
                    else:
                        self.logger.info("Downloads disabled - skipping missing file")

        if not datasets:
            self.logger.warning(f"No datasets available for {var}")
            return None

        try:
            ds_merged = xr.concat(datasets, dim="time")
            self.logger.info(f"Merged {len(datasets)} datasets for {var}")
            return ds_merged
        except Exception as e:
            self.logger.error(f"Failed to merge datasets for {var}: {e}")
            return None

    def process_climate_variable(self, df: pd.DataFrame, ds_merged: xr.Dataset,
                                climate_var: str) -> pd.DataFrame:
        """
        Process a single climate variable, extracting both daily and monthly data.

        Args:
            df: Trap data DataFrame
            ds_merged: Merged climate dataset
            climate_var: Name of the climate variable to process

        Returns:
            DataFrame with added climate columns
        """
        lat_col = self.config['lat_col']
        lon_col = self.config['lon_col']
        time_col = self.config['time_col']
        time_window = self.config['time_window']
        time_window_avg = self.config['time_window_avg']
        months_to_average = self.config['months_to_average']

        self.logger.info(f"Processing {climate_var} - monthly average")

        # Extract monthly averaged data
        df = extract_climate_data_to_df(
            df, ds_merged, climate_var,
            lat_col=lat_col, lon_col=lon_col, time_col=time_col,
            time_window=time_window_avg
        )

        # Create monthly aggregated column
        df[f"{climate_var}_monthly"] = df[climate_var].apply(
            lambda x: aggregate_to_monthly(x, num_months=months_to_average)
        )

        self.logger.info(f"Processing {climate_var} - daily data")

        # Extract daily data
        df = extract_climate_data_to_df(
            df, ds_merged, climate_var,
            lat_col=lat_col, lon_col=lon_col, time_col=time_col,
            time_window=time_window
        )

        return df

    def process_all_climate_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all climate variables for the trap data.

        Args:
            df: Trap data DataFrame

        Returns:
            DataFrame with all climate variables added
        """
        input_vars = self.config['input_vars']
        first_year, last_year = self.determine_required_years(df)

        for var in input_vars:
            self.logger.info(f"Processing base variable: {var}")

            # Load climate datasets for this variable
            ds_merged = self.load_climate_datasets(var, first_year, last_year)

            if ds_merged is None:
                self.logger.warning(f"Skipping {var} - no data available")
                continue

            # Get all available variables in this dataset
            available_vars = list(ds_merged.data_vars.keys())
            self.logger.info(f"Variables available in dataset: {available_vars}")

            # Process each climate variable in the dataset
            for climate_var in available_vars:
                try:
                    df = self.process_climate_variable(df, ds_merged, climate_var)
                except Exception as e:
                    self.logger.error(f"Failed to process {climate_var}: {e}")
                    continue

            # Close dataset to free memory
            ds_merged.close()

        return df

    def save_results(self, df: pd.DataFrame, output_prefix: str):
        """
        Save the processed data to compressed CSV (ZIP) and pickle formats.

        The CSV is ZIP-compressed with full float precision (%.10g) and
        ISO 8601 date formatting, suitable for archival and database publication.

        Args:
            df: Processed DataFrame
            output_prefix: Prefix for output filenames
        """
        csv_file = f"{output_prefix}.csv.zip"
        pkl_file = f"{output_prefix}.pkl"

        self.logger.info(f"Saving results to {csv_file} and {pkl_file}")

        try:
            # Create a copy to avoid mutating the input DataFrame
            export_df = df.copy()

            # Cast object-typed boolean columns to appropriate types
            for col in ("keep", "climate_nan"):
                if col in export_df.columns:
                    if export_df[col].dtype == object:
                        # 'keep' is boolean-like, 'climate_nan' is string "yes"/"no"
                        if col == "keep":
                            export_df[col] = export_df[col].astype(bool)

            # Derive archive-internal filename
            archive_name = os.path.basename(f"{output_prefix}.csv")

            export_df.to_csv(
                csv_file,
                index=False,
                compression={"method": "zip", "archive_name": archive_name},
                float_format="%.10g",
                date_format="%Y-%m-%d",
            )
            self.logger.info(f"Saved compressed CSV to {csv_file}")

            df.to_pickle(pkl_file)
            self.logger.info(f"Successfully saved {len(df)} records")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

    def run(self, input_file: str, output_prefix: str):
        """
        Run the complete processing pipeline.

        Args:
            input_file: Path to input pickle file
            output_prefix: Prefix for output files
        """
        self.logger.info("Starting trap-climate data processing pipeline")

        # Load and filter trap data
        df = self.load_and_filter_trap_data(input_file)

        # Determine required years
        first_year, last_year = self.determine_required_years(df)
        
        # Ensure all climate data is available (if downloader enabled)
        if self.config.get('enable_downloads', False):
            self.logger.info("Ensuring all climate data is available...")
            self.ensure_all_climate_data(
                self.config['input_vars'],
                first_year,
                last_year
            )

        # Process climate variables
        df = self.process_all_climate_variables(df)

        # Save results
        self.save_results(df, output_prefix)

        self.logger.info("Processing pipeline completed successfully")


def create_default_config() -> Dict:
    """Create default configuration dictionary."""
    # --- Path configuration ---
    # Get the absolute path of the directory containing this script
    script_dir = Path(__file__).parent.resolve()
    # Get the parent directory of the script's directory (e.g., 'counter/')
    base_dir = script_dir.parent

    return {
        # Date filtering
        'filter_start_date': "2020-01-01",
        'filter_end_date': "2021-01-01",

        # Climate data configuration
        'input_vars': [
            "total_precipitation",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "volumetric_soil_water_layer_1"
        ],

        # Data extraction parameters
        'lat_col': "decimalLatitude",
        'lon_col': "decimalLongitude",
        'time_col': "end_date",
        'time_window': "89D",
        'months_to_average': 3,
        'days_per_month': 30,

        # File system paths (relative to script location)
        'freq': "daily",
        'climate_data_dir': str(base_dir / "input_data" / "climate"),
        'path_dir': str(base_dir / "input_data" / "climate" / "processed" / "europe" / "daily") + "/",
    }


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process albopictus trap data with Copernicus climate data"
    )
    parser.add_argument(
        '--input-file',
        help="Path to input pickle file containing trap data. If not provided, defaults to 'output_data/albopictus.pkl' relative to the script's parent directory."
    )
    parser.add_argument(
        '-o', '--output',
        default="./output_data/AIMSurv_albopictus_2020_era5_land",
        help="Output filename prefix (default: ./output_data/AIMSurv_albopictus_2020_era5_land)"
    )
    parser.add_argument(
        '--climate-path',
        help="Path to climate data directory (overrides default). This should point to the base climate directory containing 'raw/' and 'processed/' subdirectories."
    )
    parser.add_argument(
        '--start-date',
        default="2020-01-01",
        help="Start date for filtering (YYYY-MM-DD format)"
    )
    parser.add_argument(
        '--end-date',
        default="2021-01-01",
        help="End date for filtering (YYYY-MM-DD format)"
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set logging level"
    )
    parser.add_argument(
        '--enable-downloads',
        action='store_true',
        help="Enable automatic downloading of missing climate data from Copernicus CDS"
    )
    parser.add_argument(
        '--force-redownload',
        action='store_true',
        help="Force redownload of existing climate data files"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create configuration
    config = create_default_config()

    # --- Handle input file path ---
    # Default input file path relative to the script's parent directory
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent
    default_input_file = base_dir / "output_data" / "albopictus.pkl"
    
    input_file = Path(args.input_file) if args.input_file else default_input_file

    # Override configuration with command line arguments
    if args.climate_path:
        # User provided a custom climate data base directory
        config['climate_data_dir'] = args.climate_path
        config['path_dir'] = str(Path(args.climate_path) / "processed" / "europe" / "daily") + "/"
    if args.start_date:
        config['filter_start_date'] = args.start_date
    if args.end_date:
        config['filter_end_date'] = args.end_date

    # Add download configuration
    config['enable_downloads'] = args.enable_downloads
    config['force_redownload'] = args.force_redownload

    # Add computed configuration
    config['time_window_avg'] = f"{config['days_per_month'] * config['months_to_average'] - 1}D"

    # Validate input file
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    # Process data
    try:
        processor = TrapClimateProcessor(config)
        processor.run(str(input_file), args.output)

        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()