"""
Conservative ERA5-Land downloader for very restrictive CDS limits.
Downloads in small daily or weekly chunks if monthly chunks are still too large.
"""

from era5_land_downloader import ERA5LandDownloader
import cdsapi
import xarray as xr
from pathlib import Path
from typing import List
import calendar


class ConservativeERA5Downloader(ERA5LandDownloader):
    """
    Ultra-conservative downloader that uses very small chunks to avoid CDS limits.
    """

    def _download_weekly_chunk(self, variable: str, year: int, start_day: int, end_day: int, client: cdsapi.Client) -> Path:
        """Download data for a single week."""
        chunk_file = self._get_raw_file_path(variable, year).parent / f"chunk_{variable}_{year}_days_{start_day:03d}_{end_day:03d}.nc"

        # European bounding box
        area = [75, -25, 25, 45]

        # Convert day of year to month/day
        import datetime
        start_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=start_day-1)
        end_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=end_day-1)

        months = []
        days = []

        current_date = start_date
        while current_date <= end_date:
            month_str = f"{current_date.month:02d}"
            day_str = f"{current_date.day:02d}"

            if month_str not in months:
                months.append(month_str)
            days.append(day_str)

            current_date += datetime.timedelta(days=1)

        print(f"  📥 Downloading {variable} for {year} days {start_day}-{end_day}...")

        client.retrieve(
            'reanalysis-era5-land',
            {
                'variable': self.era5_variable_mapping[variable],
                'year': str(year),
                'month': months,
                'day': sorted(list(set(days))),
                'time': ['00:00', '12:00'],  # Only 2 times per day
                'area': area,
                'grid': [0.2, 0.2],  # Coarser resolution to reduce size
                'format': 'netcdf',
            },
            str(chunk_file)
        )
        return chunk_file

    def download_era5_land_data_conservative(self, variable: str, year: int, overwrite: bool = False) -> Path:
        """
        Very conservative download using weekly chunks.
        """
        if variable not in self.era5_variable_mapping:
            raise ValueError(f"Unsupported variable: {variable}")

        raw_file = self._get_raw_file_path(variable, year)

        if raw_file.exists() and not overwrite:
            print(f"✅ Raw file already exists: {raw_file}")
            return raw_file

        print(f"🔄 Downloading ERA5-Land {variable} for {year} in weekly chunks (conservative mode)...")

        client = self._get_cds_client()
        chunk_files = []

        try:
            # Download in weekly chunks (52 weeks per year)
            days_in_year = 366 if calendar.isleap(year) else 365

            for start_day in range(1, days_in_year + 1, 7):  # Weekly chunks
                end_day = min(start_day + 6, days_in_year)

                try:
                    chunk_file = self._download_weekly_chunk(variable, year, start_day, end_day, client)
                    chunk_files.append(chunk_file)
                    print(f"  ✅ Downloaded week {start_day}-{end_day}")
                except Exception as e:
                    print(f"  ⚠️ Failed to download week {start_day}-{end_day}: {e}")
                    # Continue with other weeks

            # Merge weekly chunks
            if chunk_files:
                print(f"🔗 Merging {len(chunk_files)} weekly chunks...")
                datasets = []
                for chunk_file in chunk_files:
                    if chunk_file.exists():
                        try:
                            ds = xr.open_dataset(chunk_file)
                            datasets.append(ds)
                        except Exception as e:
                            print(f"  ⚠️ Failed to load chunk {chunk_file}: {e}")

                if datasets:
                    merged_ds = xr.concat(datasets, dim="time")
                    merged_ds.to_netcdf(raw_file)
                    print(f"✅ Merged and saved: {raw_file}")

                    # Clean up chunk files
                    for chunk_file in chunk_files:
                        if chunk_file.exists():
                            chunk_file.unlink()
                else:
                    raise RuntimeError("No valid chunks processed")
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
            raise RuntimeError(f"Conservative download failed for {variable} {year}: {e}")


def download_with_fallback(variable: str, year: int, base_dir: str = "./copernicus_climate_data") -> Path:
    """
    Try different download strategies with fallback to more conservative approaches.
    """
    downloader = ERA5LandDownloader(base_dir)
    conservative_downloader = ConservativeERA5Downloader(base_dir)

    strategies = [
        ("Monthly chunks", downloader.download_era5_land_data),
        ("Weekly chunks (conservative)", conservative_downloader.download_era5_land_data_conservative),
    ]

    for strategy_name, download_func in strategies:
        try:
            print(f"🎯 Trying {strategy_name}...")
            return download_func(variable, year)
        except Exception as e:
            print(f"❌ {strategy_name} failed: {e}")
            print(f"⏭️  Trying next strategy...")

    raise RuntimeError(f"All download strategies failed for {variable} {year}")