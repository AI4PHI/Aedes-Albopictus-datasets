#!/usr/bin/env python3
"""
Debug script to understand why validation is failing after extraction.
"""

import sys
import os
from pathlib import Path
import zipfile
import xarray as xr


def debug_chunk_file(file_path_str):
    """Debug a specific chunk file to understand validation failure."""
    file_path = Path(file_path_str)

    print(f"🔍 Debugging file: {file_path}")
    print(f"   Exists: {file_path.exists()}")

    if not file_path.exists():
        print("❌ File doesn't exist!")
        return

    print(f"   Size: {file_path.stat().st_size:,} bytes")

    # Check if it's a ZIP file
    is_zip = zipfile.is_zipfile(file_path)
    print(f"   Is ZIP: {is_zip}")

    if is_zip:
        print("   📦 ZIP file detected - examining contents...")
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                contents = zf.namelist()
                print(f"   ZIP contents: {contents}")

                # Try to extract and validate
                extract_dir = file_path.parent / "debug_extract"
                extract_dir.mkdir(exist_ok=True)

                zf.extractall(extract_dir)

                # Find NetCDF files
                nc_files = list(extract_dir.glob("*.nc"))
                print(f"   Extracted NetCDF files: {[f.name for f in nc_files]}")

                if nc_files:
                    extracted_nc = nc_files[0]
                    print(f"   Testing extracted file: {extracted_nc}")
                    print(f"   Extracted size: {extracted_nc.stat().st_size:,} bytes")

                    # Try to open with xarray
                    try:
                        with xr.open_dataset(extracted_nc, engine='netcdf4') as ds:
                            print("   ✅ Successfully opened with xarray!")
                            print(f"   Variables: {list(ds.data_vars.keys())}")
                            print(f"   Dimensions: {dict(ds.dims)}")

                            # Try to access data
                            for var in list(ds.data_vars.keys())[:1]:  # Just first variable
                                print(f"   Testing variable '{var}' access...")
                                _ = ds[var].dims
                                print(f"   Variable '{var}' dims: {ds[var].dims}")

                        print("   ✅ All validation checks passed!")

                    except Exception as e:
                        print(f"   ❌ xarray failed: {e}")

                # Cleanup
                import shutil
                shutil.rmtree(extract_dir)

        except Exception as e:
            print(f"   ❌ ZIP processing failed: {e}")
    else:
        # Try to open as NetCDF directly
        print("   📄 NetCDF file detected - testing...")
        try:
            with xr.open_dataset(file_path, engine='netcdf4') as ds:
                print("   ✅ Successfully opened with xarray!")
                print(f"   Variables: {list(ds.data_vars.keys())}")
                print(f"   Dimensions: {dict(ds.dims)}")
        except Exception as e:
            print(f"   ❌ xarray failed: {e}")


def main():
    """Main debug function."""
    print("🔍 Chunk File Validation Debug")
    print("=" * 40)

    # Test the most recent chunk files
    test_files = [
        "copernicus_climate_data/raw/2019/chunk_total_precipitation_2019_01.nc",
        "copernicus_climate_data/raw/2020/chunk_total_precipitation_2020_01.nc"
    ]

    for test_file in test_files:
        debug_chunk_file(test_file)
        print()


if __name__ == "__main__":
    main()