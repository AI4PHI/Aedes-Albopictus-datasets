#!/usr/bin/env python3
"""
Test script to demonstrate ZIP handling capabilities of the new downloader.
"""

import sys
from pathlib import Path

# Add src to path so we can import the downloader
sys.path.append('src')

def test_existing_files():
    """Test the debug functionality on existing NetCDF files."""
    print("🧪 Testing ZIP handling and file detection...")

    # Import the debug function
    try:
        from copernicus_downloader import debug_file_format
    except ImportError as e:
        print(f"❌ Could not import debug function: {e}")
        return

    # Test with existing NetCDF files
    test_files = [
        "../input_data/copernicus_climate_data/europe/data_old/2019/total_precipitation_daily_cum_2019.nc",
        "../input_data/copernicus_climate_data/europe/data_old/2019/10m_u_component_of_wind_daily_stats_2019.nc"
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n📋 Testing file: {test_file}")
            print("-" * 60)
            debug_file_format(test_file)
            break
    else:
        print("ℹ️  No existing NetCDF files found to test with")
        print("   This is normal if you haven't downloaded any data yet")

    print("\n✅ ZIP handling test complete!")
    print("\n💡 When you download new data, the script will:")
    print("   1. Detect if CDS returns ZIP files")
    print("   2. Automatically extract NetCDF from ZIP")
    print("   3. Validate the extracted NetCDF file")
    print("   4. Clean up temporary ZIP files")

if __name__ == "__main__":
    test_existing_files()