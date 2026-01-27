#!/usr/bin/env python3
"""
Optimized chunk merging function for large ERA5-Land datasets.
Addresses memory issues and slow performance in chunk merging.
"""

import xarray as xr
import time
from pathlib import Path
from typing import List


def optimized_chunk_merge(chunk_files: List[Path], output_file: Path,
                         time_dim: str = "valid_time") -> bool:
    """
    Optimized merging of NetCDF chunk files with reduced memory usage.

    Args:
        chunk_files: List of chunk file paths
        output_file: Output merged file path
        time_dim: Time dimension name (usually 'valid_time' for ERA5-Land)

    Returns:
        bool: Success status
    """
    print(f"🚀 Starting optimized merge of {len(chunk_files)} chunks...")
    start_time = time.time()

    try:
        # Sort files to ensure proper time ordering
        chunk_files = sorted(chunk_files)

        # Calculate total data size
        total_size = sum(f.stat().st_size for f in chunk_files)
        print(f"📊 Total data size: {total_size / 1024 / 1024:.1f} MB")

        # Strategy 1: Use dask for lazy loading if dataset is very large
        if total_size > 200 * 1024 * 1024:  # > 200 MB
            print("🔄 Using lazy loading strategy for large dataset...")

            # Open all files with chunking - this is lazy and doesn't load data
            datasets = []
            for i, chunk_file in enumerate(chunk_files):
                print(f"  📂 Opening chunk {i+1}/{len(chunk_files)}: {chunk_file.name}")

                ds = xr.open_dataset(
                    chunk_file,
                    engine='netcdf4',
                    chunks={
                        time_dim: 50,      # Small time chunks
                        'latitude': 100,   # Reasonable spatial chunks
                        'longitude': 100
                    }
                )
                datasets.append(ds)

            print(f"  🔗 Concatenating {len(datasets)} lazy datasets...")
            # This creates a lazy concatenated dataset
            merged_ds = xr.concat(datasets, dim=time_dim, data_vars='minimal', coords='minimal')

            print(f"  📅 Sorting by {time_dim}...")
            merged_ds = merged_ds.sortby(time_dim)

        else:
            print("🔄 Using standard approach for moderate dataset...")

            # Standard approach for smaller datasets
            datasets = []
            for i, chunk_file in enumerate(chunk_files):
                print(f"  📂 Loading chunk {i+1}/{len(chunk_files)}: {chunk_file.name}")
                ds = xr.open_dataset(chunk_file, engine='netcdf4')
                datasets.append(ds)

            print(f"  🔗 Concatenating {len(datasets)} datasets...")
            merged_ds = xr.concat(datasets, dim=time_dim)
            merged_ds = merged_ds.sortby(time_dim)

        # Optimized encoding for better compression and performance
        print(f"  💾 Saving with optimized compression...")

        encoding = {}
        for var in merged_ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 6,      # Good compression vs speed balance
                'shuffle': True,     # Improves compression for floating point
                'fletcher32': True,  # Checksum for data integrity
                'chunksizes': (
                    min(100, merged_ds.dims.get(time_dim, 50)),  # Time chunks
                    min(200, merged_ds.dims.get('latitude', 100)),   # Lat chunks
                    min(200, merged_ds.dims.get('longitude', 100))   # Lon chunks
                )
            }

        # Write to file with progress monitoring
        print(f"  ✍️  Writing to {output_file.name}...")
        merged_ds.to_netcdf(
            output_file,
            engine='netcdf4',
            encoding=encoding
        )

        # Get final stats
        final_size = output_file.stat().st_size
        compression_ratio = (total_size / final_size) if final_size > 0 else 1

        # Cleanup
        print(f"  🧹 Cleaning up memory...")
        for ds in datasets:
            ds.close()
        merged_ds.close()

        elapsed_time = time.time() - start_time

        print(f"✅ Merge completed successfully!")
        print(f"   📊 Input size: {total_size / 1024 / 1024:.1f} MB")
        print(f"   📊 Output size: {final_size / 1024 / 1024:.1f} MB")
        print(f"   📊 Compression ratio: {compression_ratio:.1f}x")
        print(f"   ⏱️  Time taken: {elapsed_time:.1f} seconds")
        print(f"   🚀 Throughput: {total_size / 1024 / 1024 / elapsed_time:.1f} MB/s")

        return True

    except Exception as e:
        print(f"❌ Merge failed: {e}")
        # Clean up partial file
        if output_file.exists():
            output_file.unlink()
        return False


def test_optimized_merge():
    """Test the optimized merge with existing chunk files."""
    print("🧪 Testing Optimized Chunk Merge")
    print("=" * 50)

    # Find chunk files
    chunk_files = list(Path("copernicus_climate_data/raw").rglob("chunk_*.nc"))

    # Filter to only valid NetCDF files (not ZIP files)
    valid_chunks = []
    for f in chunk_files:
        if not f.name.endswith('.extracted.nc'):  # Skip extracted copies
            # Quick check if it's a valid NetCDF (not ZIP)
            try:
                with xr.open_dataset(f, engine='netcdf4') as ds:
                    # Check if it has the expected time dimension
                    if 'valid_time' in ds.dims:
                        valid_chunks.append(f)
                    else:
                        print(f"⚠️  Skipping {f.name} - no valid_time dimension")
            except:
                print(f"⚠️  Skipping {f.name} - not a valid NetCDF")

    if len(valid_chunks) < 2:
        print(f"❌ Need at least 2 valid chunks for testing, found {len(valid_chunks)}")
        return False

    # Test with first few chunks
    test_chunks = sorted(valid_chunks)[:6]  # Test with first 6 months
    output_file = Path("copernicus_climate_data/test_merged.nc")

    print(f"📁 Testing merge with {len(test_chunks)} chunks:")
    for chunk in test_chunks:
        print(f"   - {chunk.name}")

    success = optimized_chunk_merge(test_chunks, output_file)

    if success and output_file.exists():
        # Validate merged file
        try:
            with xr.open_dataset(output_file, engine='netcdf4') as ds:
                print(f"\n✅ Merged file validation:")
                print(f"   Variables: {list(ds.data_vars.keys())}")
                print(f"   Dimensions: {dict(ds.dims)}")
                print(f"   Time range: {ds.valid_time.min().values} to {ds.valid_time.max().values}")

            # Clean up test file
            output_file.unlink()
            print(f"   🧹 Cleaned up test file")

        except Exception as e:
            print(f"❌ Merged file validation failed: {e}")
            return False

    return success


if __name__ == "__main__":
    success = test_optimized_merge()
    exit(0 if success else 1)