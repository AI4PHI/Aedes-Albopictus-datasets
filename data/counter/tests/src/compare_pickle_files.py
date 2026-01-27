#!/usr/bin/env python
"""
Pickle File Comparison Script

This script compares two pickle files to check if they contain identical data.
It provides detailed information about any differences found.

Usage:
    python compare_pickle_files.py [file1] [file2]

If no arguments provided, defaults to comparing albopictus.pkl and albopictus_test.pkl
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse


def compare_dataframes(df1, df2, name1="File 1", name2="File 2"):
    """
    Compare two DataFrames and return detailed comparison results.

    Args:
        df1, df2: DataFrames to compare
        name1, name2: Names for the DataFrames in output

    Returns:
        dict: Comparison results
    """
    results = {
        'identical': True,
        'differences': []
    }

    # Check shapes
    if df1.shape != df2.shape:
        results['identical'] = False
        results['differences'].append(
            f"Shape difference: {name1} {df1.shape} vs {name2} {df2.shape}"
        )
        return results

    # Check column names
    if not df1.columns.equals(df2.columns):
        results['identical'] = False
        diff_cols = set(df1.columns) ^ set(df2.columns)
        results['differences'].append(
            f"Column differences: {diff_cols}"
        )
        return results

    # Check index
    if not df1.index.equals(df2.index):
        results['identical'] = False
        results['differences'].append("Index values are different")

    # Check data values column by column
    for col in df1.columns:
        try:
            # For numeric columns, use pandas.testing.assert_series_equal with tolerance
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                try:
                    pd.testing.assert_series_equal(
                        df1[col], df2[col],
                        check_exact=False,
                        rtol=1e-10,
                        atol=1e-10,
                        check_names=False
                    )
                except AssertionError:
                    results['identical'] = False
                    # Find specific differences
                    diff_mask = ~np.isclose(df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True)
                    n_diffs = diff_mask.sum()
                    results['differences'].append(
                        f"Column '{col}': {n_diffs} numeric differences found"
                    )

                    # Show first few differences
                    if n_diffs > 0:
                        diff_indices = df1.index[diff_mask][:5]  # First 5 differences
                        for idx in diff_indices:
                            val1 = df1.loc[idx, col]
                            val2 = df2.loc[idx, col]
                            results['differences'].append(
                                f"  Row {idx}: {val1} vs {val2}"
                            )
                        if n_diffs > 5:
                            results['differences'].append(f"  ... and {n_diffs - 5} more differences")

            # For non-numeric columns, direct comparison
            else:
                if not df1[col].equals(df2[col]):
                    results['identical'] = False
                    diff_mask = df1[col] != df2[col]
                    # Handle NaN comparisons
                    nan_mask1 = df1[col].isna()
                    nan_mask2 = df2[col].isna()
                    diff_mask = diff_mask & ~(nan_mask1 & nan_mask2)

                    n_diffs = diff_mask.sum()
                    results['differences'].append(
                        f"Column '{col}': {n_diffs} value differences found"
                    )

                    # Show first few differences
                    if n_diffs > 0:
                        diff_indices = df1.index[diff_mask][:5]
                        for idx in diff_indices:
                            val1 = df1.loc[idx, col]
                            val2 = df2.loc[idx, col]
                            results['differences'].append(
                                f"  Row {idx}: '{val1}' vs '{val2}'"
                            )
                        if n_diffs > 5:
                            results['differences'].append(f"  ... and {n_diffs - 5} more differences")

        except Exception as e:
            results['identical'] = False
            results['differences'].append(f"Error comparing column '{col}': {str(e)}")

    return results


def load_pickle_file(filepath):
    """
    Load a pickle file and return its contents.

    Args:
        filepath: Path to the pickle file

    Returns:
        The loaded object
    """
    try:
        return pd.read_pickle(filepath)
    except Exception as e:
        raise ValueError(f"Error loading {filepath}: {str(e)}")


def compare_pickle_files(file1_path, file2_path):
    """
    Compare two pickle files.

    Args:
        file1_path, file2_path: Paths to the pickle files

    Returns:
        dict: Comparison results
    """
    print(f"Comparing pickle files:")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}")
    print()

    # Check if files exist
    if not Path(file1_path).exists():
        return {"error": f"File not found: {file1_path}"}
    if not Path(file2_path).exists():
        return {"error": f"File not found: {file2_path}"}

    # Load the pickle files
    try:
        data1 = load_pickle_file(file1_path)
        data2 = load_pickle_file(file2_path)
    except ValueError as e:
        return {"error": str(e)}

    # Check file sizes
    size1 = Path(file1_path).stat().st_size
    size2 = Path(file2_path).stat().st_size
    print(f"File sizes:")
    print(f"  {file1_path}: {size1:,} bytes")
    print(f"  {file2_path}: {size2:,} bytes")

    if size1 != size2:
        print(f"  Size difference: {abs(size1 - size2):,} bytes")
    print()

    # Check data types
    if type(data1) != type(data2):
        return {
            "identical": False,
            "error": f"Different data types: {type(data1).__name__} vs {type(data2).__name__}"
        }

    # If both are DataFrames, do detailed comparison
    if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
        print("Both files contain pandas DataFrames")
        print(f"Data info:")
        print(f"  {file1_path}: {data1.shape[0]:,} rows × {data1.shape[1]} columns")
        print(f"  {file2_path}: {data2.shape[0]:,} rows × {data2.shape[1]} columns")
        print()

        return compare_dataframes(data1, data2, Path(file1_path).name, Path(file2_path).name)

    # For other data types, try direct comparison
    else:
        try:
            if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
                identical = np.array_equal(data1, data2, equal_nan=True)
            else:
                identical = data1 == data2

            return {
                "identical": identical,
                "differences": [] if identical else ["Objects are not equal"]
            }
        except Exception as e:
            return {
                "identical": False,
                "error": f"Error comparing objects: {str(e)}"
            }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare two pickle files")
    parser.add_argument("file1", nargs="?", default="albopictus.pkl",
                       help="First pickle file (default: albopictus.pkl)")
    parser.add_argument("file2", nargs="?", default="albopictus_test.pkl",
                       help="Second pickle file (default: albopictus_test.pkl)")

    args = parser.parse_args()

    # Run comparison
    results = compare_pickle_files(args.file1, args.file2)

    # Print results
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return 1

    if results["identical"]:
        print("✅ SUCCESS: The pickle files contain identical data!")
        return 0
    else:
        print("❌ DIFFERENCES FOUND:")
        print()
        for diff in results["differences"]:
            print(f"  • {diff}")
        print()
        print(f"Total differences: {len(results['differences'])}")
        return 1


if __name__ == "__main__":
    exit(main())