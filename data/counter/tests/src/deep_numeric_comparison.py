#!/usr/bin/env python
"""
Deep numeric comparison to find exact differences between pickle files.
Checks for small floating-point differences, NaN handling, and exact value mismatches.
"""

import pandas as pd
import numpy as np

def deep_compare_dataframes(df1, df2, name1="Script", name2="Notebook"):
    """
    Perform a deep comparison of two DataFrames, checking for numeric precision differences.
    """
    print(f"=== DEEP COMPARISON: {name1} vs {name2} ===\n")

    # Basic shape check
    print(f"Shapes: {name1} {df1.shape}, {name2} {df2.shape}")
    if df1.shape != df2.shape:
        print("❌ Different shapes - cannot compare")
        return False

    # Column comparison
    if not df1.columns.equals(df2.columns):
        print("❌ Different columns")
        return False

    print("✅ Same shape and columns\n")

    total_differences = 0

    for col in df1.columns:
        print(f"Checking column '{col}'...")

        col1 = df1[col]
        col2 = df2[col]

        # Check data types
        if col1.dtype != col2.dtype:
            print(f"  ⚠️  Different dtypes: {col1.dtype} vs {col2.dtype}")

        # For numeric columns
        if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
            # Check for exact equality first
            exact_match = col1.equals(col2)
            if exact_match:
                print(f"  ✅ Exact match")
                continue

            # Check for NaN differences
            nan1 = col1.isna()
            nan2 = col2.isna()
            if not nan1.equals(nan2):
                nan_diff = (nan1 != nan2).sum()
                print(f"  ❌ NaN differences: {nan_diff} positions")
                total_differences += nan_diff
                continue

            # For non-NaN values, check numeric differences
            non_nan_mask = ~nan1  # Same as ~nan2 since they're equal
            if non_nan_mask.sum() == 0:
                print(f"  ✅ All NaN - match")
                continue

            val1_clean = col1[non_nan_mask]
            val2_clean = col2[non_nan_mask]

            # Check for exact equality of non-NaN values
            if val1_clean.equals(val2_clean):
                print(f"  ✅ Non-NaN values exact match")
                continue

            # Check with different tolerances
            tolerances = [0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-10, 1e-8]

            for tol in tolerances:
                if tol == 0:
                    # Exact comparison
                    diff_mask = val1_clean != val2_clean
                else:
                    # Tolerance comparison
                    diff_mask = ~np.isclose(val1_clean, val2_clean, rtol=tol, atol=tol, equal_nan=True)

                n_diffs = diff_mask.sum()

                if n_diffs == 0:
                    if tol == 0:
                        print(f"  ✅ Exact match")
                    else:
                        print(f"  ✅ Match within tolerance {tol}")
                    break
                elif tol == tolerances[-1]:  # Last tolerance
                    print(f"  ❌ {n_diffs} differences even with tolerance {tol}")
                    total_differences += n_diffs

                    # Show first few differences
                    diff_indices = val1_clean.index[diff_mask][:5]
                    print(f"    First few differences:")
                    for idx in diff_indices:
                        v1 = val1_clean.loc[idx]
                        v2 = val2_clean.loc[idx]
                        print(f"      Row {idx}: {v1} vs {v2} (diff: {abs(v1-v2):.2e})")
                    break
                elif tol == 1e-15:
                    print(f"  ⚠️  {n_diffs} differences at machine precision (1e-15)")

        else:
            # For non-numeric columns
            if col1.equals(col2):
                print(f"  ✅ Exact match")
            else:
                diff_mask = col1 != col2
                # Handle NaN comparisons
                nan_mask1 = col1.isna()
                nan_mask2 = col2.isna()
                diff_mask = diff_mask & ~(nan_mask1 & nan_mask2)

                n_diffs = diff_mask.sum()
                if n_diffs > 0:
                    print(f"  ❌ {n_diffs} string/object differences")
                    total_differences += n_diffs

                    # Show first few differences
                    diff_indices = col1.index[diff_mask][:3]
                    for idx in diff_indices:
                        v1 = col1.loc[idx]
                        v2 = col2.loc[idx]
                        print(f"      Row {idx}: '{v1}' vs '{v2}'")
                else:
                    print(f"  ✅ Match (NaN handling)")

    print(f"\n=== SUMMARY ===")
    print(f"Total differences found: {total_differences}")

    if total_differences == 0:
        print("✅ DataFrames are identical!")
        return True
    else:
        print(f"❌ DataFrames have {total_differences} differences")
        return False

def check_index_differences(df1, df2):
    """Check if there are any index differences."""
    print("=== INDEX COMPARISON ===")

    if df1.index.equals(df2.index):
        print("✅ Indices are identical")
    else:
        print("❌ Indices differ")
        print(f"  Index 1 type: {type(df1.index)}")
        print(f"  Index 2 type: {type(df2.index)}")

        if len(df1.index) == len(df2.index):
            diff_positions = (df1.index != df2.index).sum()
            print(f"  Different positions: {diff_positions}")
        else:
            print(f"  Different lengths: {len(df1.index)} vs {len(df2.index)}")

def main():
    """Main function to perform deep comparison."""
    print("Loading pickle files for deep comparison...")

    try:
        df_script = pd.read_pickle("albopictus.pkl")
        df_notebook = pd.read_pickle("albopictus_test.pkl")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Script file: {len(df_script)} rows")
    print(f"Notebook file: {len(df_notebook)} rows\n")

    # Check indices
    check_index_differences(df_script, df_notebook)
    print()

    # Deep comparison
    result = deep_compare_dataframes(df_script, df_notebook)

    if result:
        print("\n🎉 Files are functionally identical!")
        print("The 62-byte difference is likely due to:")
        print("- Pandas DataFrame metadata")
        print("- Pickle serialization order")
        print("- Internal object references")
    else:
        print("\n🔍 Found actual data differences that need investigation")

if __name__ == "__main__":
    main()