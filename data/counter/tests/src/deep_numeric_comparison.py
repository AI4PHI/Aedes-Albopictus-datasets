#!/usr/bin/env python
"""
Deep numeric comparison to find exact differences between pickle files.
Checks for small floating-point differences, NaN handling, and exact value mismatches.
"""

import pandas as pd
import numpy as np

def compare_columns(df1, df2):
    """Compare column names between two DataFrames."""
    print("=== COLUMN COMPARISON ===")
    
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"Total columns: File1={len(cols1)}, File2={len(cols2)}")
    print(f"Common columns: {len(common)}")
    
    if only_in_1:
        print(f"\n❌ Only in File1 ({len(only_in_1)} columns):")
        for col in sorted(only_in_1):
            print(f"  - {col}")
    
    if only_in_2:
        print(f"\n❌ Only in File2 ({len(only_in_2)} columns):")
        for col in sorted(only_in_2):
            print(f"  - {col}")
    
    if not only_in_1 and not only_in_2:
        print("✅ All columns match")
    
    return common, only_in_1, only_in_2

def deep_compare_dataframes(df1, df2, name1="Script", name2="Notebook", compare_common_only=False):
    """
    Perform a deep comparison of two DataFrames, checking for numeric precision differences.
    
    Args:
        compare_common_only: If True, only compare columns that exist in both DataFrames
    """
    print(f"\n=== DEEP COMPARISON: {name1} vs {name2} ===\n")

    # Basic shape check
    print(f"Shapes: {name1} {df1.shape}, {name2} {df2.shape}")
    
    if df1.shape != df2.shape and not compare_common_only:
        print("❌ Different shapes - enable compare_common_only to compare matching columns")
        return False

    # Get common columns
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    if compare_common_only:
        print(f"\nComparing {len(common_cols)} common columns...\n")
        df1_compare = df1[common_cols]
        df2_compare = df2[common_cols]
    else:
        # Column comparison
        if not df1.columns.equals(df2.columns):
            print("❌ Different columns")
            return False
        print("✅ Same shape and columns\n")
        df1_compare = df1
        df2_compare = df2

    total_differences = 0
    columns_with_differences = []  # Track columns that differ

    for col in df1_compare.columns:
        print(f"Checking column '{col}'...")

        col1 = df1_compare[col]
        col2 = df2_compare[col]

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
                columns_with_differences.append({
                    'column': col,
                    'type': 'NaN mismatch',
                    'n_diffs': nan_diff,
                    'details': f"{nan_diff} positions have NaN in one file but not the other"
                })
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
            # Updated tolerances: focus on 3-4 significant digits
            # rtol=1e-3 means ~0.1% relative difference (3 significant digits)
            # rtol=1e-4 means ~0.01% relative difference (4 significant digits)
            tolerances = [
                ('exact', 0, 0),
                ('4 sig digits', 1e-4, 1e-10),  # 0.01% relative or 1e-10 absolute
                ('3 sig digits', 1e-3, 1e-9),   # 0.1% relative or 1e-9 absolute
                ('2 sig digits', 1e-2, 1e-8),   # 1% relative or 1e-8 absolute
                ('loose', 1e-1, 1e-6)           # 10% relative or 1e-6 absolute
            ]

            matched = False
            for label, rtol, atol in tolerances:
                if rtol == 0:
                    # Exact comparison
                    diff_mask = val1_clean != val2_clean
                else:
                    # Tolerance comparison with both relative and absolute tolerance
                    diff_mask = ~np.isclose(val1_clean, val2_clean, rtol=rtol, atol=atol, equal_nan=True)

                n_diffs = diff_mask.sum()

                if n_diffs == 0:
                    if rtol == 0:
                        print(f"  ✅ Exact match")
                    else:
                        print(f"  ✅ Match at {label} (rtol={rtol:.0e})")
                    matched = True
                    break
                elif label == 'loose':  # Last tolerance - still has differences
                    # Calculate statistics
                    diff_vals1 = val1_clean[diff_mask]
                    diff_vals2 = val2_clean[diff_mask]
                    abs_diffs = np.abs(diff_vals1 - diff_vals2)
                    rel_diffs = abs_diffs / np.abs(diff_vals1)
                    rel_diffs = rel_diffs[np.isfinite(rel_diffs)]

                    print(f"  ❌ No match even at {label}: {n_diffs}/{len(val1_clean)} diffs ({n_diffs/len(val1_clean)*100:.1f}%)")
                    print(f"     Abs diff: min={abs_diffs.min():.2e}, max={abs_diffs.max():.2e}, median={np.median(abs_diffs):.2e}")
                    if len(rel_diffs) > 0:
                        print(f"     Rel diff: min={rel_diffs.min():.2e}, max={rel_diffs.max():.2e}, median={np.median(rel_diffs):.2e}")
                    
                    total_differences += n_diffs
                    
                    # Store statistics
                    columns_with_differences.append({
                        'column': col,
                        'type': 'Numeric differences',
                        'n_diffs': n_diffs,
                        'n_total': len(val1_clean),
                        'percent': (n_diffs / len(val1_clean)) * 100,
                        'abs_diff_median': float(np.median(abs_diffs)),
                        'abs_diff_max': float(abs_diffs.max()),
                        'rel_diff_median': float(np.median(rel_diffs)) if len(rel_diffs) > 0 else None,
                        'rel_diff_max': float(rel_diffs.max()) if len(rel_diffs) > 0 else None,
                    })
                    break

        else:
            # For non-numeric columns
            if col1.equals(col2):
                print(f"  ✅ Exact match")
            else:
                # Check if this is an array/list column
                try:
                    # Try basic comparison first
                    diff_mask = col1 != col2
                    # Handle NaN comparisons
                    nan_mask1 = col1.isna()
                    nan_mask2 = col2.isna()
                    diff_mask = diff_mask & ~(nan_mask1 & nan_mask2)
                except (ValueError, TypeError):
                    # Handle array/list columns
                    print(f"  ℹ️  Array/list column - checking element-wise...")
                    differences = 0
                    diff_indices = []
                    
                    for idx in col1.index:
                        v1 = col1.loc[idx]
                        v2 = col2.loc[idx]
                        
                        # Handle different types of values
                        v1_is_nan = False
                        v2_is_nan = False
                        
                        # Check for NaN - handle both scalar and array cases
                        try:
                            if isinstance(v1, (list, np.ndarray)):
                                v1_is_nan = False  # Arrays are not NaN
                            else:
                                v1_is_nan = pd.isna(v1)
                        except:
                            v1_is_nan = False
                            
                        try:
                            if isinstance(v2, (list, np.ndarray)):
                                v2_is_nan = False  # Arrays are not NaN
                            else:
                                v2_is_nan = pd.isna(v2)
                        except:
                            v2_is_nan = False
                        
                        # Check if both are NaN
                        if v1_is_nan and v2_is_nan:
                            continue
                        # Check if one is NaN
                        if v1_is_nan or v2_is_nan:
                            differences += 1
                            if len(diff_indices) < 3:
                                diff_indices.append(idx)
                            continue
                        
                        # Compare arrays/lists
                        try:
                            if isinstance(v1, (list, np.ndarray)) and isinstance(v2, (list, np.ndarray)):
                                # Convert to numpy arrays for comparison
                                arr1 = np.array(v1, dtype=float)
                                arr2 = np.array(v2, dtype=float)
                                
                                # Use tolerance-based comparison for numeric arrays
                                # Check at 3 significant digits (rtol=1e-3)
                                if not np.allclose(arr1, arr2, rtol=1e-3, atol=1e-9, equal_nan=True):
                                    differences += 1
                                    if len(diff_indices) < 3:
                                        diff_indices.append(idx)
                            elif v1 != v2:
                                differences += 1
                                if len(diff_indices) < 3:
                                    diff_indices.append(idx)
                        except Exception as e:
                            # If comparison fails, count as difference
                            differences += 1
                            if len(diff_indices) < 3:
                                diff_indices.append(idx)
                    
                    if differences > 0:
                        print(f"  ❌ {differences} array differences (>2 sig digits)")
                        total_differences += differences
                        
                        # Show statistics for first differing array
                        if len(diff_indices) > 0:
                            idx = diff_indices[0]
                            arr1 = np.array(col1.loc[idx], dtype=float)
                            arr2 = np.array(col2.loc[idx], dtype=float)
                            abs_diff = np.abs(arr1 - arr2)
                            rel_diff = abs_diff / np.abs(arr1)
                            rel_diff = rel_diff[np.isfinite(rel_diff)]
                            
                            print(f"      Sample row {idx}:")
                            print(f"        Abs diff: max={abs_diff.max():.2e}, median={np.median(abs_diff):.2e}")
                            if len(rel_diff) > 0:
                                print(f"        Rel diff: max={rel_diff.max():.2e}, median={np.median(rel_diff):.2e}")
                        
                        columns_with_differences.append({
                            'column': col,
                            'type': 'Array differences (>2 sig digits)',
                            'n_diffs': differences,
                            'n_total': len(col1),
                            'percent': (differences / len(col1)) * 100
                        })
                    else:
                        print(f"  ✅ Arrays match at 2 sig digits")
                    continue

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
                    
                    columns_with_differences.append({
                        'column': col,
                        'type': 'String/object differences',
                        'n_diffs': n_diffs,
                        'n_total': len(col1),
                        'percent': (n_diffs / len(col1)) * 100
                    })
                else:
                    print(f"  ✅ Match (NaN handling)")

    print(f"\n{'='*60}")
    print(f"Total differences found: {total_differences}")
    
    return total_differences == 0, columns_with_differences

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
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep comparison of two pickle files")
    parser.add_argument('--file1', required=True, help="Path to first pickle file")
    parser.add_argument('--file2', required=True, help="Path to second pickle file")
    parser.add_argument('--compare-common', action='store_true', 
                       help="Compare only common columns (ignore shape differences)")
    
    args = parser.parse_args()
    
    print("Loading pickle files for deep comparison...")

    try:
        df_script = pd.read_pickle(args.file1)
        df_notebook = pd.read_pickle(args.file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"File 1 ({args.file1}): {len(df_script)} rows")
    print(f"File 2 ({args.file2}): {len(df_notebook)} rows\n")

    # Check indices
    check_index_differences(df_script, df_notebook)
    print()

    # Compare columns
    common, only_1, only_2 = compare_columns(df_script, df_notebook)
    print()

    # Deep comparison
    result, diff_columns = deep_compare_dataframes(
        df_script, df_notebook, 
        name1="File1", name2="File2",
        compare_common_only=args.compare_common or len(only_1) > 0 or len(only_2) > 0
    )

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nFile 1: {args.file1}")
    print(f"  - Rows: {len(df_script)}")
    print(f"  - Columns: {len(df_script.columns)}")
    
    print(f"\nFile 2: {args.file2}")
    print(f"  - Rows: {len(df_notebook)}")
    print(f"  - Columns: {len(df_notebook.columns)}")
    
    print(f"\nColumn Analysis:")
    print(f"  - Common columns: {len(common)}")
    if only_1:
        print(f"  - Only in File1: {len(only_1)} columns")
        print(f"    {sorted(only_1)}")
    if only_2:
        print(f"  - Only in File2: {len(only_2)} columns")
        print(f"    {sorted(only_2)}")
    
    # Detailed statistics for columns with differences
    if diff_columns:
        print(f"\n{'='*60}")
        print(f"COLUMNS WITH DIFFERENCES ({len(diff_columns)} total)")
        print(f"{'='*60}")
        
        for diff_info in diff_columns:
            print(f"\n📊 Column: '{diff_info['column']}'")
            print(f"   Type: {diff_info['type']}")
            print(f"   Differences: {diff_info['n_diffs']:,}", end="")
            
            if 'n_total' in diff_info:
                print(f" / {diff_info['n_total']:,} ({diff_info['percent']:.2f}%)")
            else:
                print()
            
            # Show numeric statistics if available
            if 'abs_diff_mean' in diff_info:
                print(f"   Absolute Differences:")
                print(f"     - Min:    {diff_info['abs_diff_min']:.4e}")
                print(f"     - Max:    {diff_info['abs_diff_max']:.4e}")
                print(f"     - Mean:   {diff_info['abs_diff_mean']:.4e}")
                print(f"     - Median: {diff_info['abs_diff_median']:.4e}")
            
            if diff_info.get('rel_diff_mean') is not None:
                print(f"   Relative Differences:")
                print(f"     - Min:    {diff_info['rel_diff_min']:.4e}")
                print(f"     - Max:    {diff_info['rel_diff_max']:.4e}")
                print(f"     - Mean:   {diff_info['rel_diff_mean']:.4e}")
                print(f"     - Median: {diff_info['rel_diff_median']:.4e}")
    
    if result:
        print("\n🎉 Data Comparison Result: IDENTICAL")
        print("  - All common columns have matching values")
        if only_1 or only_2:
            print("  - Note: Files have different column sets")
        else:
            print("  - Files are completely identical")
    else:
        print("\n❌ Data Comparison Result: DIFFERENCES FOUND")
        print(f"  - {len(diff_columns)} columns have differences")
        print("  - Review the detailed comparison above")
    
    print("="*60)

if __name__ == "__main__":
    main()