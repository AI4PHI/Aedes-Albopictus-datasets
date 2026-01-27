#!/usr/bin/env python3
"""
Compare old reference CORDEX results with newly generated results.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.check_similarity_df import load_csv_zip
import pandas as pd
import numpy as np

def compare_dataframes(df1, df2, label1="OLD", label2="NEW"):
    """Compare two dataframes and report differences"""
    print("\n" + "="*80)
    print(f"📊 Dataframe Comparison: {label1} vs {label2}")
    print("="*80)
    
    # Shape comparison
    print(f"\n📐 Shape:")
    print(f"  {label1}: {df1.shape[0]:,} rows × {df1.shape[1]} columns")
    print(f"  {label2}: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
    if df1.shape == df2.shape:
        print("  ✅ Shapes are IDENTICAL")
    else:
        print("  ⚠️  Shapes DIFFER")
    
    # Column comparison
    print(f"\n📋 Columns:")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 == cols2:
        print(f"  ✅ All {len(cols1)} columns match")
    else:
        missing_in_new = cols1 - cols2
        extra_in_new = cols2 - cols1
        if missing_in_new:
            print(f"  ⚠️  Missing in {label2}: {missing_in_new}")
        if extra_in_new:
            print(f"  ⚠️  Extra in {label2}: {extra_in_new}")
    
    # Common columns
    common_cols = sorted(cols1 & cols2)
    print(f"\n🔍 Comparing {len(common_cols)} common columns...")
    
    # Data type comparison
    print(f"\n📊 Data Types:")
    type_diffs = []
    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            type_diffs.append((col, df1[col].dtype, df2[col].dtype))
    
    if type_diffs:
        print(f"  ⚠️  {len(type_diffs)} columns have different types:")
        for col, dtype1, dtype2 in type_diffs[:10]:  # Show first 10
            print(f"     - {col}: {dtype1} → {dtype2}")
    else:
        print("  ✅ All data types match")
    
    # Value comparison for key columns
    print(f"\n🔢 Value Comparison:")
    
    key_numeric_cols = [col for col in common_cols if df1[col].dtype in ['float64', 'int64']]
    key_categorical_cols = ['Suitable', 'Temperature Suitable', 'Precipitation Suitable', 'presence_numeric']
    key_categorical_cols = [col for col in key_categorical_cols if col in common_cols]
    
    # Numeric columns
    if key_numeric_cols:
        print(f"\n  Numeric columns (showing first 5):")
        for col in key_numeric_cols[:5]:
            val1 = df1[col].dropna()
            val2 = df2[col].dropna()
            
            if len(val1) > 0 and len(val2) > 0:
                diff = np.abs(val1 - val2).mean()
                max_diff = np.abs(val1 - val2).max()
                print(f"    {col}:")
                print(f"      Mean absolute diff: {diff:.6f}")
                print(f"      Max absolute diff: {max_diff:.6f}")
    
    # Categorical columns
    if key_categorical_cols:
        print(f"\n  Categorical columns:")
        for col in key_categorical_cols:
            if col in df1.columns and col in df2.columns:
                # Value counts comparison
                vc1 = df1[col].value_counts().sort_index()
                vc2 = df2[col].value_counts().sort_index()
                
                print(f"    {col}:")
                print(f"      {label1}: {dict(vc1)}")
                print(f"      {label2}: {dict(vc2)}")
                if vc1.equals(vc2):
                    print(f"      ✅ Identical")
                else:
                    print(f"      ⚠️  Different")

def main():
    # Paths (relative to the tests/ directory)
    old_file = 'data/albopictus_presence_absence_ecdc_copernicus_2020.zip'
    new_file = '../data/outputs/ecdc_albopictus_cordex_2020.zip'
    
    print("\n" + "="*80)
    print("📊 Comparing CORDEX Results: OLD vs NEW")
    print("="*80)
    print(f"OLD: {old_file}")
    print(f"NEW: {new_file}")
    
    try:
        # Load both datasets
        print("\n📂 Loading datasets...")
        old_df = load_csv_zip(old_file)
        new_df = load_csv_zip(new_file)
        
        print(f"✅ OLD: {old_df.shape[0]} rows × {old_df.shape[1]} columns")
        print(f"✅ NEW: {new_df.shape[0]} rows × {new_df.shape[1]} columns")
        
        # Compare
        print("\n🔍 Running similarity check...\n")
        compare_dataframes(
            old_df, 
            new_df,
            label1='OLD (reference)',
            label2='NEW (current)'
        )
        
        print("\n" + "="*80)
        print("✅ Comparison complete!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        print("Make sure you've run the CORDEX processing first:")
        print("  python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()