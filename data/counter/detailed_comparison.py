#!/usr/bin/env python
"""
Detailed comparison between albopictus.pkl (script output) and albopictus_test.pkl (notebook output)
to identify exactly what's different.
"""

import pandas as pd
import numpy as np

def analyze_differences():
    """Analyze the specific differences between the two files."""

    # Load both files
    print("Loading pickle files...")
    df_script = pd.read_pickle("albopictus.pkl")  # From albopictus.py script
    df_notebook = pd.read_pickle("albopictus_test.pkl")  # From notebook

    print(f"Script output: {df_script.shape[0]:,} rows × {df_script.shape[1]} columns")
    print(f"Notebook output: {df_notebook.shape[0]:,} rows × {df_notebook.shape[1]} columns")
    print(f"Row difference: {df_script.shape[0] - df_notebook.shape[0]} rows")
    print()

    # Check if columns are the same
    if not df_script.columns.equals(df_notebook.columns):
        print("❌ Columns are different!")
        script_only = set(df_script.columns) - set(df_notebook.columns)
        notebook_only = set(df_notebook.columns) - set(df_script.columns)
        if script_only:
            print(f"Columns only in script: {script_only}")
        if notebook_only:
            print(f"Columns only in notebook: {notebook_only}")
        return
    else:
        print("✅ Columns are identical")

    # If script has more rows, find which rows are extra
    if df_script.shape[0] > df_notebook.shape[0]:
        print(f"\nScript has {df_script.shape[0] - df_notebook.shape[0]} extra rows")

        # Try to identify the extra rows by creating a composite key
        # and finding rows that exist in script but not in notebook

        # Create a unique identifier for each row based on key columns
        key_cols = ['decimalLatitude', 'decimalLongitude', 'eventDate', 'individualCount', 'lifeStage']

        # Check if these columns exist
        missing_key_cols = [col for col in key_cols if col not in df_script.columns]
        if missing_key_cols:
            print(f"Warning: Missing key columns for comparison: {missing_key_cols}")
            # Use available columns
            key_cols = [col for col in key_cols if col in df_script.columns]

        print(f"Using key columns for comparison: {key_cols}")

        # Create composite keys
        df_script['_temp_key'] = df_script[key_cols].apply(
            lambda x: '|'.join(x.astype(str)), axis=1
        )
        df_notebook['_temp_key'] = df_notebook[key_cols].apply(
            lambda x: '|'.join(x.astype(str)), axis=1
        )

        # Find rows in script but not in notebook
        script_keys = set(df_script['_temp_key'])
        notebook_keys = set(df_notebook['_temp_key'])

        extra_in_script = script_keys - notebook_keys
        extra_in_notebook = notebook_keys - script_keys

        print(f"\nRows only in script: {len(extra_in_script)}")
        print(f"Rows only in notebook: {len(extra_in_notebook)}")

        if extra_in_script:
            print("\nFirst 5 rows only in script:")
            extra_script_rows = df_script[df_script['_temp_key'].isin(list(extra_in_script)[:5])]
            display_cols = ['decimalLatitude', 'decimalLongitude', 'eventDate', 'individualCount', 'lifeStage', 'keep']
            available_cols = [col for col in display_cols if col in extra_script_rows.columns]
            print(extra_script_rows[available_cols])

        if extra_in_notebook:
            print("\nFirst 5 rows only in notebook:")
            extra_notebook_rows = df_notebook[df_notebook['_temp_key'].isin(list(extra_in_notebook)[:5])]
            available_cols = [col for col in display_cols if col in extra_notebook_rows.columns]
            print(extra_notebook_rows[available_cols])

        # Clean up temporary columns
        df_script.drop('_temp_key', axis=1, inplace=True)
        df_notebook.drop('_temp_key', axis=1, inplace=True)

    # Check data quality indicators
    print(f"\n=== Data Quality Comparison ===")

    # Check filtering results
    if 'keep' in df_script.columns and 'keep' in df_notebook.columns:
        script_keep = df_script['keep'].value_counts()
        notebook_keep = df_notebook['keep'].value_counts()

        print("'keep' column distribution:")
        print(f"Script  - True: {script_keep.get(True, 0):,}, False: {script_keep.get(False, 0):,}")
        print(f"Notebook - True: {notebook_keep.get(True, 0):,}, False: {notebook_keep.get(False, 0):,}")

    # Check life stages
    if 'lifeStage' in df_script.columns and 'lifeStage' in df_notebook.columns:
        print(f"\nLife stage distribution:")
        script_stages = df_script['lifeStage'].value_counts()
        notebook_stages = df_notebook['lifeStage'].value_counts()

        all_stages = set(script_stages.index) | set(notebook_stages.index)
        for stage in sorted(all_stages):
            script_count = script_stages.get(stage, 0)
            notebook_count = notebook_stages.get(stage, 0)
            diff = script_count - notebook_count
            status = "=" if diff == 0 else f"({diff:+d})"
            print(f"  {stage}: Script {script_count:,}, Notebook {notebook_count:,} {status}")

    # Check individual counts
    if 'individualCount' in df_script.columns and 'individualCount' in df_notebook.columns:
        print(f"\nIndividual count statistics:")
        print(f"Script  - Mean: {df_script['individualCount'].mean():.3f}, "
              f"Zero counts: {(df_script['individualCount'] == 0).sum():,}, "
              f"Positive: {(df_script['individualCount'] > 0).sum():,}")
        print(f"Notebook - Mean: {df_notebook['individualCount'].mean():.3f}, "
              f"Zero counts: {(df_notebook['individualCount'] == 0).sum():,}, "
              f"Positive: {(df_notebook['individualCount'] > 0).sum():,}")

    # Check sampling effort validation
    if 'samplingEffort_diff' in df_script.columns and 'samplingEffort_diff' in df_notebook.columns:
        print(f"\nSampling effort validation:")
        script_valid = (abs(df_script['samplingEffort_diff']) <= 2).sum()
        notebook_valid = (abs(df_notebook['samplingEffort_diff']) <= 2).sum()
        print(f"Script  - Valid sampling effort: {script_valid:,}")
        print(f"Notebook - Valid sampling effort: {notebook_valid:,}")

if __name__ == "__main__":
    analyze_differences()