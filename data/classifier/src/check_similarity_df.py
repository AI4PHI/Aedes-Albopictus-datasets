#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import pandas as pd

def load_csv_zip(path: str) -> pd.DataFrame:
    # Pandas auto-detects the single CSV inside the zip
    return pd.read_csv(path, compression="zip")

def normalize_non_numeric(s: pd.Series, strip: bool) -> pd.Series:
    if strip and s.dtype == object:
        return s.astype("string").str.strip()
    return s

def compare_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    index_col: str | None,
    rtol: float,
    atol: float,
    equal_nan: bool,
    strip: bool,
    samples: int,
) -> bool:
    ok = True
    print("=== STRUCTURE CHECKS ===")

    # Optional index alignment by a key (so row order doesn't matter)
    if index_col is not None:
        if index_col not in df1.columns or index_col not in df2.columns:
            print(f"❌ index column '{index_col}' not present in both dataframes.")
            return False
        # Set index and align on intersection of keys
        df1 = df1.set_index(index_col, drop=False)
        df2 = df2.set_index(index_col, drop=False)

        common_keys = df1.index.intersection(df2.index)
        only1 = df1.index.difference(df2.index)
        only2 = df2.index.difference(df1.index)

        print(f"Keys only in file1: {len(only1)}")
        print(f"Keys only in file2: {len(only2)}")
        if len(only1) or len(only2):
            ok = False
            # Show a few examples
            if len(only1):
                print("  e.g. keys only in file1:", list(only1[:min(len(only1), samples)]))
            if len(only2):
                print("  e.g. keys only in file2:", list(only2[:min(len(only2), samples)]))

        df1 = df1.loc[common_keys]
        df2 = df2.loc[common_keys]

    # Shape check
    same_rows = len(df1) == len(df2)
    same_cols = list(df1.columns) == list(df2.columns)
    same_colset = set(df1.columns) == set(df2.columns)

    print(f"Rows: file1={len(df1)} file2={len(df2)} -> {'OK' if same_rows else 'DIFF'}")
    if not same_rows:
        ok = False

    if same_cols:
        print("Columns: identical order and names -> OK")
    elif same_colset:
        print("Columns: same set but different order -> WARN")
    else:
        print("❌ Columns: different sets")
        missing_in_1 = set(df2.columns) - set(df1.columns)
        missing_in_2 = set(df1.columns) - set(df2.columns)
        if missing_in_1:
            print("  Present only in file2:", sorted(missing_in_1))
        if missing_in_2:
            print("  Present only in file1:", sorted(missing_in_2))
        return False

    # If order differs but set is same, align column order
    if not same_cols and same_colset:
        df2 = df2[df1.columns]

    print("\n=== PER-COLUMN COMPARISON ===")
    ncols = len(df1.columns)
    n_equal = 0
    n_diff = 0

    # Compare columns
    for col in df1.columns:
        s1 = df1[col]
        s2 = df2[col]

        # Handle dtype differences gently
        is_num1 = pd.api.types.is_numeric_dtype(s1)
        is_num2 = pd.api.types.is_numeric_dtype(s2)
        if is_num1 and is_num2:
            # Numeric: allclose with tolerance
            a = s1.to_numpy(dtype=float, copy=False)
            b = s2.to_numpy(dtype=float, copy=False)

            # both NaN -> equal_nan
            with np.errstate(invalid='ignore'):
                comp = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

            n_bad = int((~comp).sum())
            if n_bad == 0:
                print(f"[OK] numeric {col}")
                n_equal += 1
            else:
                ok = False
                n_diff += 1
                # stats on numeric diffs
                diffs = np.abs(a - b)
                diffs = diffs[~np.isnan(diffs)]
                maxdiff = float(diffs.max()) if diffs.size else np.nan
                meandiff = float(diffs.mean()) if diffs.size else np.nan
                print(f"[DIFF] numeric {col}: {n_bad} values differ (rtol={rtol}, atol={atol}). "
                      f"max|Δ|={maxdiff}, mean|Δ|={meandiff}")

                # show sample rows
                bad_idx = np.where(~comp)[0][:samples]
                for i in bad_idx:
                    idx_label = df1.index[i] if index_col is not None else i
                    print(f"  example @{idx_label}: file1={a[i]!r} file2={b[i]!r}")

        else:
            # Non-numeric or mixed: compare as strings (optional strip)
            s1n = normalize_non_numeric(s1.astype("string"), strip)
            s2n = normalize_non_numeric(s2.astype("string"), strip)

            equals = s1n.fillna(pd.NA).equals(s2n.fillna(pd.NA))
            if equals:
                print(f"[OK] non-numeric {col}")
                n_equal += 1
            else:
                ok = False
                n_diff += 1
                # Find where they differ
                diff_mask = (s1n.fillna("##NA##").values != s2n.fillna("##NA##").values)
                n_bad = int(diff_mask.sum())
                print(f"[DIFF] non-numeric {col}: {n_bad} values differ.")
                # show sample rows
                idxs = np.where(diff_mask)[0][:samples]
                for i in idxs:
                    idx_label = df1.index[i] if index_col is not None else i
                    v1 = s1n.iloc[i]
                    v2 = s2n.iloc[i]
                    print(f"  example @{idx_label}: file1={v1!r} file2={v2!r}")

    print("\n=== SUMMARY ===")
    print(f"Columns compared: {ncols} | OK: {n_equal} | DIFF: {n_diff}")
    print("Result:", "✅ identical (within tolerance)" if ok else "⚠️ differ")
    return ok

def main():
    parser = argparse.ArgumentParser(
        description="Compare two zipped CSV DataFrames, ignoring small numeric differences."
    )
    parser.add_argument("file1", help="Path to first .zip (CSV inside)")
    parser.add_argument("file2", help="Path to second .zip (CSV inside)")
    parser.add_argument("--index-col", help="Column to align rows by (e.g., location_id)", default=None)
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for numeric comparison")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for numeric comparison")
    parser.add_argument("--samples", type=int, default=5, help="Max differing examples to print per column")
    parser.add_argument("--no-equal-nan", action="store_true", help="Treat NaNs as different (default: equal)")
    parser.add_argument("--strip", action="store_true", help="Strip whitespace for non-numeric columns before compare")
    args = parser.parse_args()

    try:
        df1 = load_csv_zip(args.file1)
        df2 = load_csv_zip(args.file2)
    except Exception as e:
        print(f"❌ Failed to load files: {e}")
        sys.exit(2)

    equal_nan = not args.no_equal_nan

    ok = compare_dataframes(
        df1, df2,
        index_col=args.index_col,
        rtol=args.rtol,
        atol=args.atol,
        equal_nan=equal_nan,
        strip=args.strip,
        samples=args.samples,
    )

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
