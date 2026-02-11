#!/usr/bin/env python
"""
Pipeline output comparison tests.

Compares:
  1. albopictus.pkl (pipeline output) vs albopictus_test.pkl (reference)
  2. albopictus_with_climate_3m.pkl vs reference (if available)
  3. eggs_y_norm*.pkl (created from climate output) vs references (if available)

Usage:
    # Run all available tests
    python test_pipeline.py

    # Run only albopictus comparison
    python test_pipeline.py --stage albopictus

    # Run only climate comparison
    python test_pipeline.py --stage climate

    # Run only eggs comparison (will create eggs files first)
    python test_pipeline.py --stage eggs

    # Compare only common columns (useful when column sets differ)
    python test_pipeline.py --compare-common

    # Specify custom paths
    python test_pipeline.py --output-pkl path/to/output.pkl --reference-pkl path/to/ref.pkl
"""

import sys
import argparse
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure test helpers are importable
TESTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent / "src"))

from conftest import (
    COUNTER_DIR, SRC_DIR, OUTPUT_DIR, TEST_DATA_DIR,
    REF_ALBOPICTUS, REF_CLIMATE, REF_EGGS_NORM, REF_EGGS_7DAYS, REF_EGGS_14DAYS,
    OUT_ALBOPICTUS, OUT_CLIMATE, OUT_EGGS_NORM, OUT_EGGS_7DAYS, OUT_EGGS_14DAYS,
)
from deep_numeric_comparison import (
    deep_compare_dataframes, compare_columns, check_index_differences,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_pkl(path: Path, label: str) -> pd.DataFrame:
    """Load a pickle file or raise with a clear message."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_pickle(path)
    print(f"  Loaded {label}: {df.shape[0]} rows × {df.shape[1]} cols  ({path.name})")
    return df


def _run_comparison(output_path: Path, reference_path: Path,
                    label: str, compare_common: bool = False) -> bool:
    """
    Run a full comparison between an output and a reference pickle file.

    Returns True if data matches (within tolerance), False otherwise.
    """
    print(f"\n{'='*70}")
    print(f"  COMPARING: {label}")
    print(f"{'='*70}")
    print(f"  Output:    {output_path}")
    print(f"  Reference: {reference_path}")
    print()

    df_out = _load_pkl(output_path, "Output")
    df_ref = _load_pkl(reference_path, "Reference")

    # Index check
    check_index_differences(df_out, df_ref)
    print()

    # Column check
    common, only_out, only_ref = compare_columns(df_out, df_ref)
    print()

    # Decide whether to force common-only comparison
    force_common = compare_common or bool(only_out) or bool(only_ref)
    if force_common and not compare_common:
        print("⚠️  Column sets differ — automatically comparing common columns only.\n")

    # Deep value comparison
    is_equal, diff_cols = deep_compare_dataframes(
        df_out, df_ref,
        name1="Output", name2="Reference",
        compare_common_only=force_common,
    )

    # Summary
    print(f"\n{'─'*70}")
    if is_equal:
        print(f"✅  {label}: PASSED — data matches reference")
    else:
        print(f"❌  {label}: FAILED — {len(diff_cols)} column(s) with differences")
        for d in diff_cols:
            col = d.get("column", "?")
            dtype = d.get("type", "?")
            n = d.get("n_diffs", "?")
            print(f"     • {col}: {dtype} ({n} diffs)")
    print(f"{'─'*70}")

    return is_equal


# ── stage runners ────────────────────────────────────────────────────────────

def test_albopictus(compare_common: bool = False,
                    output_pkl: Path = None, reference_pkl: Path = None) -> bool:
    """Compare albopictus.pkl against reference."""
    out = output_pkl or OUT_ALBOPICTUS
    ref = reference_pkl or REF_ALBOPICTUS
    return _run_comparison(out, ref, "Albopictus base data", compare_common)


def test_climate(compare_common: bool = False,
                 output_pkl: Path = None, reference_pkl: Path = None) -> bool:
    """Compare albopictus_with_climate_3m.pkl against reference."""
    out = output_pkl or OUT_CLIMATE
    ref = reference_pkl or REF_CLIMATE
    return _run_comparison(out, ref, "Climate-enriched data", compare_common)


def test_eggs(compare_common: bool = False,
              climate_pkl: Path = None) -> dict:
    """
    Create eggs normalized datasets from climate output, then compare
    each against its reference.

    Returns dict of {name: passed_bool}.
    """
    climate_input = climate_pkl or OUT_CLIMATE

    if not climate_input.exists():
        print(f"❌  Cannot run eggs test — climate file not found: {climate_input}")
        return {"eggs_norm": False, "eggs_7days": False, "eggs_14days": False}

    # Generate eggs datasets
    print(f"\n{'='*70}")
    print("  GENERATING EGGS NORMALIZED DATASETS")
    print(f"{'='*70}")

    try:
        from create_eggs_dataset import create_eggs_dataset
        create_eggs_dataset(
            input_path=str(climate_input),
            output_dir=str(TESTS_DIR),
        )
        print("✅  Eggs datasets created successfully\n")
    except Exception as e:
        print(f"❌  Failed to create eggs datasets: {e}")
        traceback.print_exc()
        return {"eggs_norm": False, "eggs_7days": False, "eggs_14days": False}

    # Compare each eggs file
    results = {}

    pairs = [
        ("eggs_norm", OUT_EGGS_NORM, REF_EGGS_NORM, "Eggs normalized (full)"),
        ("eggs_7days", OUT_EGGS_7DAYS, REF_EGGS_7DAYS, "Eggs normalized (7-day)"),
        ("eggs_14days", OUT_EGGS_14DAYS, REF_EGGS_14DAYS, "Eggs normalized (14-day)"),
    ]

    for key, out_path, ref_path, label in pairs:
        if not out_path.exists():
            print(f"⚠️  Output not found, skipping: {out_path.name}")
            results[key] = False
            continue
        if not ref_path.exists():
            print(f"⚠️  Reference not found, skipping: {ref_path.name}")
            results[key] = None  # None = not testable
            continue
        results[key] = _run_comparison(out_path, ref_path, label, compare_common)

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline outputs against reference pickle files"
    )
    parser.add_argument(
        "--stage",
        choices=["albopictus", "climate", "eggs", "all"],
        default="all",
        help="Which comparison stage(s) to run (default: all)",
    )
    parser.add_argument(
        "--compare-common",
        action="store_true",
        help="Only compare columns present in both files",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path, default=None,
        help="Override output pickle path (for albopictus/climate stage)",
    )
    parser.add_argument(
        "--reference-pkl",
        type=Path, default=None,
        help="Override reference pickle path (for albopictus/climate stage)",
    )
    parser.add_argument(
        "--climate-pkl",
        type=Path, default=None,
        help="Override climate pickle used as input for eggs creation",
    )
    args = parser.parse_args()

    print("="*70)
    print("  COUNTER PIPELINE — OUTPUT COMPARISON TESTS")
    print("="*70)
    print(f"  Test data dir:  {TEST_DATA_DIR}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print()

    # Track overall results
    all_results = {}
    stages = [args.stage] if args.stage != "all" else ["albopictus", "climate", "eggs"]

    # ── 1. Albopictus ────────────────────────────────────────────────────
    if "albopictus" in stages:
        try:
            all_results["albopictus"] = test_albopictus(
                compare_common=args.compare_common,
                output_pkl=args.output_pkl,
                reference_pkl=args.reference_pkl,
            )
        except FileNotFoundError as e:
            print(f"⚠️  Skipping albopictus test: {e}")
            all_results["albopictus"] = None

    # ── 2. Climate ───────────────────────────────────────────────────────
    if "climate" in stages:
        try:
            all_results["climate"] = test_climate(
                compare_common=args.compare_common,
                output_pkl=args.output_pkl if args.stage == "climate" else None,
                reference_pkl=args.reference_pkl if args.stage == "climate" else None,
            )
        except FileNotFoundError as e:
            print(f"⚠️  Skipping climate test: {e}")
            all_results["climate"] = None

    # ── 3. Eggs ──────────────────────────────────────────────────────────
    if "eggs" in stages:
        try:
            eggs_results = test_eggs(
                compare_common=args.compare_common,
                climate_pkl=args.climate_pkl,
            )
            all_results.update(eggs_results)
        except Exception as e:
            print(f"⚠️  Eggs test failed: {e}")
            traceback.print_exc()
            all_results["eggs_norm"] = False

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  FINAL TEST SUMMARY")
    print(f"{'='*70}\n")

    passed = 0
    failed = 0
    skipped = 0

    for name, result in all_results.items():
        if result is True:
            print(f"  ✅  {name}")
            passed += 1
        elif result is False:
            print(f"  ❌  {name}")
            failed += 1
        else:
            print(f"  ⏭️   {name} (skipped — reference or output missing)")
            skipped += 1

    print(f"\n  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
    print(f"{'='*70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
