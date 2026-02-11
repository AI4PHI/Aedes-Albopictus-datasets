#!/bin/bash
set -euo pipefail

# ─── Counter Pipeline Test Runner ────────────────────────────────────────────
#
# Usage:
#   bash run_tests.sh              # Run all comparison tests
#   bash run_tests.sh albopictus   # Only albopictus stage
#   bash run_tests.sh climate      # Only climate stage
#   bash run_tests.sh eggs         # Only eggs stage
#   bash run_tests.sh --help       # Show help
#
# Prerequisites:
#   - Reference files must exist in tests/data/
#   - Pipeline outputs must exist in output_data/
#     (run make_counter_dataset.sh first)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Counter Pipeline — Test Runner"
echo "=============================================="
echo "  Working dir: $SCRIPT_DIR"
echo ""

# Check that reference data directory exists
if [ ! -d "data" ]; then
    echo "❌  tests/data/ directory not found."
    echo "    Create it and add reference pickle files:"
    echo "      data/albopictus_test.pkl"
    echo "      data/albopictus_with_climate_3m_test.pkl  (optional)"
    echo "      data/eggs_y_norm_test.pkl                 (optional)"
    echo "      data/eggs_y_norm_7days_test.pkl           (optional)"
    echo "      data/eggs_y_norm_14days_test.pkl          (optional)"
    exit 1
fi

# Check for at least the base reference file
if [ ! -f "data/albopictus_test.pkl" ]; then
    echo "⚠️   data/albopictus_test.pkl not found — albopictus test will be skipped."
fi

# Run the test script, forwarding all arguments
STAGE="${1:-all}"

echo "Running stage: $STAGE"
echo ""

if [ "$STAGE" = "--help" ] || [ "$STAGE" = "-h" ]; then
    python test_pipeline.py --help
    exit 0
fi

python test_pipeline.py --stage "$STAGE" --compare-common

echo ""
echo "Tests complete."
