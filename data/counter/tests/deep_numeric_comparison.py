"""Re-export from tests/src/ so test_pipeline.py can import at package level."""
# filepath: /home/biazzin/git/AIedes_data/data/counter/tests/deep_numeric_comparison.py
from src.deep_numeric_comparison import (
    deep_compare_dataframes,
    compare_columns,
    check_index_differences,
)

__all__ = [
    "deep_compare_dataframes",
    "compare_columns",
    "check_index_differences",
]
