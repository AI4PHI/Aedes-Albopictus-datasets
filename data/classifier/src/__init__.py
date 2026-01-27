"""
Source modules for data processing and analysis.
"""

try:
    from . import copernicus
    from . import aedes_suitability
    __all__ = ['copernicus', 'aedes_suitability']
except ImportError:
    # Handle case where modules might not be available
    __all__ = []