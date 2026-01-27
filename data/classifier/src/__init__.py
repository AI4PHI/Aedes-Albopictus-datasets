"""
Source modules for data processing and analysis.
"""

try:
    from . import copernicus
    from . import aedes_suitability
    from . import unified_climate_downloader
    __all__ = ['copernicus', 'aedes_suitability', 'unified_climate_downloader']
except ImportError:
    # Handle case where modules might not be available
    __all__ = []