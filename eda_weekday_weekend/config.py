"""Centralized configuration for the weekday/weekend EDA package."""
from __future__ import annotations

from pathlib import Path

# Base paths
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "upload" / "index_1.csv"

# Day type ordering and color mapping
DAY_TYPE_ORDER = ["Weekday", "Weekend", "Holiday"]
DAY_TYPE_COLORS = ["#3498db", "#2ecc71", "#e74c3c"]
DAY_TYPE_COLOR_MAP = dict(zip(DAY_TYPE_ORDER, DAY_TYPE_COLORS))

# Plotting defaults
FIG_SIZE = (14, 6)
FIG_SIZE_WIDE = (16, 6)
FIG_SIZE_TRIPLE = (18, 6)


def resolve_data_path(path: str | Path | None = None) -> Path:
    """
    Return a resolved path to the transactions csv.
    
    Args:
        path: Optional path to data file. If None, uses DEFAULT_DATA_PATH
        
    Returns:
        Resolved Path object pointing to the transactions csv
    """
    if path is None:
        return DEFAULT_DATA_PATH
    return Path(path).expanduser().resolve()

