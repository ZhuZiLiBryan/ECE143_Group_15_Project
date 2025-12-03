"""Centralized configuration for the user model pipeline package."""
from __future__ import annotations

from pathlib import Path

# Base paths
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "upload" / "index_1.csv"

# Modeling defaults
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Plotting defaults
PLOT_STYLE = "whitegrid"
FIG_SIZE = (12, 8)


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

