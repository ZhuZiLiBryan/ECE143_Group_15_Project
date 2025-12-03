"""Plotting style configuration."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set default plotting style
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def init_style() -> None:
    """
    Initialize matplotlib and seaborn plotting styles.
    
    Args:
        None
        
    Returns:
        None. Configures global plotting defaults
    """
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

