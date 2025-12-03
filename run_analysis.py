"""
Runner script for coffee sales analysis.
Can be executed from the project root directory.
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from promotional_analysis.promotional_analysis_main import promotional_analsysis_main
from eda_milk_ratio.eda_milk_ratio_main import eda_milk_main
from user_analysis import user_analysis_main
from eda_weekday_weekend import eda_weekday_weekend_main

if __name__ == "__main__":
    eda_milk_main()
    eda_weekday_weekend_main()
    promotional_analsysis_main()
    user_analysis_main()

