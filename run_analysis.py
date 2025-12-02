"""
Runner script for coffee sales analysis.
Can be executed from the project root directory.
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from scripts.main import main

if __name__ == "__main__":
    main()

