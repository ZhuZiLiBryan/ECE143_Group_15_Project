"""
Configuration loader for coffee sales analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, looks for config.json in project root.
        
    Returns:
        Dictionary with configuration settings
    """
    if config_path is None:
        # Look for config.json in project root (parent of scripts directory)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_profit_margins(config: Optional[Dict[str, Any]] = None, 
                       coffee_names: Optional[list] = None) -> Dict[str, float]:
    """
    Get profit margins from config, with fallback to defaults.
    
    Args:
        config: Configuration dictionary. If None, loads from config.json
        coffee_names: List of coffee names to ensure all are covered
        
    Returns:
        Dictionary mapping coffee names to profit margins
    """
    if config is None:
        config = load_config()
    
    profit_margins = config.get('profit_margins', {})
    default_margin = profit_margins.pop('default', 2.0)
    
    # Ensure all coffee names have profit margins
    if coffee_names:
        for name in coffee_names:
            if name not in profit_margins:
                profit_margins[name] = default_margin
    
    return profit_margins

