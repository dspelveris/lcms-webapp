"""Configuration defaults for LC-MS Web App."""

import os

def get_default_path():
    """Get a sensible default path for LC-MS data."""
    # Check environment variable first
    env_path = os.environ.get("LCMS_BASE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Common LC-MS data locations
    candidates = [
        "/Volumes/chab_loc_lang_s1",  # macOS network drive
        os.path.expanduser("~/Documents"),  # User documents
        os.path.expanduser("~"),  # Home directory
        "/data/lcms",  # Docker default
        "C:\\",  # Windows root
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return os.path.expanduser("~")  # Fallback to home

# Base path for LC-MS data
BASE_PATH = get_default_path()

# UV wavelength to display (nm)
UV_WAVELENGTH = 280

# Smoothing parameters
UV_SMOOTHING_WINDOW = 36
EIC_SMOOTHING_WINDOW = 5

# Default m/z values for DS-030 analysis
DEFAULT_MZ_VALUES = [680.1, 382.98, 243.99, 321.15]

# Default m/z window for EIC extraction
DEFAULT_MZ_WINDOW = 0.5

# Export settings
EXPORT_DPI = 300

# Plot colors for time progression
TIME_COLORS = {
    "initial": "#808080",  # Grey
    "mid": "#1f77b4",      # Blue
    "final": "#d62728",    # Red
}

# Plot color cycle for multiple EICs
EIC_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
]
