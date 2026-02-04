"""Configuration defaults for LC-MS Web App."""

import os

# Base path for LC-MS data (mounted network drive)
BASE_PATH = os.environ.get("LCMS_BASE_PATH", "/data/lcms")

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
