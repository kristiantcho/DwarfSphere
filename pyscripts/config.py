"""
Configuration file for DwarfSphere package.
Contains settings for file paths, simulation parameters, and analysis options.
"""

import os

# File paths
SNAPSHOT_DIR = os.environ.get('DSPH_SNAPSHOT_DIR', '/path/to/snapshots')
FOF_DIR = os.environ.get('DSPH_FOF_DIR', '/path/to/fof_files')
RESULTS_DIR = os.environ.get('DSPH_RESULTS_DIR', '/path/to/results')

# Simulation parameters
MILKY_WAY_MASS = 1.3e12  # Solar masses
HUBBLE_PARAMETER = 1.0

# MW potential scaling factors
MW_POTENTIAL_FACTORS = {
    'light': 1.0,   # Standard Bovy potential
    'heavy': 2.0,   # Bovy16 (2x standard)
    # Add other factors as needed
}

# Cosmology parameters
COSMOLOGY_PARAMS = {
    'H0': 100.0,  # Hubble constant in km/s/Mpc
    'Om0': 0.315,  # Matter density
    'Ode0': 0.685  # Dark energy density
}

# Analysis parameters
ANALYSIS_PARAMS = {
    'bin_size': 31,
    'use_counts': True,
    'use_log': False,
    'density_threshold': 250,
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'max_distance': 1000,  # kpc
    'frame_time': 150,     # ms
    'animation_format': 'imagemagick',
}

"""
Example usage of a custom potential factor:

To use a custom potential factor, you can add an entry to MW_POTENTIAL_FACTORS,
or specify the factor directly in your analysis script:

```python
import config
from dm_tools import find_pot_tidal_r

# Using a predefined factor
pot_factor = config.MW_POTENTIAL_FACTORS['heavy']

# Or using a custom factor
custom_factor = 1.5

# Use the factor in your analysis
# ...
```
"""