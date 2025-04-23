# DwarfSphere

A set of python scripts for analyzing dwarf spheroidal galaxy simulations and their dark and stellar matter content. This code was used to produce many of the key results in the Broken Expectations paper (please cite this if you use this code!)

Paper: (coming soon)

## Overview

DwarfSphere is a comprehensive toolkit for analyzing dark matter dynamics, density profiles, and mass distributions in simulated dwarf spheroidal galaxies. It provides tools for:

- Calculating tidal radii and bound regions
- Fitting density profiles using various models
- Computing velocity dispersions and mass estimates with the Wolf estimator
- Performing J-factor calculations for dark matter analyses
- Creating visualizations of mass distributions and orbits

## Scripts

### Core Functionality

- **dm_tools.py**: Core utilities module containing functions for calculating tidal radii, density profiles, velocity dispersions, and various astrophysical properties. Acts as the foundation for the other analysis scripts.
- **config.py**: Configuration file with default settings for file paths, simulation parameters, and analysis options.

### Analysis Scripts

- **wolf_cal_dm.py**: Implements the Wolf mass estimator for dark matter halos, which uses the velocity dispersion at the r3 radius to estimate enclosed mass.
- **wolf_cal_stars.py**: Similar to wolf_cal_dm.py but specialized for stellar components.
- **jeans_j_factor_cal.py**: Calculates J-factors from densities calculated with (py)GravSphere.
- **true_j_factor_cal.py**: Computes true J-factors directly from simulation data.
- **ecc_cal.py**: Tools for analyzing eccentric orbits of dwarf galaxies around their host galaxies.

### Example Configuration Files

- **wolf_config.yaml**: Configuration for Wolf mass estimator calculations
- **jfactor_config.yaml**: Configuration for J-factor calculations
- **true_jfactor_config.yaml**: Configuration for true J-factor calculations
- **ecc_config.yaml**: Configuration for orbit eccentricity calculations

## Key Features

- **Robust Mass Estimators**: Implementation of the Wolf et al. (2010) estimator for dark matter mass
- **Profile Fitting**: Tools to fit density profiles (Plummer & Zhao) to simulation data
- **Tidal Evolution**: Analysis of tidal effects on dwarf galaxies in host potentials
- **Velocity Dispersion**: Advanced methods to calculate velocity dispersion profiles
- **J-Factor Calculations**: Tools for computing dark matter annihilation J-factors
- **Dynamical Analysis**: Functions to analyze orbital dynamics and structural evolution

## Getting Started

1. Set up the required environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the package by updating values in `config.py` or creating configuration YAML files for specific analyses.

3. Run an analysis (example for Wolf mass estimator):
   ```bash
   python wolf_cal_dm.py wolf_config.yaml
   ```

## Required Data Format

The package expects simulation data in the following formats from Gadget4:
- HDF5 snapshot files containing particle positions, velocities, and other properties
- Subfind + Friends-of-friends (FOF) files for halo identification (optional for some analyses)
In the case of **jeans_j_factor_cal.py**, it expects pyGravSphere MCMC chains files.

## Configuration

You can configure the package in multiple ways:
1. Edit default values in `config.py`
2. Set environment variables (e.g., `DSPH_SNAPSHOT_DIR`)
3. Create specific YAML configuration files for individual analyses

## Dependencies

See `requirements.txt` for a complete list of dependencies.