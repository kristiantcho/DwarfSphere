import numpy as np
import os
import glob
import h5py
import sys
import argparse
import yaml
sys.path.insert(1, '/ptmp/mpa/kristcho/Sim-DSph-M-Estimator')
import dm_tools as dm


def load_config(config_file):
    """
    Load configuration from file
    
    Parameters:
    -----------
    config_file : str
        Path to configuration file (YAML)
        
    Returns:
    --------
    dict : Configuration dictionary
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Check for required sections
    required_sections = ['file_paths', 'run_parameters']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section '{section}' in configuration file")
    
    return cfg


def main():
    """Main function for eccentricity calculation"""
    # Parse command line arguments - only accept config file path
    parser = argparse.ArgumentParser(description="Eccentricity calculator for dark matter simulations")
    parser.add_argument('config_file', type=str, help='Path to configuration file (YAML)')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config_file)
    
    # Get parameters from config
    dsph_name = cfg['run_parameters']['dsph_name']
    mw_type = cfg['run_parameters']['mw_type']
    dm_particle_type_id = cfg['run_parameters'].get('dm_particle_type_id', 3)
    stellar_particle_type_id = cfg['run_parameters'].get('stellar_particle_type_id', 1)
    filter_mod = cfg['run_parameters'].get('filter_mode', 'bound')
    
    # Get file paths from config
    orbit_base_dir = cfg['file_paths']['orbit_base_dir']
    
    # Print configuration info
    print(f"Running eccentricity calculation for {dsph_name}")
    print(f"Using MW potential type: {mw_type}")
    
    # Setup paths
    output_orbit_dir = os.path.join(orbit_base_dir, "ecc_orbits", dsph_name, f"output_orbit_{mw_type}")
    fits_dir = os.path.join(orbit_base_dir, "ecc_orbits", dsph_name, "fits")
    
    # Find snapshot and FOF files
    snap_files = sorted(glob.glob(os.path.join(output_orbit_dir, "snapshot_*.hdf5")))[:]
    fof_files = sorted(glob.glob(os.path.join(output_orbit_dir, "fof_*.hdf5")))[:]

    if not snap_files or not fof_files:
        raise FileNotFoundError(f"No snapshot or FOF files found in {output_orbit_dir}")
    
    # Set up output path for results
    # Use custom final_fit_path if provided in config, otherwise construct it
    if 'final_fit_path' in cfg['file_paths']:
        final_fit_path = cfg['file_paths']['final_fit_path']
        print(f"Using specified output path: {final_fit_path}")
        # Create the output directory if it doesn't exist
        os.makedirs(final_fit_path, exist_ok=True)
    else:
        # Find next available run number
        os.makedirs(fits_dir, exist_ok=True)
        
        counts = [-1]
        try:
            for direc in glob.glob(f'{fits_dir}/{mw_type}_wolf_fit_ecc_{filter_mod}_*'):
                new_count = direc.split('_')[-1]
                counts.append(int(new_count))
            count = max(counts) + 1
            final_fit_path = f'{fits_dir}/{mw_type}_wolf_fit_ecc_{filter_mod}_{count}'
        except Exception as e:
            print(f"Error finding run number: {e}")
            final_fit_path = f'{fits_dir}/{mw_type}_wolf_fit_ecc_{filter_mod}_0'
        
        os.makedirs(final_fit_path, exist_ok=True)
    
    print(f"Saving results in: {final_fit_path}")
    
    # Initialize arrays for data collection
    minor_2d = []
    major_2d = []
    minor_3d = []
    major_3d = []
    minor_stel_2d = []
    major_stel_2d = []
    minor_stel_3d = []
    major_stel_3d = []
    fcorots = []
    kapparots = []
    stel_fcorots = []
    stel_kapparots = []
    times = []
    stel_bound = []
    bound = []
    
    # Process each snapshot
    for i in range(len(snap_files)):
        print(f"Processing snapshot {i}", flush=True)
        
        try:
            with h5py.File(snap_files[i], 'r') as hf:
                raw_coords = np.array(hf[f'PartType{dm_particle_type_id}']['Coordinates'])
                raw_vels = np.array(hf[f'PartType{dm_particle_type_id}']['Velocities'])
                time = hf['Header'].attrs['Time']
                mass = hf['Header'].attrs['MassTable'][dm_particle_type_id] * 1e10
                raw_stel_coords = np.array(hf[f'PartType{stellar_particle_type_id}']['Coordinates'])
                raw_stel_vels = np.array(hf[f'PartType{stellar_particle_type_id}']['Velocities'])
                stel_mass = hf['Header'].attrs['MassTable'][stellar_particle_type_id] * 1e10
                part_ids = np.array(hf[f'PartType{dm_particle_type_id}']['ParticleIDs'])
                stel_part_ids = np.array(hf[f'PartType{stellar_particle_type_id}']['ParticleIDs'])
        except Exception as e:
            print(f"Error reading snapshot {i}: {e}")
            continue
            
        try:    
            with h5py.File(fof_files[i], 'r') as hf:
                part_len = hf['Subhalo']['SubhaloLenType'][0][dm_particle_type_id]
                stel_part_len = hf['Subhalo']['SubhaloLenType'][0][stellar_particle_type_id]
                SF_offset = hf['Subhalo']['SubhaloOffsetType'][0][dm_particle_type_id]
                stel_SF_offset = hf['Subhalo']['SubhaloOffsetType'][0][stellar_particle_type_id]
                SF_com = np.array(hf['Subhalo']['SubhaloCM'][0])
                SF_vel = np.array(hf['Subhalo']['SubhaloVel'][0])
        except Exception as e:
            print(f"No Subhalo in file {i}: {e}")
            continue
        
        # Track particles across snapshots
        if i == 0:
            mask = np.isin(part_ids, part_ids[SF_offset:SF_offset + part_len])
            stel_mask = np.isin(stel_part_ids, stel_part_ids[stel_SF_offset:stel_SF_offset + stel_part_len])
        else:
            new_mask = np.isin(part_ids, part_ids[SF_offset:SF_offset + part_len])
            new_stel_mask = np.isin(stel_part_ids, stel_part_ids[stel_SF_offset:stel_SF_offset + stel_part_len])
            mask = np.logical_and(np.isin(part_ids, old_ids), new_mask)
            stel_mask = np.logical_and(np.isin(stel_part_ids, stel_old_ids), new_stel_mask)
        
        # Get positions and velocities in center-of-mass frame
        new_pos = raw_coords[mask]
        new_vels = raw_vels[mask]
        new_stel_pos = raw_stel_coords[stel_mask]
        new_stel_vels = raw_stel_vels[stel_mask]
        old_ids = part_ids[mask]
        stel_old_ids = stel_part_ids[stel_mask]
        
        com_pos = new_pos - SF_com
        com_vels = new_vels - SF_vel
        stel_com_pos = new_stel_pos - SF_com
        stel_com_vels = new_stel_vels - SF_vel
        
        # Calculate eccentricity metrics
        try:
            ellipsoid_3d = dm.sphericity3d(com_pos, mass)
            ellipsoid_2d = dm.sphericity2d(com_pos[:,:2], mass)
            ellipsoid_stel_3d = dm.sphericity3d(stel_com_pos, stel_mass)
            ellipsoid_stel_2d = dm.sphericity2d(stel_com_pos[:,:2], stel_mass)
            fcorot, kapparot = dm.support(com_pos, com_vels, mass)
            fcorot_stel, kapparot_stel = dm.support(stel_com_pos, stel_com_vels, stel_mass)
        except Exception as e:
            print(f"Error calculating metrics at snapshot {i}: {e}")
            continue
        
        # Store results
        fcorots.append(fcorot)
        kapparots.append(kapparot)
        stel_fcorots.append(fcorot_stel)
        stel_kapparots.append(kapparot_stel)
        minor_3d.append(np.min(ellipsoid_3d))
        major_3d.append(np.max(ellipsoid_3d))
        minor_stel_3d.append(np.min(ellipsoid_stel_3d))
        major_stel_3d.append(np.max(ellipsoid_stel_3d))
        minor_2d.append(np.min(ellipsoid_2d))
        major_2d.append(np.max(ellipsoid_2d))
        minor_stel_2d.append(np.min(ellipsoid_stel_2d))
        major_stel_2d.append(np.max(ellipsoid_stel_2d))
        bound.append(len(new_pos))
        stel_bound.append(len(new_stel_pos))
        times.append(time)
        
        # Save results after each snapshot
        np.save(os.path.join(final_fit_path, 'minor_3d_dm.npy'), minor_3d)
        np.save(os.path.join(final_fit_path, 'major_3d_dm.npy'), major_3d)
        np.save(os.path.join(final_fit_path, 'minor_2d_dm.npy'), minor_2d)
        np.save(os.path.join(final_fit_path, 'major_2d_dm.npy'), major_2d)
        np.save(os.path.join(final_fit_path, 'minor_3d_stars.npy'), minor_stel_3d)
        np.save(os.path.join(final_fit_path, 'major_3d_stars.npy'), major_stel_3d) 
        np.save(os.path.join(final_fit_path, 'minor_2d_stars.npy'), minor_stel_2d)
        np.save(os.path.join(final_fit_path, 'major_2d_stars.npy'), major_stel_2d)
        np.save(os.path.join(final_fit_path, 'fcorots_dm.npy'), fcorots)
        np.save(os.path.join(final_fit_path, 'kapparots_dm.npy'), kapparots)
        np.save(os.path.join(final_fit_path, 'fcorots_stars.npy'), stel_fcorots)
        np.save(os.path.join(final_fit_path, 'kapparots_stars.npy'), stel_kapparots)
        np.save(os.path.join(final_fit_path, 'last_ids.npy'), old_ids)
        np.save(os.path.join(final_fit_path, 'stel_last_ids.npy'), stel_old_ids)
        np.save(os.path.join(final_fit_path, 'times.npy'), times)
        np.save(os.path.join(final_fit_path, 'bound_parts.npy'), bound)
        np.save(os.path.join(final_fit_path, 'stel_bound_parts.npy'), stel_bound)
    
    print('Calculation completed successfully!')


if __name__ == "__main__":
    main()



