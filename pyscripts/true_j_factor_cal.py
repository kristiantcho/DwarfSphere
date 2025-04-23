import numpy as np
import os
import glob
import h5py
import copy
import argparse
import yaml
from galpy.orbit import Orbit
import galpy.potential as gp
import astropy.units as u
from scipy.signal import find_peaks
from scipy import spatial


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
    required_sections = ['file_paths', 'run_parameters', 'galaxy_distances']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section '{section}' in configuration file")
    
    return cfg


def get_mw_potential(potential_type, potential_factor=None):
    """
    Get the Milky Way potential based on configuration
    
    Parameters:
    -----------
    potential_type : str
        Type of potential to use (light, heavy, custom)
    potential_factor : float, optional
        Scaling factor for custom potential
        
    Returns:
    --------
    galpy.potential : Galpy potential object
    """
    # Get potential configuration
    if potential_type == 'light':
        # Standard Bovy potential
        scaling_factor = 1.0
    elif potential_type == 'heavy':
        # Bovy potential with factor of 2
        scaling_factor = 2.0
    elif potential_type == 'bovy16':
        # Bovy potential with factor of 1.6/0.8
        scaling_factor = 1.6/0.8
    elif potential_type == 'custom':
        # Custom potential with user-specified factor
        if potential_factor is None:
            raise ValueError("For 'custom' potential type, 'potential_factor' must be specified in the config")
        scaling_factor = float(potential_factor)
    else:
        # Default to standard Bovy potential
        scaling_factor = 1.0
    
    # Create the MW potential
    pot = copy.deepcopy(gp.MWPotential2014)
    pot[2] *= scaling_factor  # Scale the dark matter halo component
    
    return pot


def main():
    """Main function for calculating true J-factor"""
    # Parse command line arguments - only accept config file path
    parser = argparse.ArgumentParser(description="True J-factor calculator for dark matter simulations")
    parser.add_argument('config_file', type=str, help='Path to configuration file (YAML)')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config_file)
    
    # Get parameters from config
    dsph_name = cfg['run_parameters']['dsph_name']
    galpy_name = cfg['run_parameters']['galpy_name']
    mw_type = cfg['run_parameters']['mw_type']
    snap_time = cfg['run_parameters']['snap_time']
    part_type = cfg['run_parameters'].get('particle_type', 'dm')
    extras = cfg['run_parameters'].get('extras', '')
    particle_type_id = cfg['run_parameters'].get('particle_type_id', 3)
    stellar_type_id = cfg['run_parameters'].get('stellar_type_id', 1)
    
    # Get file paths from config
    orbit_base_dir = cfg['file_paths']['orbit_base_dir']
    results_dir = cfg['file_paths']['results_dir']
    
    # Get galaxy distance
    gal_dist = cfg['galaxy_distances'].get(dsph_name, None)
    if gal_dist is None:
        raise ValueError(f"Galaxy distance for {dsph_name} not found in config file")
    
    # Print configuration info
    print(f"Running true J-factor calculation for {galpy_name}")
    print(f"Using MW potential type: {mw_type}")
    print(f"Snapshot time: {snap_time}")
    
    # Setup paths
    fits_dir = os.path.join(orbit_base_dir, dsph_name, "fits")
    output_orbit_dir = os.path.join(orbit_base_dir, dsph_name, f"output_orbit_{extras}_{mw_type}")
    
    # Ensure output directory exists
    os.makedirs(os.path.join(results_dir, "masks"), exist_ok=True)
    
    # Use custom final_fit_path if provided in config, otherwise construct it
    if 'final_fit_path' in cfg['file_paths']:
        final_fit_path = cfg['file_paths']['final_fit_path']
        print(f"Using specified Wolf fit path: {final_fit_path}")
    else:
        # Find the latest wolf_fit results
        final_fit_paths = sorted(glob.glob(f'{fits_dir}/{mw_type}_wolf_fit_bound_{extras}_{part_type}_*'))
        if not final_fit_paths:
            raise FileNotFoundError(f"No Wolf fit results found in {fits_dir}")
        
        counts = []
        for path in final_fit_paths:
            counts.append(int(path.split('_')[-1]))
        final_count = max(counts)
        print(f"Using Wolf fit results from run {final_count}")
        
        final_fit_path = f'{fits_dir}/{mw_type}_wolf_fit_bound_{extras}_{part_type}_{final_count}'
    
    # Check if the fit path exists
    if not os.path.exists(final_fit_path):
        raise FileNotFoundError(f"Wolf fit path does not exist: {final_fit_path}")
    
    # Load Wolf fit results
    times = np.load(os.path.join(final_fit_path, 'times.npy'))
    mw_rs = np.load(os.path.join(final_fit_path, 'mw_rs.npy'))
    
    # Setup MW potential
    pot = get_mw_potential(mw_type)
    
    # Calculate orbit
    o = Orbit.from_name(galpy_name, ro=8., vo=220.)
    ts = np.linspace(0., -11., 10000) * u.Gyr
    o.integrate(ts, pot)
    print(f"Orbit radius range: {min(o.r(ts)):.2f} - {max(o.r(ts)):.2f}")
    print(f"Current orbit radius: {o.r(0*u.Gyr):.2f}")
    
    # Convert to non-astropy time array
    ts = np.linspace(0., -11., 10000)
    
    # Find apocenter time
    apo_t = ts[(ts < -7.5) & (ts > -10)][np.argmax(o.r(ts[(ts < -7.5) & (ts > -10)] * u.Gyr))]
    print(f"Apocenter time: {-apo_t:.2f} Gyr")
    
    # Determine velocity sign and normalized target radius
    vel_sign = np.sign(o.vr(0 * u.Gyr))
    print(f"Velocity sign: {vel_sign}")
    
    norm_target_rad = (o.r(0 * u.Gyr) - min(o.r(ts * u.Gyr))) / (max(o.r(ts * u.Gyr)) - min(o.r(ts * u.Gyr)))
    
    # Find peaks and troughs in orbit radius
    peak_inds, _ = find_peaks(mw_rs)
    trough_inds, _ = find_peaks(-mw_rs)
    
    if len(peak_inds) == 0 or len(trough_inds) == 0:
        raise ValueError("Could not find peaks or troughs in orbit data")
    
    apo_r = mw_rs[peak_inds[-1]]
    peri_r = mw_rs[trough_inds[-1]]
    
    # Special case for Fornax with bovy potential
    if (dsph_name == 'fornax' and mw_type == 'bovy') and len(trough_inds) >= 2:
        peri_r = mw_rs[trough_inds[-2]]
    
    # Normalize radii
    normed_rs = (mw_rs - peri_r) / (apo_r - peri_r)
    
    # Find today index
    today_ind = np.argmin(np.abs(times + apo_t))
    
    # Refine today index based on normalized radius
    try:
        if (dsph_name, mw_type) not in [('sculptor', 'bovy16'), ('ursa_min', 'bovy16')]:
            try:
                new_today_ind = today_ind + find_peaks(1 / np.abs(normed_rs[today_ind:] - norm_target_rad))[0][0]
                today_time = times[new_today_ind] + apo_t
                
                # Check velocity sign consistency
                if (mw_rs[new_today_ind + 1] - mw_rs[new_today_ind]) * vel_sign < 0:
                    print('Wrong velocity sign, adjusting index')
                    if dsph_name == 'draco':
                        new_today_ind = today_ind - 5 + find_peaks(1 / np.abs(normed_rs[today_ind - 5:] - norm_target_rad))[0][0]
                    else:
                        new_today_ind = today_ind + find_peaks(1 / np.abs(normed_rs[today_ind:] - norm_target_rad))[0][1]
            except Exception as e:
                print(f'Could not find peaks: {e}')
                new_today_ind = find_peaks(1 / np.abs(normed_rs[:today_ind] - norm_target_rad))[0][-1]
                today_time = times[new_today_ind] + apo_t
                
                if (mw_rs[new_today_ind + 1] - mw_rs[new_today_ind]) * vel_sign < 0:
                    new_today_ind = find_peaks(1 / np.abs(normed_rs[:today_ind] - norm_target_rad))[0][-2]
        else:
            new_today_ind = find_peaks(1 / np.abs(normed_rs[:today_ind] - norm_target_rad))[0][-1]
            today_time = times[new_today_ind] + apo_t
    except Exception as e:
        print(f"Error finding appropriate snapshot: {e}")
        print("Using closest match to apocenter time")
        new_today_ind = today_ind
    
    today_time = times[new_today_ind] + apo_t
    today_ind = new_today_ind
    
    # Find apocenter and pericenter indices
    try:
        peaks, _ = find_peaks(mw_rs[times <= times[today_ind]])
        troughs, _ = find_peaks(-mw_rs[times <= times[today_ind]])
        apo_ind = peaks[-1]
        peri_ind = troughs[-1]
    except Exception:
        peaks, _ = find_peaks(mw_rs)
        troughs, _ = find_peaks(-mw_rs)
        apo_ind = peaks[0]
        peri_ind = troughs[0]
    
    # Find snapshot files
    snap_files = sorted(glob.glob(os.path.join(output_orbit_dir, "snapshot_*.hdf5")))[:]
    fof_files = sorted(glob.glob(os.path.join(output_orbit_dir, "fof_*.hdf5")))[:]
    
    if not snap_files or not fof_files:
        raise FileNotFoundError(f"No snapshot or FOF files found in {output_orbit_dir}")
    
    # Handle missing subhalos in files by adjusting indices
    for i in range(len(snap_files)):
        try:
            with h5py.File(fof_files[i], 'r') as hf:
                SF_com = np.array(hf['Subhalo']['SubhaloCM'][0])
        except Exception:
            print(f"No subhalo in file {i}, adjusting indices")
            if apo_ind >= i:
                apo_ind += 1
            if peri_ind >= i:
                peri_ind += 1
            if today_ind >= i:
                today_ind += 1
    
    # Select snapshot based on chosen time
    if snap_time == 'today':
        index = today_ind
    elif snap_time == 'apo':
        index = apo_ind
    elif snap_time == 'peri':
        index = peri_ind
    else:
        raise ValueError(f"Invalid snap_time: {snap_time}. Use 'today', 'apo', or 'peri'")
    
    print(f"Selected snapshot index: {index}")
    
    # Process particles from snapshot
    try:
        with h5py.File(fof_files[index], 'r') as hf:
            SF_com = np.array(hf['Subhalo']['SubhaloCM'][0])
    except Exception as e:
        raise RuntimeError(f"Error reading subhalo from FOF file: {e}")
    
    com_rad = np.sqrt(np.sum(SF_com * SF_com))
    print(f"Center of mass radius: {com_rad:.2f} kpc")
    
    # Track particles through snapshots to ensure consistency
    for i in range(index + 1):
        try:
            with h5py.File(snap_files[i], 'r') as hf:
                time = hf['Header'].attrs['Time']
                mass = hf['Header'].attrs['MassTable'][particle_type_id] * 1e10
                part_ids = np.array(hf['PartType' + str(particle_type_id)]['ParticleIDs'])
                stel_part_ids = np.array(hf['PartType' + str(stellar_type_id)]['ParticleIDs'])
        except Exception as e:
            print(f"Error reading snapshot {i}: {e}")
            continue
            
        try:    
            with h5py.File(fof_files[i], 'r') as hf:
                part_len = hf['Subhalo']['SubhaloLenType'][0][particle_type_id]
                stel_part_len = hf['Subhalo']['SubhaloLenType'][0][stellar_type_id]
                SF_offset = hf['Subhalo']['SubhaloOffsetType'][0][particle_type_id]
                stel_SF_offset = hf['Subhalo']['SubhaloOffsetType'][0][stellar_type_id]
                SF_com = np.array(hf['Subhalo']['SubhaloCM'][0])
                SF_vel = np.array(hf['Subhalo']['SubhaloVel'][0])
        except Exception:
            print(f'No Subhalo in file {i}')
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
        
        old_ids = part_ids[mask]
        stel_old_ids = stel_part_ids[stel_mask]
    
    # Load final snapshot data for J-factor calculation
    with h5py.File(snap_files[index], 'r') as hf:
        raw_coords = np.array(hf['PartType' + str(particle_type_id)]['Coordinates'])
        raw_vels = np.array(hf['PartType' + str(particle_type_id)]['Velocities'])
        time = hf['Header'].attrs['Time']
        mass = hf['Header'].attrs['MassTable'][particle_type_id] * 1e10
        raw_stel_coords = np.array(hf['PartType' + str(stellar_type_id)]['Coordinates'])
        raw_stel_vels = np.array(hf['PartType' + str(stellar_type_id)]['Velocities'])
        stel_mass = hf['Header'].attrs['MassTable'][stellar_type_id] * 1e10
        part_ids = np.array(hf['PartType' + str(particle_type_id)]['ParticleIDs'])
        stel_part_ids = np.array(hf['PartType' + str(stellar_type_id)]['ParticleIDs'])
    
    with h5py.File(fof_files[index], 'r') as hf:
        part_len = hf['Subhalo']['SubhaloLenType'][0][particle_type_id]
        stel_part_len = hf['Subhalo']['SubhaloLenType'][0][stellar_type_id]
        SF_offset = hf['Subhalo']['SubhaloOffsetType'][0][particle_type_id]
        stel_SF_offset = hf['Subhalo']['SubhaloOffsetType'][0][stellar_type_id]
        SF_com = np.array(hf['Subhalo']['SubhaloCM'][0])
        SF_vel = np.array(hf['Subhalo']['SubhaloVel'][0])
    
    # Calculate local density for J-factor
    new_pos = raw_coords[mask]
    tree = spatial.cKDTree(new_pos)
    densities = []
    for i in range(len(new_pos)):
        d, n = tree.query(new_pos[i], k=32)
        densities.append(32 * mass / (4/3 * np.pi * np.amax(d)**3))
    densities = np.array(densities)
    
    # Calculate J-factor
    observer_vector = SF_com
    observer_vector = observer_vector / np.sqrt(np.sum(observer_vector**2.))
    positions_obs_frame = new_pos
    line_of_sight = np.sum(observer_vector * positions_obs_frame, axis=1)
    R3d = np.sqrt(np.sum(positions_obs_frame**2., axis=1))
    xy = np.sqrt(R3d**2. - line_of_sight**2.)
    theta = 0.5 * 0.0174533  # 0.5 degrees in radians
    dtheta = com_rad * theta
    a, = np.where(xy <= dtheta)
    
    # Calculate J-factor
    jfactor = np.log10(np.sum(densities[a] * mass * (1.12**2 * 1e9 / 3.087**5) / gal_dist**2))
    
    print(f"Calculated J-factor (log10): {jfactor:.4f}")
    
    # Save results
    output_file = os.path.join(results_dir, "true_log10_jfactors.txt")
    with open(output_file, "a+") as file_for_saving:
        file_for_saving.write(f'{galpy_name}-{mw_type}-{snap_time} {jfactor}\n')
    
    indices_file = os.path.join(results_dir, "snap_indices.txt")
    with open(indices_file, "a+") as file_for_saving:
        file_for_saving.write(f'{galpy_name}-{mw_type}-{snap_time} {index}\n')
    
    # Save masks
    np.save(os.path.join(results_dir, "masks", f'{galpy_name}-{mw_type}-{snap_time}_stars_mask.npy'), stel_mask)
    np.save(os.path.join(results_dir, "masks", f'{galpy_name}-{mw_type}-{snap_time}_dm_mask.npy'), mask)
    
    print("Results saved successfully")


if __name__ == "__main__":
    main()