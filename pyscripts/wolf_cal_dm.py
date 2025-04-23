import numpy as np
import os
import glob
import h5py
import sys
import scipy
import scipy.integrate
import argparse
import yaml
import warnings

# Import local modules
try:
    from . import dm_tools as dm
except ImportError:
    # Try to import from current directory if not found as package
    import dm_tools as dm

import galpy.potential as gp
import astropy.units as u
from galpy.orbit import Orbit


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
    required_sections = ['file_paths', 'mw_potentials', 'analysis', 'run_parameters']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section '{section}' in configuration file")
    
    return cfg


def get_mw_potential(cfg):
    """
    Get the Milky Way potential based on configuration
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary
        
    Returns:
    --------
    galpy.potential : Galpy potential object
    """
    potential_type = cfg['run_parameters']['potential_type']
    potential_factor = cfg['run_parameters'].get('potential_factor', None)
    
    # Get potential configuration
    if potential_type == 'light':
        # Standard Bovy potential
        scaling_factor = 1.0
    elif potential_type == 'heavy':
        # Bovy potential with factor of 2
        scaling_factor = 2.0
    elif potential_type == 'custom':
        # Custom potential with user-specified factor
        if potential_factor is None:
            raise ValueError("For 'custom' potential type, 'potential_factor' must be specified in the config")
        scaling_factor = float(potential_factor)
    else:
        raise ValueError(f"Unknown potential type '{potential_type}'. Use 'light', 'heavy', or 'custom'")
    
    # Create the MW potential
    MW_pot = gp.MWPotential2014
    MW_pot[2] *= scaling_factor  # Scale the dark matter halo component
    
    return MW_pot


def main():
    """Main function for the Wolf estimator"""
    # Parse command line arguments - only accept config file path
    parser = argparse.ArgumentParser(description="Wolf estimator for dark matter simulations")
    parser.add_argument('config_file', type=str, help='Path to configuration file (YAML)')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config_file)
    
    # Get parameters from config
    galpy_name = cfg['run_parameters']['galpy_name']
    p_type = int(cfg['run_parameters'].get('particle_type', 3))
    use_tidal_filter = cfg['analysis'].get('use_tidal_filter', False)
    
    # Get file paths from config
    snapshot_dir = cfg['file_paths']['snapshot_dir']
    fof_dir = cfg['file_paths']['fof_dir']
    results_dir = cfg['file_paths']['results_dir']
    
    # Get MW potential
    MW_pot = get_mw_potential(cfg)
    
    # Setup filter mode for saving
    filter_mod = 'tidal' if use_tidal_filter else 'bound'
    
    # Print configuration info
    print(f"Running Wolf estimator for {galpy_name}")
    print(f"Using MW potential: {cfg['run_parameters']['potential_type']}")
    print(f"Tidal filter: {'enabled' if use_tidal_filter else 'disabled'}")
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Build file path patterns
    snap_pattern = os.path.join(snapshot_dir, "snapshot_*.hdf5")
    fof_pattern = os.path.join(fof_dir, "fof_*.hdf5")
    
    # Get file lists
    snap_files = sorted(glob.glob(snap_pattern))[:]
    fof_files = sorted(glob.glob(fof_pattern))[:]
    
    if not snap_files:
        print(f"Error: No snapshot files found at {snap_pattern}")
        return
    
    if not fof_files:
        print(f"Warning: No FOF files found at {fof_pattern}")
    
    # Setup orbit calculation
    o = Orbit.from_name(galpy_name, ro=8., vo=220.)
    ts = np.linspace(20.,-10.,100000)*u.Gyr
    o.integrate(ts, MW_pot)
    ts = np.linspace(0.,-10.,100000)
    
    # Find apocenter time
    potential_type = cfg['run_parameters']['potential_type']
    if (potential_type == 'light') and (galpy_name.lower() == 'fornax'):
        apo_t = ts[(ts<-7) & (ts>-10)][np.argmax(o.r(ts[(ts<-7) & (ts>-10)]*u.Gyr))]
    else:
        apo_t = ts[(ts<-8) & (ts>-10)][np.argmax(o.r(ts[(ts<-8) & (ts>-10)]*u.Gyr))]
    
    gp.turn_physical_on(MW_pot, ro=8, vo=220)
    h = cfg.get('hubble_parameter', 1.0)
    
    # Use custom final_fit_path if provided in config, otherwise construct it
    if 'final_fit_path' in cfg['file_paths']:
        final_fit_path = cfg['file_paths']['final_fit_path']
        print(f"Using specified output path: {final_fit_path}")
        # Create the output directory if it doesn't exist
        os.makedirs(final_fit_path, exist_ok=True)
    else:
        # Setup output directory using standard naming convention
        potential_type = cfg['run_parameters']['potential_type']
        final_fit_path = os.path.join(results_dir, f"{potential_type}_wolf_fit_{filter_mod}")
        # Create the output directory
        os.makedirs(final_fit_path, exist_ok=True)
    
    print('Saving results in ' + final_fit_path)

    # Hernquist and Zhao functions:
    def hern_I(radii, M, a):
        # ... existing code ...
        inds = np.argsort(radii)
        rev_inds = np.argsort(inds) 
        s = radii/a
        s = s[inds]
        X = 1/np.sqrt(1-s[s<=1]**2) * np.log((1+ np.sqrt(1-s[s<=1]**2))/s[s<=1])
        X = np.append(X,1/np.sqrt(s[s>1]**2 - 1) * np.arccos(1/s[s>1]))
        I = M*((2+s**2)*X - 3)/(2 * np.pi * a**2 * 1 * (1-s**2)**2)
        return np.log(I[inds[rev_inds]])

    def Zhao_density(radii,r_c,rho_c,a,b,g):
        # ... existing code ...
        r_c = 10**r_c
        rho_c = 10**rho_c
        a = 10**a
        b = 10**b
        g = 10**g
        x = radii/r_c
        rho = 2**((b-g)/a) * rho_c / (x**g * (1 + x**(a))**((b-g)/a))
        return np.log(rho)

    # fit hernquist projection profile onto zhao density profile to get gamma param
    with h5py.File(snap_files[0], 'r') as hf:
                mass = hf['Header'].attrs['MassTable'][int(p_type)]*1e10
                part_len = len(hf['PartType'+str(p_type)]['ParticleIDs'])
                # time = hf['Header'].attrs['Time']
    M = part_len*mass
    radii = np.logspace(-3,3, 500)
    r200 = (M*4.301*1e3/h**2)**(1/3)
    a = dm.find_a_hern(infall_t = None, r_vir=r200, mass=M, z = 0)
    I = hern_I(radii, M, a/1000)
    dens_params, cov = scipy.optimize.curve_fit(Zhao_density, radii, I, bounds=(np.log10([0.1,100,0.0001,0.1,0.1]),np.log10([50,1e9,7,40,2])), maxfev = 1800)
    print(f'initial theoretical fit params: {10**dens_params}')


    def Zhao_profile_mass_log_log_mod(radii,r_c,rho_c,a,b,g, r_t, delta): # coreZHAOtides model based off coreNFWtides
        # ... existing code ...
        cmd = []
        r_c = 10**r_c
        rho_c = 10**rho_c
        a = 10**a
        b = 10**b
        r_t = 10**r_t
        delta = 10**delta
        g = 10**g
        new_radii = np.log10(radii)
        new_radii = sorted(new_radii)
        for i, r in enumerate(new_radii):
            if i == 0:
                bins = np.linspace(min(new_radii)-1,r,100)
                newbins = 10**bins
                x = newbins/r_c
                rho = 2**((b-g)/a) * rho_c / (x**g * (1 + x**(a))**((b-g)/a))
                cmd.append(scipy.integrate.simpson(y=2*np.log(10)*np.pi*rho*newbins**2,x=bins))
            else:    
                bins = np.linspace(new_radii[i-1],r,100)
                newbins = 10**bins
                if 10**new_radii[i-1] > r_t:
                    x = r_t/r_c
                    rho = (2**((b-g)/a) * rho_c / (x**g * (1 + x**(a))**((b-g)/a))) * (newbins/r_t)**(-delta)
                else:
                    x = newbins[newbins<r_t]/r_c
                    rho = 2**((b-g)/a) * rho_c / (x**g * (1 + x**(a))**((b-g)/a))
                    if 10**r > r_t:
                        x = np.full(len(newbins[newbins>=r_t]),r_t)/r_c
                        rho = np.append(rho,(2**((b-g)/a) * rho_c / (x**g * (1 + x**(a))**((b-g)/a))) * (newbins[newbins>=r_t]/r_t)**(-delta))
                cmd.append(cmd[i-1]+ scipy.integrate.simpson(y=2*np.log(10)*np.pi*rho*newbins**2,x=bins))
        return np.array(cmd)


    def Zhao_density_mod(radii,r_c,rho_c,a,b,g, r_t, delta):
        # ... existing code ...
        r_c = 10**r_c
        rho_c = 10**rho_c
        a = 10**a
        b = 10**b
        g = 10**g
        r_t = 10**r_t
        delta = 10**delta
        rhos = []
        for radius in radii:
            if radius < r_t:
                x = radius/r_c
                rhos.append(2**((b-g)/a) * rho_c / (x**g * (1 + x**(a))**((b-g)/a)))
            else:
                x = r_t/r_c
                rhos.append((2**((b-g)/a) * rho_c / (x**g * (1 + x**(a))**((b-g)/a))) * (radius/r_t)**(-delta))
        return np.array(rhos).squeeze()

    def log_likelihood_mod(theta, x, y, y_err, total_mass):
        # ... existing code ...
        r_c,rho_c,a,b,g, r_t, delta = theta
        model_func = lambda x,r_c, rho_c, a, b,g, r_t, delta: Zhao_profile_mass_log_log_mod(radii=x, r_c=r_c,rho_c=rho_c,a=a,b=b,g=g,r_t=r_t,delta=delta)
        model = model_func(x, r_c,rho_c,a,b,g, r_t, delta)/total_mass
        sigma2 = y_err**2
        return -0.5 * np.sum(((y - model)** 2 / sigma2) + np.log(sigma2))

    # Initialize analysis arrays
    alphas = []
    betas = []
    rho_ss = []
    r_cs = []
    r_ts = []
    deltas = []
    half_r_true = []
    half_r_fit = [] 
    times = []
    r3s = []
    nr_parts = []
    true_3ds = []
    true_2ds = []
    wolf_3ds = []
    wolf_2ds = []
    fit_mass_2d = []
    mw_rads = []
    tidal_rs = []
    bound_rs = []
    fit_bounds = []
    gammas = []
    
    # main loop through snap files:
    back_count = 0
    for i in range(len(snap_files)):
        print(f'Working on Snapshot {i}', flush=True)
        try:    
            with h5py.File(snap_files[i], 'r') as hf:
                raw_coords = np.array(hf['PartType'+str(p_type)]['Coordinates'])
                raw_vels = np.array(hf['PartType'+str(p_type)]['Velocities'])
                time = hf['Header'].attrs['Time']
                mass = hf['Header'].attrs['MassTable'][int(p_type)]*1e10
                part_ids = np.array(hf['PartType'+str(p_type)]['ParticleIDs'])
        except Exception:
            print(f'No Snapshot!')
            continue    
        try:  
            with h5py.File(fof_files[i], 'r') as hf:
                part_len = hf['Subhalo']['SubhaloLenType'][0][int(p_type)]
                SF_offset = hf['Subhalo']['SubhaloOffsetType'][0][int(p_type)]
                SF_com = np.array(hf['Subhalo']['SubhaloCM'][0])
        except Exception:
            print(f'No Subhalo!')
            back_count -= 1
            continue
        
        # transformations:
        com_rad = np.sqrt(np.sum(SF_com*SF_com))
        radii, radii_3d, los_vels_2d, los_vels_3d = dm.transform_coords(raw_coords, raw_vels, SF_com)
        
        # ... rest of the existing processing code ...
        if i == 0:
            mask = np.isin(part_ids, part_ids[SF_offset:SF_offset+part_len])
            new_radii_3d = radii_3d[mask]
            new_vels = np.rollaxis(raw_vels[mask],1,0)
            new_radii_2d = radii[mask]
            los_vels_2d = los_vels_2d[mask]
            los_vels_3d = los_vels_3d[:part_len]
        else:
            new_mask = np.isin(part_ids, part_ids[SF_offset:SF_offset+part_len])
            mask = np.logical_and(np.isin(part_ids,old_ids), new_mask)
            new_radii_3d = radii_3d[mask]
            new_vels = np.rollaxis(raw_vels[mask],1,0)
            new_radii_2d = radii[mask]
            los_vels_2d = los_vels_2d[mask]
            los_vels_3d = los_vels_3d[mask]
        old_ids = part_ids[mask]
        radii = radii[~np.isnan(radii)]
        los_vels_2d = los_vels_2d[~np.isnan(new_radii_2d)]
        new_radii_2d = new_radii_2d[~np.isnan(new_radii_2d)]
        
        try:
            bound_r = max(new_radii_3d)
        except Exception as e:
            print(e)
            print('strange max() error')
            back_count -= 1
            continue
            
        # fits and calculations:
        tidal_r = dm.find_pot_tidal_r(new_radii_3d, MW_pot, mass, com_rad)
        if tidal_r <= 0.02:
            tidal_r = gp.rtide(MW_pot, R=np.sqrt(np.sum(SF_com[:2]*SF_com[:2]))*u.kpc, z=SF_com[2]*u.kpc, t=(apo_t+time)*u.Gyr, M=mass*len(new_radii_3d)*u.Msun)

        try:
            r3 = dm.fit_r3(new_radii_3d, use_counts=cfg['analysis'].get('use_counts', True))
        except Exception as e:
            print(e)
            r3 = dm.fit_r3(new_radii_3d, r3_type='diff')
        
        total_mass = mass*len(new_radii_3d)                 
        # lin_bins_2d = 10**np.linspace(np.log10(sorted(new_radii_2d)[0]), np.log10(sorted(new_radii_2d)[-1]), num=1000)
        
        # ... rest of calculations and fitting ...
        fit_bins = []
        sorted_2d_radii = np.sort(new_radii_2d)
        bin_count = int(len(new_radii_2d)/50)
        for k in range(int(len(new_radii_2d)/bin_count)):
            fit_bins.append(sorted_2d_radii[k*bin_count])
        if len(new_radii_2d) % bin_count != 0:
            fit_bins.append(sorted_2d_radii[-1])
        fit_bins = np.array(fit_bins)
        
        fit_bins = np.logspace(np.log10(sorted_2d_radii[0]), np.log10(sorted_2d_radii[-1]), num=201)
        pdf, bins_edges = np.histogram(new_radii_2d, bins=fit_bins, density=True)
        cdf = total_mass*np.cumsum(pdf*np.diff(bins_edges))
        bin_centers = bins_edges[1:]
        bin_centers = np.array(bin_centers)
        part_n = []
        for bc in bin_centers:
            part_n.append(len(new_radii_2d[new_radii_2d<bc]))
        pois_errs = np.sqrt(part_n)*mass
        nll = lambda *args: -log_likelihood_mod(*args)
        
        if i == 0:    
            init_params = 10**dens_params[:]
            initial = np.log10([*init_params, tidal_r - tidal_r/1.5, 3])
        else:
            initial = params
            
        bounds = ((initial[0]-2,np.log10(10**initial[0] +10)),
                (initial[1]-4,initial[1]+4),
                (initial[2]-1,np.log10(10**initial[2] +5)),
                (initial[3]-1,np.log10(10**initial[3] +5)),
                (initial[4]-1,np.log10(10**initial[4] +5)),
                (initial[5]-2,np.log10(10**initial[5] +20)),
                (initial[6]-2,np.log10(10**initial[6] +5)))
                
        short = 0
        
        if use_tidal_filter:
            filter_r = tidal_r
        else:
            filter_r = max(bin_centers)
   
        soln = scipy.optimize.minimize(
                        nll, 
                        x0=initial, 
                        args=(bin_centers[bin_centers<=filter_r-short],
                                            cdf[bin_centers<=filter_r-short], 
                                            pois_errs[bin_centers<=filter_r-short],
                                            1), 
                        bounds=bounds)
        params = soln.x
        
        test_r = np.linspace(min(bin_centers), 400, 10000)
        fit_cdf = Zhao_profile_mass_log_log_mod(test_r, *params)
        index = len(fit_cdf[fit_cdf/max(fit_cdf)<=0.5])
        
        try:
            r_2d_true = dm.find_half_r(new_radii_3d)
        except Exception as e:
            print(e)
            back_count -= 1
            continue
        
        # add params and other calculations to final lists:
        try:
            fit_mass_2d.append(fit_cdf[fit_cdf/max(fit_cdf)<=0.5][-1])
        except Exception as e:
            print(e)
            continue
        
        half_r_fit.append(test_r[index])
        half_r_true.append(r_2d_true)
        r_cs.append(10**params[0])
        rho_ss.append(10**params[1])
        alphas.append(10**params[2])
        betas.append(10**params[3])
        gammas.append(10**params[4])
        r_ts.append(10**params[5])
        deltas.append(10**params[6])
        tidal_rs.append(tidal_r)
        
        times.append(time)
        r3s.append(r3)
        mw_rads.append(com_rad)
        bound_rs.append(bound_r)
        nr_parts.append(len(new_radii_2d))
        fit_bounds.append(bounds)
        
        # Velocity dispersion calculations
        vel_disp_3d = dm.fit_tot_vel_disp_r3(new_radii_3d, new_vels, r3, 
                                            bin_num=cfg['analysis'].get('bin_size', 41),
                                            use_counts=cfg['analysis'].get('use_counts', True),
                                            use_log=cfg['analysis'].get('use_log', False))
        vel_disp_2d = np.std(los_vels_2d[new_radii_2d<=test_r[index]])
        
        int_radii = np.logspace(np.log10(np.min(new_radii_2d)), np.log10(np.max(new_radii_2d)), 
                                cfg['analysis'].get('bin_size', 41))
                                
        r_zhao_2d_dens = Zhao_density_mod(int_radii, *params)*int_radii
        
        los_vel_disp_bins = []
        sorted_radii = np.sort(new_radii_2d)
        los_bin_counts = int(len(new_radii_2d)/cfg['analysis'].get('bin_size', 41))
        for k in range(int(len(new_radii_2d)/los_bin_counts)):
            los_vel_disp_bins.append(sorted_radii[k*los_bin_counts])
        if len(new_radii_2d) % los_bin_counts != 0:
            los_vel_disp_bins.append(sorted_radii[-1])
        los_vel_disp_bins = np.array(los_vel_disp_bins)
        
        los_vel_disps = []
        for k in range(len(los_vel_disp_bins) - 1):
            bin_mask = (new_radii_2d < los_vel_disp_bins[k + 1]) & (new_radii_2d >= los_vel_disp_bins[k])
            bin_data = los_vels_2d[bin_mask]
            if len(bin_data) > 1:
                los_vel_disps.append(np.log10(np.std(bin_data) ** 2))
            else:
                los_vel_disps.append(np.nan)

        los_vel_disps = np.array(los_vel_disps)
        los_vel_disp_bin_centers = (los_vel_disp_bins[1:] + los_vel_disp_bins[:-1]) / 2
        
        # Remove NaN values for interpolation
        valid_mask = ~np.isnan(los_vel_disps)
        s = 0
        s_count = 0
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=RuntimeWarning)
                    los_tck_s = scipy.interpolate.splrep(los_vel_disp_bin_centers[valid_mask], 
                                                        los_vel_disps[valid_mask], k=1, s=0.002)
                    break
            except RuntimeWarning as e:
                if "spline with fp=s has been reached" in str(e):
                    s += 0.05
                else:
                    break
                s_count += 1
                if s_count > 10:
                    print("Warning: s value has been increased 10 times. Breaking loop.")
                    break
                    
        vel_disps = 10**scipy.interpolate.BSpline(*los_tck_s)(int_radii)
        vel_r_zhao_2d_dens = vel_disps * r_zhao_2d_dens
        vel_disp_2d = scipy.integrate.simpson(vel_r_zhao_2d_dens, int_radii)/scipy.integrate.simpson(r_zhao_2d_dens, int_radii)

        wolf_3ds.append((4.301E-3)**(-1) * (vel_disp_3d)**2 * r3s[-1]*1000)
        wolf_2ds.append(4 * (4.301E-3)**(-1) * vel_disp_2d * half_r_fit[-1]*1000)
        true_3ds.append(len(new_radii_3d[new_radii_3d<=r3s[-1]])*mass)
        true_2ds.append(len(new_radii_3d[new_radii_3d<=half_r_true[-1]])*mass)
        
        # Save results
        np.save(os.path.join(final_fit_path, 'last_ids.npy'), old_ids)
        np.save(os.path.join(final_fit_path, 'alphas.npy'), alphas)
        np.save(os.path.join(final_fit_path, 'betas.npy'), betas)
        np.save(os.path.join(final_fit_path, 'gammas.npy'), gammas)
        np.save(os.path.join(final_fit_path, 'r_ts.npy'), r_ts)
        np.save(os.path.join(final_fit_path, 'deltas.npy'), deltas)
        np.save(os.path.join(final_fit_path, 'rho_ss.npy'), rho_ss)
        np.save(os.path.join(final_fit_path, 'r_cs.npy'), r_cs)
        np.save(os.path.join(final_fit_path, 'tidal_rs.npy'), tidal_rs)
        np.save(os.path.join(final_fit_path, 'bound_rs.npy'), bound_rs)
        np.save(os.path.join(final_fit_path, 'half_r_true.npy'), half_r_true)
        np.save(os.path.join(final_fit_path, 'half_r_fit.npy'), half_r_fit)
        np.save(os.path.join(final_fit_path, 'times.npy'), times)
        np.save(os.path.join(final_fit_path, 'r3s.npy'), r3s)
        np.save(os.path.join(final_fit_path, 'mw_rs.npy'), mw_rads)
        np.save(os.path.join(final_fit_path, 'nr_parts.npy'), nr_parts)
        np.save(os.path.join(final_fit_path, 'true_3ds.npy'), true_3ds)
        np.save(os.path.join(final_fit_path, 'true_2ds.npy'), true_2ds)
        np.save(os.path.join(final_fit_path, 'wolf_3ds.npy'), wolf_3ds)
        np.save(os.path.join(final_fit_path, 'wolf_2ds.npy'), wolf_2ds)
        np.save(os.path.join(final_fit_path, 'fit_mass_2d.npy'), fit_mass_2d)
        np.save(os.path.join(final_fit_path, 'fit_bounds.npy'), fit_bounds)
        
        print('Done!')
        
        # Refresh file lists in case new files appeared
        snap_files = sorted(glob.glob(snap_pattern))[:]
        fof_files = sorted(glob.glob(fof_pattern))[:]


if __name__ == "__main__":
    main()
