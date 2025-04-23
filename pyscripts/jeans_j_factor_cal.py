from scipy.integrate import nquad
import numpy as np
import pandas as pd
from mpi4py import MPI
import argparse
import os
import yaml


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


def save_last_processed_line(line_number, filename):
    """Save the last processed line to keep track of progress"""
    with open(filename, 'a+') as f:
        f.write(str(line_number) + "\n")


def setup_integrand_functions(plaw, rbins=None):
    """Set up the integrand functions based on power law flag"""
    if not plaw:
        def integrand(x, theta, rhos, rs, D, theta_max, alpha, beta, gamma):
            return 2. * np.pi * np.sin(theta) * rhos**2. /(((D**2. * np.sin(theta)**2. + x**2.)/rs**2)**gamma * (1. + (np.sqrt(D**2. * np.sin(theta)**2. + x**2.)/rs)**(alpha))**(2*(beta-gamma)/alpha))

        def bounds_x(theta, rhos, rs, D, theta_max, alpha, beta, gamma):
            return [-np.inf, np.inf]

        def bounds_theta(rhos, rs, D, theta_max, alpha, beta, gamma):
            return [0, theta_max]
            
        return integrand, bounds_x, bounds_theta
    else:
        def integrand(x, theta, rhos, D, theta_max, gamma0, gamma1, gamma2, gamma3, gamma4, rbins):
            r = np.sqrt(D**2. * np.sin(theta)**2. + x**2.)
            if r < rbins[0]:
                rho = rhos*(r/rbins[0])**(-gamma0)
            if r > rbins[0] and r < rbins[1]:
                rho = rhos*(r/rbins[1])**(-gamma1) * (rbins[1]/rbins[0])**(-gamma1)
            if r > rbins[1] and r < rbins[2]:
                rho = rhos*(r/rbins[2])**(-gamma2) * (rbins[2]/rbins[1])**(-gamma2) * (rbins[1]/rbins[0])**(-gamma1)
            if r > rbins[2] and r < rbins[3]:
                rho = rhos*(r/rbins[3])**(-gamma3) * (rbins[3]/rbins[2])**(-gamma3) * (rbins[2]/rbins[1])**(-gamma2) * (rbins[1]/rbins[0])**(-gamma1)
            if r > rbins[3]:
                rho = rhos*(r/rbins[4])**(-gamma4) * (rbins[4]/rbins[3])**(-gamma4) * (rbins[3]/rbins[2])**(-gamma3) * (rbins[2]/rbins[1])**(-gamma2) * (rbins[1]/rbins[0])**(-gamma1)
        
            return 2. * np.pi * np.sin(theta) * rho**2.
        
        def bounds_x(theta, rhos, D, theta_max, gamma0, gamma1, gamma2, gamma3, gamma4, rbins):
            return [-np.inf, np.inf]

        def bounds_theta(rhos, D, theta_max, gamma0, gamma1, gamma2, gamma3, gamma4, rbins):
            return [0, theta_max]
            
        return integrand, bounds_x, bounds_theta


def main():
    """Main function for the J-factor calculator"""
    # Parse command line arguments - only accept config file path
    parser = argparse.ArgumentParser(description="J-factor calculator for dark matter simulations")
    parser.add_argument('config_file', type=str, help='Path to configuration file (YAML)')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config_file)
    
    # Get parameters from config
    dsph_name = cfg['run_parameters']['dsph_name']
    mw_type = cfg['run_parameters']['mw_type']
    snap_time = cfg['run_parameters']['snap_time']
    plaw = cfg['run_parameters'].get('power_law', False)
    baes = cfg['run_parameters'].get('baes', False)
    theta_max = cfg['run_parameters'].get('theta_max', 0.5*0.0174533)  # Default is 0.5 degrees in radians
    
    # Get file paths from config
    results_dir = cfg['file_paths']['results_dir']
    chains_dir = cfg['file_paths']['chains_dir']
    galaxy_data_dir = cfg['file_paths'].get('galaxy_data_dir', os.path.join(results_dir, 'GalaxyData'))
    
    # Construct paths
    output_dir = os.path.join(results_dir, f"{dsph_name}_{mw_type}_{snap_time}_gravsph")
    chains_file = os.path.join(chains_dir, f"{dsph_name}_{mw_type}_{snap_time}_gravsph_Chains{dsph_name}_{mw_type}_{snap_time}.txt")
    output_file = os.path.join(output_dir, "log10_jfactors.txt")
    processed_file = os.path.join(output_dir, "last_processed_line.txt")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get galaxy distance based on name
    gal_dist = cfg['galaxy_distances'].get(dsph_name, None)
    if gal_dist is None:
        raise ValueError(f"Galaxy distance for {dsph_name} not found in config file")
    
    print(f"Starting J-factor calculation for {dsph_name}...")
    print(f"Using MW type: {mw_type}, snapshot time: {snap_time}")
    print(f"Power law: {'yes' if plaw else 'no'}, Baes: {'yes' if baes else 'no'}")
    
    # Initialize file for saving
    with open(output_file, "a+") as file_for_check:
        file_length = len(file_for_check.read().split('\n'))
        if file_length == 1:
            print(f"Processing {dsph_name}_{mw_type}_{snap_time}")
    
    # Setup MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Load r_half for power law model if needed
    rbins = None
    if plaw:
        rhalf_file = os.path.join(galaxy_data_dir, f"{dsph_name}_{mw_type}_{snap_time}_gravsph_Rhalf.txt")
        try:
            r_half = float(open(rhalf_file, "r").readline())
            rbins = np.array([0.25, 0.5, 1, 2, 4]) * r_half
        except FileNotFoundError:
            raise FileNotFoundError(f"R_half file not found: {rhalf_file}")
    
    # Setup integrand functions
    integrand, bounds_x, bounds_theta = setup_integrand_functions(plaw, rbins)
    
    # Load chains data
    try:
        df = pd.read_csv(chains_file, sep=" ", header=None)
    except FileNotFoundError:
        raise FileNotFoundError(f"Chains file not found: {chains_file}")
    
    # Set column names based on model type
    if not plaw and not baes:
        df.columns = ['logrhos', 'logrs', 'alpha', 'beta', 'gamma', 'anis', 'logM', 'chi2']
    elif not plaw and baes:
        df.columns = ['logrhos', 'logrs', 'alpha', 'beta', 'gamma', 'anis', 'beta_inf', 'logra', 'eta', 'logM', 'chi2']
    elif plaw and not baes:
        df.columns = ['logrhos', 'gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4', 'anis', 'logM', 'chi2']
    elif plaw and baes:
        df.columns = ['logrhos', 'gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4', 'anis', 'beta_inf', 'logra', 'eta', 'M_p1', 'a_p1', 'M_p2', 'a_p2', 'M_p3', 'a_p3', 'logM', 'chi2']
    
    # Apply sigma clipping to chi2 values
    log_chi = df.chi2
    ratio = 1
    while ratio > 0.000001:
        old_chi = log_chi
        median = np.median(log_chi)
        std = np.std(log_chi)
        log_chi = log_chi[log_chi < median + 5*std]
        log_chi = log_chi[log_chi > median - 5*std]
        ratio = (np.std(old_chi) - np.std(log_chi))/np.std(log_chi)
    
    chi_filter = min(log_chi)
    df = df[df['chi2'] >= chi_filter]
    
    # Prepare indices for processing
    number_of_outputs = len(df)
    array_of_output_indices = np.arange(number_of_outputs)
    
    # Read already processed indices
    if os.path.exists(processed_file):
        processed_indices_df = pd.read_csv(processed_file, header=None)
        processed_indices = set(processed_indices_df[0])
        array_of_output_indices = np.array([i for i in array_of_output_indices if i not in processed_indices])
    
    # Split work among MPI ranks
    split_array = np.array_split(array_of_output_indices, size)
    
    # Process assigned indices
    for i, result in enumerate(split_array[rank]):
        if not plaw:
            rhos = 10**df['logrhos'].iloc[result]
            rs = 10**df['logrs'].iloc[result]
            alpha = df['alpha'].iloc[result]
            beta = df['beta'].iloc[result]
            gamma = df['gamma'].iloc[result]
            jfactor = nquad(integrand, [bounds_x, bounds_theta], args=(rhos, rs, gal_dist, theta_max, alpha, beta, gamma))[0]
        else:
            rhos = 10**df['logrhos'].iloc[result]
            gamma0 = df['gamma0'].iloc[result]
            gamma1 = df['gamma1'].iloc[result]
            gamma2 = df['gamma2'].iloc[result]
            gamma3 = df['gamma3'].iloc[result]
            gamma4 = df['gamma4'].iloc[result]
            jfactor = nquad(integrand, [bounds_x, bounds_theta], args=(rhos, gal_dist, theta_max, gamma0, gamma1, gamma2, gamma3, gamma4, rbins))[0]
        
        # Calculate log J factor with unit conversion
        log_j_factor = np.log10(jfactor*1.12**2 * 1e9 / 3.087**5)
        
        # Save result
        with open(output_file, "a+") as file_for_saving:
            file_for_saving.write(str(log_j_factor) + "\n")
            
        save_last_processed_line(result, processed_file)
    
    print('Done!')


if __name__ == "__main__":
    main()

