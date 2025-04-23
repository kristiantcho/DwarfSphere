import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import galpy.potential as gp
from galpy.util import conversion
import pandas as pd
import astropy.units as u
from astropy.cosmology import LambdaCDM, z_at_value

# Import configuration settings (if available)
try:
    from . import config
except ImportError:
    # Fallback to default values if config is not found
    config = None
    
def get_config_value(section, key, default_value):
    """
    Helper function to get configuration values with fallback to defaults.
    
    This function retrieves configuration values from a config module,
    falling back to the provided default value if the config is not available
    or the specified key is missing.
    
    Parameters:
    -----------
    section : str
        The configuration section name to access
    key : str
        The configuration key within the section
    default_value : any
        The default value to return if the config or key is not found
        
    Returns:
    --------
    any : The configuration value if found, otherwise the default value
    """
    if config is None:
        return default_value
    return getattr(config, section, {}).get(key, default_value)


def find_hern_tidal_r(radii, M, m_dm, R0, a):
    """
    Calculate the tidal radius in a Hernquist potential.
    
    This function estimates the tidal radius of a dwarf galaxy orbiting within
    a host galaxy described by a Hernquist potential. The tidal radius is the
    distance from the center of the dwarf at which the gravitational force from
    the dwarf equals the differential pull from the host galaxy.
    
    Parameters:
    -----------
    radii : array_like
        Radii of particles from the center of the dwarf galaxy (kpc)
    M : float
        Mass of the host galaxy (Milky Way) in solar masses
    m_dm : float
        Mass of a single DM particle in solar masses
    R0 : float
        Orbital radius of the dwarf galaxy (kpc)
    a : float
        Scale length of the host galaxy's Hernquist profile (kpc)
        
    Returns:
    --------
    float : Tidal radius of the dwarf galaxy (kpc)
    """
    k = R0/(M*(3 - (2*a/(R0+a)))/((R0+a)**2))
    r_tests = np.linspace(1, 200, 1000)
    m_tests = []
    for r in r_tests:
        m_tests.append(m_dm*len(radii[radii<=r]))
    m_tests = np.array(m_tests)
    diffs = r_tests**3 - k*m_tests
    return r_tests[np.argmin(np.abs(diffs))]


def find_pot_tidal_r(radii, pot, m_dm, R0, t=0, potential_factor=None):
    """
    Calculate the tidal radius in a general potential.
    
    This function computes the tidal radius of a satellite galaxy orbiting in a host
    galaxy potential. The tidal radius is calculated by finding where the differential
    gravitational force from the host exceeds the self-gravity of the satellite.
    
    Parameters:
    -----------
    radii : array_like
        Radii of particles from the center of the satellite (kpc)
    pot : galpy.potential.Potential
        Galpy potential object representing the host galaxy
    m_dm : float
        Mass of a single dark matter particle in solar masses
    R0 : float
        Orbital radius of the satellite (kpc)
    t : float, optional
        Time in Gyr for time-dependent potentials (default: 0)
    potential_factor : float or str, optional
        Scaling factor for the potential
        
    Returns:
    --------
    float : Tidal radius of the satellite (kpc)
    """
    # Apply potential scaling if specified
    if potential_factor is not None:
        if isinstance(potential_factor, str) and config is not None:
            factor = config.MW_POTENTIAL_FACTORS.get(potential_factor, 1.0)
        else:
            factor = float(potential_factor)
        # Scale the potential forces (not implemented here as it depends on pot type)
        # This would require modifying the pot object according to the specified factor
    
    # Calculate the logarithmic derivative of the enclosed mass
    diff = (np.log(gp.mass(pot, (R0+0.001)*u.kpc, t=t*u.Gyr)) - 
            np.log(gp.mass(pot, (R0-0.001)*u.kpc, t=t*u.Gyr)))/(np.log(R0+0.001)-np.log(R0-0.001))
    
    # Test different radii to find the tidal radius
    r_tests = np.logspace(-2, np.log10(max(radii)), 500)
    m_tests = []
    for r in r_tests:
        m_tests.append(m_dm*len(radii[radii<=r]))
    diffs = r_tests - R0*(m_tests/(gp.mass(pot, R0*u.kpc, t=t*u.Gyr)*(3-diff)))**(1/3)
    return r_tests[np.argmin(np.abs(diffs))]


def sigma_clip(vels, radii, factor=3, limit=0.0001):
    """
    Perform sigma clipping on velocity data to remove outliers.
    
    This function iteratively removes velocity outliers that are more than
    a specified number of standard deviations away from the mean, continuing
    until the fractional change in standard deviation falls below a threshold.
    
    Parameters:
    -----------
    vels : array_like
        Velocities of particles to be clipped
    radii : array_like
        Radii of particles from the center
    factor : float, optional
        Number of standard deviations to use as clipping threshold (default: 3)
    limit : float, optional
        Convergence limit for the fractional change in standard deviation (default: 0.0001)
        
    Returns:
    --------
    float : Maximum radius of particles after clipping
    """
    ratio = 1
    while ratio > limit:
        old_vels = vels
        mean = np.mean(vels)
        std = np.std(vels)
        radii = radii[vels < mean + factor*std]
        vels = vels[vels < mean + factor*std]
        radii = radii[vels > mean - factor*std]
        vels = vels[vels > mean - factor*std]
        ratio = (np.std(old_vels) - np.std(vels))/np.std(vels)
    return max(radii)
    
    
def find_virial_r(radii, m_dm, h=None):
    """
    Calculate the virial radius of a dark matter halo.
    
    The virial radius is defined as the radius at which the average density of
    the halo equals 200 times the critical density of the universe. This function
    iteratively finds this radius by comparing the enclosed mass profile with the
    expected virial mass at each radius.
    
    Parameters:
    -----------
    radii : array_like
        Radii of particles from the center (kpc)
    m_dm : float
        Mass of a single DM particle in solar masses
    h : float, optional
        Hubble parameter (H0/100 km/s/Mpc). If None, uses the value from configuration
        
    Returns:
    --------
    float : Virial radius of the halo (kpc)
    """
    # Use configuration value for h if not specified
    if h is None:
        h = get_config_value('HUBBLE_PARAMETER', None, 1.0)
        
    # Calculate the probability density function
    pdf, bins_edges = np.histogram(
                            radii,        # array of data
                            bins=10000,   # specify the number of bins for distribution function
                            density=True  # True to return probability density function (pdf) instead of count
                        )
    # Calculate the cumulative mass function
    cdf = len(radii)*m_dm*np.cumsum(pdf*np.diff(bins_edges))
    
    # Calculate bin centers
    bin_centers = []
    for k in range(len(pdf)):
        bin_centers.append((bins_edges[k]+bins_edges[k+1])/2)
    bin_centers = np.array(bin_centers)
    
    # Calculate the mass corresponding to the virial density (200 times critical density)
    test_M = (bin_centers*1000)**3 * h**2 / (200*4.301e3) 
    
    # Find the radius where the actual mass equals the virial mass
    index = np.argmin(np.abs(cdf[bin_centers>10] - test_M[bin_centers>10]))
    return bin_centers[bin_centers>10][index]


def find_half_r(radii):
    """
    Calculate the half-mass radius of a distribution of particles.
    
    The half-mass radius is the radius within which half of the total mass is contained.
    
    Parameters:
    -----------
    radii : array
        Radii of particles from the center (kpc)
        
    Returns:
    --------
    float : Half-mass radius (kpc)
    """
    # Calculate the probability density function
    pdf, bins_edges = np.histogram(
                radii,         # array of data
                bins=10000,    # specify the number of bins for distribution function
                density=True   # True to return probability density function (pdf) instead of count
            )
    # Calculate the cumulative distribution function
    cdf = np.cumsum(pdf*np.diff(bins_edges))
    
    # Use bin edges as bin centers
    bin_centers = bins_edges[1:]
    
    # Find the index where the CDF reaches 0.5
    index = len(cdf[cdf/max(cdf)<0.5])
    
    return bin_centers[index]


def fit_r3(new_radii_3d, r3_type='spline', lim=None, bin_num=41, use_counts=True, use_log=False):
    """
    Calculate the r3 radius for a dark matter halo.
    
    The r3 radius is the radius at which the logarithmic slope of the density profile
    equals -3. This function provides three different methods to calculate r3:
    
    1. 'spline': Uses a spline interpolation of the density profile (recommended)
    2. 'diff': Simple point-to-point differentiation to find where slope = -3
    3. 'fit': Fits a Hernquist profile to the density data
    
    Parameters:
    -----------
    new_radii_3d : array_like
        Array of 3D radii of particles from the center
    r3_type : str, optional
        Method to use for calculation: 'spline', 'diff', or 'fit' (default: 'spline')
    lim : float, optional
        Maximum radius to consider for binning (default: max radius in data)
    bin_num : int, optional
        Number of bins to use (default: 41)
    use_counts : bool, optional
        If True, bin by particle count rather than radius (default: True)
    use_log : bool, optional
        If True, use logarithmic binning (default: False)
        
    Returns:
    --------
    float : The r3 radius in the same units as the input radii
    """
    
    # bins = np.linspace(0, int(np.ceil(max(radii))), 200)
    new_radii_3d = np.sort(new_radii_3d)
    if lim is None:
        sf_r = max(new_radii_3d)
    else:
        sf_r = lim
    if not use_counts:
        bins = np.logspace(0, np.log10(sf_r), 
                        #    num = int(np.ceil(bound_r)*bin_size)
                        num = bin_num
                        )
    else:
        bin_count = int(len(new_radii_3d)/bin_num)
        bins = []
        for i in range(int(len(new_radii_3d)/bin_count)):
            bins.append(new_radii_3d[i*bin_count])
        if int(len(new_radii_3d)/bin_count) != len(new_radii_3d)/bin_count:
            bins.append(new_radii_3d[-1])
        bins = np.array(bins)
    densities, _= np.histogram(new_radii_3d, bins)
    final_densities = []
    for i, dens in enumerate(densities):
        final_densities.append((dens+np.sqrt(dens))/((bins[i+1]**3 - bins[i]**3)*4*np.pi/3))
    final_densities = np.array(final_densities)
    r_bin_centers = abs(bins[1:] + bins[:-1])/2
    if len(final_densities[densities >= 1]) > 20:
        diff = np.diff(np.log10(final_densities[densities >= 1]))/np.diff(np.log10(r_bin_centers[densities >= 1]))
        new_r_cents = r_bin_centers[densities >= 1]
    else:
        diff = np.diff(np.log10(final_densities[densities >= 0]))/np.diff(np.log10(r_bin_centers[densities >= 0]))
        new_r_cents = r_bin_centers[densities >= 0]
    # true_r_cents = 10**(np.diff(np.log10(r_bin_centers[densities >= 1]))/2 + np.log10(r_bin_centers[densities >= 1])[:-1])
    # true_r_cents = true_r_cents[diff>=-4]
    
    new_r_cents = abs(new_r_cents[1:] + new_r_cents[:-1])/2
    if len(diff[diff>=-4]) > 5:
        new_r_cents = new_r_cents[diff>=-4]
        diff = diff[diff>=-4]
    
    hern_diff = lambda r,a : -1 - 3*r/(r+a)
    # m = len(diff[true_r_cents<=bound_r])
    new_r_cents = new_r_cents[~np.isnan(diff) & ~np.isinf(diff)]
    diff = diff[~np.isnan(diff) & ~np.isinf(diff)]
    

    test_r = np.linspace(0,max(new_radii_3d),500)
    spline_r = np.linspace(min(new_r_cents), max(new_r_cents), 300)

    if r3_type == 'diff':
        return new_r_cents[np.argmin(np.abs(diff+3))]
    
    if r3_type == 'fit':
        params = scipy.optimize.curve_fit(hern_diff, new_r_cents[new_r_cents<=sf_r], diff[new_r_cents<=sf_r])
        return test_r[np.argmin(np.abs(hern_diff(test_r, params[0]) + 3))]
    
    if r3_type == 'spline':
        if not use_log:
            tck_s = scipy.interpolate.splrep(new_r_cents, diff, s=len(new_r_cents) + 20)
            return spline_r[np.argmin(np.abs(scipy.interpolate.BSpline(*tck_s)(spline_r) + 3))]
        else:
            tck_s = scipy.interpolate.splrep(new_r_cents, np.log10(diff - min(diff) + 1), s=0.01, k=1)
            return spline_r[np.argmin(np.abs(10**(scipy.interpolate.BSpline(*tck_s)(spline_r)) + min(diff) + 2))]

def fit_tot_vel_disp_r3(new_radii_3d, new_vels, r3, lim=None, bin_num=31, use_counts=True, use_log=False):
    """
    Calculate the total 3D velocity dispersion at r3 using interpolating splines.
    
    This function computes the velocity dispersion profile as a function of radius
    and extrapolates it to the specific radius r3, which is the radius where the
    logarithmic density slope equals -3. This is an important parameter for the
    Wolf mass estimator.
    
    Parameters:
    -----------
    new_radii_3d : array_like
        Array of 3D radii of particles from the center
    new_vels : array_like
        Array of velocity components [vx, vy, vz] for each particle
    r3 : float
        The radius at which to calculate the velocity dispersion
    lim : float, optional
        Maximum radius to consider for binning (default: max radius in data)
    bin_num : int, optional
        Number of bins to use (default: 31)
    use_counts : bool, optional
        If True, bin by particle count rather than radius (default: True)
    use_log : bool, optional
        If True, use logarithmic interpolation (default: False)
        
    Returns:
    --------
    float : The velocity dispersion at radius r3 in the same units as input velocities
    """
    
    # bins = np.linspace(0, int(np.ceil(max(radii))), 200)
    if lim is None:
        sf_r = max(new_radii_3d)
    else:
        sf_r = lim
    if not use_counts:
        bins = np.logspace(np.log10(sorted(new_radii_3d)[0]), np.log10(sf_r), 
                        #    num = int(np.ceil(bound_r)*bin_size)
                        num = bin_num
                        )
    else:    
        bins = []
        bin_count= int(len(new_radii_3d)/bin_num)
        for i in range(int(len(new_radii_3d)/bin_count)):
            bins.append(sorted(new_radii_3d)[i*bin_count])
        if int(len(new_radii_3d)/bin_num) != len(new_radii_3d)/bin_num:
            bins.append(sorted(new_radii_3d)[-1])
        bins = np.array(bins)
    # densities, _= np.histogram(new_radii_3d, bins)
    final_disps = []
    for i in range(len(bins)-1):
        final_disps.append(np.sqrt(np.std(new_vels[0][(new_radii_3d < bins[i+1]) & (new_radii_3d >= bins[i])])**2 
                                + np.std(new_vels[1][(new_radii_3d < bins[i+1]) & (new_radii_3d >= bins[i])])**2 
                                + np.std(new_vels[2][(new_radii_3d < bins[i+1]) & (new_radii_3d >= bins[i])])**2))
    final_disps = np.array(final_disps)
    r_bin_centers = abs(bins[1:] + bins[:-1])/2
    if not use_log:
        tck_s = scipy.interpolate.splrep(r_bin_centers, final_disps, s=len(r_bin_centers))
        return scipy.interpolate.BSpline(*tck_s)(r3) 
    else:
        tck_s = scipy.interpolate.splrep(r_bin_centers, np.log10(final_disps), k=1, s = 0.002)
        return 10**scipy.interpolate.BSpline(*tck_s)(r3) 
    

def c_mass_relation(M,z,h=1):
    """
    Calculate the concentration parameter for a dark matter halo.
    
    This function implements a concentration-mass relation for dark matter halos
    as a function of mass and redshift, following standard cosmological models.
    The concentration parameter c is a key property of NFW profiles.
    
    Parameters:
    -----------
    M : float
        Virial mass of the halo in solar masses
    z : float
        Redshift
    h : float, optional
        Hubble parameter (H0/100 km/s/Mpc) (default: 1.0)
        
    Returns:
    --------
    float : Concentration parameter c for the specified halo
    """
    c0 = 3.395 * (1 + z)**(-0.215)
    B =  0.307 * (1 + z)**0.540
    g1 = 0.628 * (1 + z)**(-0.047)
    g2 = 0.317 * (1 + z)**(-0.893)
    a = 1/(1+z)
    dsc = 1.686
    Omm0 = 0.27
    Oml0 =  0.73
    Oml = Oml0/(Oml0 + Omm0*(1+z)**3)
    Omm = 1 - Oml
    Psi = Omm**(4/7) - Oml +(1 + Omm/2)*(1 + Oml/70)
    Psi0 = Omm0**(4/7) - Oml0 +(1 + Omm0/2)*(1 + Oml0/70)
    D = Omm*Psi0*a/(Omm0*Psi)
    v0 = (4.135 - 0.564*(a**(-1)) - 0.210*(a**(-2)) + 
          0.0557*(a**(-3)) - 0.00348*(a**(-4))) / D
    E = (M*h/(1E10))**(-1)
    sig = D*22.26 * (E**0.292) /(1 + 
            (1.53 * (E**0.275)) + (3.36 * (E**0.198)))
    v = dsc/sig
    c = c0 * ((v/v0)**(-g1)) * (1 + ((v/v0)**(1/B)))**(-B*(g2-g1)) 
    return c



def hern_mass(r,rho,a):
    """
    Calculate the enclosed mass for a Hernquist density profile.
    
    This function computes the mass enclosed within radius r for a Hernquist
    profile with characteristic density rho and scale length a.
    
    Parameters:
    -----------
    r : float or array_like
        Radius at which to calculate the enclosed mass (kpc)
    rho : float
        Characteristic density of the Hernquist profile (Msun/kpc^3)
    a : float
        Scale length of the Hernquist profile (kpc)
        
    Returns:
    --------
    float or array_like : Enclosed mass within radius r (solar masses)
    """
    return 4*np.pi*rho*a**3 * r**2/(2*(a + r)**2)


def find_a_hern(infall_t, r_vir = 287, mass = 1.3e12, h = 1, z = None):
    """
    Calculate the Hernquist scale length for a dark matter halo.
    
    This function computes the Hernquist scale length (a) for a dark matter halo
    based on its virial radius, mass, and redshift. The scale length is calculated
    by first determining the NFW concentration parameter from the mass-concentration
    relation, then converting to the Hernquist profile.
    
    Parameters:
    -----------
    infall_t : float or None
        Age of the universe at infall time (Gyr), used to determine redshift.
        If None, the provided z value is used directly.
    r_vir : float, optional
        Virial radius of the halo in pc (default: 287)
    mass : float, optional
        Virial mass of the halo in solar masses (default: 1.3e12)
    h : float, optional
        Hubble parameter (H0/100 km/s/Mpc) (default: 1.0)
    z : float, optional
        Redshift. If None, calculated from infall_t using the specified cosmology.
        
    Returns:
    --------
    float : Hernquist scale length in pc
    """
    cosmo = LambdaCDM(h*100 * (u.km/u.s/u.Mpc), 0.315, 0.685)
    # cosmo = LambdaCDM(100 * (u.km/u.s/u.Mpc), 0.3, 0.7)
    if z is None:    
        z = z_at_value(cosmo.age, infall_t * u.Gyr)
    # print(f'redshift at {infall_t} Gyr = {z}')
    c = c_mass_relation(mass,z, h)
    # print(f'MW concentration = {c} (at {infall_t} Gyr)')
    rs = r_vir/c
    a = rs*np.sqrt(2*(np.log(1+c) - c/(1+c)))
    return a


def print_halo_info(M, M_mw = 1.3e12, infall_t=0.5, z = None, h = 1, r_mw = None):
    """
    Print detailed information about a dark matter halo and its host.
    
    This function calculates and prints properties of both a dwarf spheroidal
    galaxy halo and its host (typically the Milky Way), including virial radii,
    scale lengths, concentrations, and velocities at a given redshift.
    
    Parameters:
    -----------
    M : float
        Virial mass of the dwarf galaxy halo in solar masses
    M_mw : float, optional
        Virial mass of the host galaxy (Milky Way) in solar masses (default: 1.3e12)
    infall_t : float, optional
        Age of the universe at infall time in Gyr (default: 0.5)
    z : float, optional
        Redshift (if None, calculated from infall_t) (default: None)
    h : float, optional
        Hubble parameter (H0/100 km/s/Mpc) (default: 1.0)
    r_mw : float, optional
        Orbital radius of the dwarf in the Milky Way in kpc (default: None)
        If provided, calculates the circular velocity at this radius
        
    Returns:
    --------
    None : Results are printed to the console
    """
    cosmo = LambdaCDM(h*100 * (u.km/u.s/u.Mpc), 0.315, 0.685)
    # cosmo = LambdaCDM(100 * (u.km/u.s/u.Mpc), 0.3, 0.7)
    if z is None:    
        z = z_at_value(cosmo.age, infall_t * u.Gyr)
    else:
        infall_t = cosmo.age(z).value
    h_z = h*cosmo.H(z).value/cosmo.H(0).value
    print(f'redshift at {infall_t} Gyr = {z}')
    c = c_mass_relation(M_mw,z, h)
    print(f'MW concentration = {c} (at {infall_t} Gyr)')
    MW_r200 = (M_mw*4.301*1e3/h_z**2)**(1/3)/1000
    rs = MW_r200/c
    print(f'MW NFW scale radius = {rs} kpc')
    a_mw = rs*np.sqrt(2*(np.log(1+c) - c/(1+c)))
    print(f'MW Hernquist scale length = {a_mw} kpc (at {infall_t} Gyr)')
    print(f'MW virial mass = {M_mw/1e12} 10^12 solar masses')
    print(f'MW virial radius = {MW_r200} kpc')
    dsph_m200 = M
    dsph_r200 = (dsph_m200*4.301*1e3/h_z**2)**(1/3)
    print(f'DSph with {M/1e9} 10^9 solar masses, virial radius = {dsph_r200/1000} kpc')
    c_dsph = c_mass_relation(dsph_m200,z)
    print(f'DSph NFW scale radius = {(dsph_r200/c_dsph)/1000} kpc')
    print(f'DSph concentration = {c_dsph}')
    dsph_v200 = np.sqrt(4.301*1e-3*dsph_m200/dsph_r200)
    print(f'DSph virial velocity = {dsph_v200} km/s')
    a_dpsh = find_a_hern(infall_t, r_vir=dsph_r200, mass = M, z = z, h=h)
    print(f'DSph Hernquist scale length = {a_dpsh/1000} kpc')
    if r_mw is not None:
        M_r = 1.3E12*r_mw**2 / (r_mw + a_mw)**2
        vr_dsph = np.sqrt(4.301e-6*M_r/r_mw)
        print(f'DSph circular velocity at {r_mw} kpc = {vr_dsph} km/s')
    return 


def shrinking_spheres(r200, pos, SF_com = None, dens = 250):
    """
    Find the center of mass of a particle distribution using the shrinking spheres method.
    
    This algorithm iteratively finds the center of mass by repeatedly calculating the median
    position of particles within a shrinking sphere. The sphere radius is reduced until the
    number of particles falls below a threshold density.
    
    Parameters:
    -----------
    r200 : float
        Initial radius of the sphere (typically virial radius) in kpc
    pos : array_like
        Array of particle positions with shape (3, n) where n is the number of particles
    SF_com : array_like, optional
        Initial guess for the center of mass. If None, uses the median of positions.
    dens : int, optional
        Target minimum number of particles in the final sphere (default: 250)
        
    Returns:
    --------
    array : Final center of mass position with shape (3,)
    """
    if SF_com is None:
        com_guess = np.median(pos, axis = 1)
    else:
        com_guess = SF_com
    rad = r200
    radii = []
    for i in range(len(pos)):
        if i == 0:
            radii = (pos[i] - com_guess[i])**2 
        else:
            radii += (pos[i] - com_guess[i])**2 
    wh, = np.where(np.sqrt(radii) < rad)
    nparts = len(wh)
    while nparts > dens:
        rad = rad * 0.9
        for i in range(len(pos)):
            if i == 0:
                radii = (pos[i] - com_guess[i])**2 
            else:
                radii += (pos[i] - com_guess[i])**2
        wh, = np.where(np.sqrt(radii) < rad)
        nparts = len(wh)
        for i in range(len(pos)):
            com_guess[i] = np.median(pos[i][wh])
    return com_guess


def find_bound_r(real_radii, vels, pots):
    """
    Find the radius at which particles become unbound from the system.
    
    This function calculates the radius at which the average kinetic energy of
    particles exceeds their potential energy (in absolute value), meaning they are
    no longer gravitationally bound to the system.
    
    Parameters:
    -----------
    real_radii : array_like
        Radii of particles from the center (kpc)
    vels : array_like
        Array of velocity components [vx, vy, vz] for each particle (km/s)
    pots : array_like
        Gravitational potential energies of particles
        
    Returns:
    --------
    float or None : Radius beyond which particles become unbound (kpc), or None if
                   all particles are bound
    """
    norm_vels = np.sqrt(vels[0]**2 + vels[1]**2 + vels[2]**2)
    norm_vels = norm_vels - np.average(norm_vels[real_radii<3])
    kin_es = 0.5*norm_vels**2
    avg_kes = []
    avg_radii = []
    avg_pots = []
    rad_step = 1
    for i in range(int(np.ceil(max(real_radii)/rad_step))):
        mask = (real_radii >= i*rad_step) & (real_radii < (i+1)*rad_step)
        if mask.any():
            avg_kes.append(np.mean(kin_es[mask]))
            avg_radii.append((i+1)*rad_step/2 + (i)*rad_step/2)
            avg_pots.append(np.mean(pots[mask]))
    avg_pots = np.array(avg_pots)
    avg_kes = np.array(avg_kes)
    avg_radii = np.array(avg_radii)
    if np.any(avg_kes>-avg_pots):
        bound_r = min(avg_radii[avg_kes>-avg_pots])
        return bound_r
    else:
        return

def DK_mass_profile(radii,r_c,rho_c,a,b,g):
    """
    Calculate the mass profile for a Dekel Zhao density model.
    
    Parameters:
    -----------
    radii : array_like
        Radii at which to calculate the enclosed mass (kpc)
    r_c : float
        Core radius of the profile (kpc)
    rho_c : float
        Central density of the profile (Msun/kpc^3)
    a : float
        Inner slope parameter
    b : float
        Transition sharpness parameter
    g : float
        Outer slope parameter
        
    Returns:
    --------
    array_like : Enclosed mass at the specified radii (solar masses)
    """
    x = radii/r_c
    rho = rho_c/(x**a * (1 + x**(1/b))**(b*(g-a)))
    return 4*np.pi * radii**3 * rho/3


def find_fit_half_mass(radii, total_mass, tidal_r):
    """
    Calculate the half-mass radius using a fitted density profile.
    
    This function fits a Dehnen-Kazantzidis density profile to the mass distribution
    data and uses it to determine the radius containing half of the total mass.
    This approach is more robust than direct binning for noisy data.
    
    Parameters:
    -----------
    radii : array_like
        Radii of particles from the center (kpc)
    total_mass : float
        Total mass of the system in solar masses
    tidal_r : float
        Tidal radius of the system in kpc, used as a cutoff for the fit
        
    Returns:
    --------
    float : Half-mass radius (kpc) determined from the fitted profile
    """
    pdf, bins_edges = np.histogram(
                    radii,        # array of data
                    bins=100000,    # specify the number of bins for distribution function
                    density=True # True to return probability density function (pdf) instead of count
                )
    cdf = total_mass*np.cumsum(pdf*np.diff(bins_edges))
    bin_centers = []
    for i in range(len(pdf)):
        bin_centers.append((bins_edges[i]+bins_edges[i+1])/2)
    bin_centers = np.array(bin_centers)
    params, covs = scipy.optimize.curve_fit(DK_mass_profile, bin_centers[bin_centers<tidal_r], cdf[bin_centers<tidal_r])
    test_r = np.linspace(0,300,10000)
    fit_cdf = DK_mass_profile(test_r, *params)/total_mass
    index = len(fit_cdf[fit_cdf<=0.5])
    return test_r[index]
    


def get_ecc_vel(M,a, r_peri, r_apo):
    """
    Calculate the circular velocity at apocenter for an eccentric orbit in a Hernquist potential.
    
    This function solves for the energy and angular momentum of an orbit with given
    pericenter and apocenter distances in a Hernquist potential, then returns the
    circular velocity at the apocenter.
    
    Parameters:
    -----------
    M : float
        Mass of the central body in 10^10 solar masses
    a : float
        Scale length of the Hernquist profile in kpc
    r_peri : float
        Pericenter distance of the orbit in kpc
    r_apo : float
        Apocenter distance of the orbit in kpc
        
    Returns:
    --------
    float : Circular velocity at apocenter in km/s
    """
    # M in 10^10 solar mass
    # a, rs in kpc
    G = 43018. #in code units

    def phi_hq(r, a, M):
        """
        Calculate the gravitational potential of a Hernquist profile.
        
        This helper function computes the gravitational potential at radius r
        for a Hernquist density profile with scale length a and mass M.
        
        Parameters:
        -----------
        r : float or array_like
            Radius at which to calculate the potential (kpc)
        a : float
            Scale length of the Hernquist profile (kpc)
        M : float
            Total mass of the Hernquist profile (10^10 solar masses)
            
        Returns:
        --------
        float or array_like : Gravitational potential at radius r
        """
        return -G * M / (r + a)

    def equations(p, rp, ra):
        """
        Solve the energy and angular momentum constraint equations for orbital parameters.
        
        This helper function defines the system of equations that must be satisfied
        for an orbit with given pericenter and apocenter in a Hernquist potential.
        
        Parameters:
        -----------
        p : tuple
            (E, L) tuple containing energy and angular momentum to solve for
        rp : float
            Pericenter distance of the orbit in kpc
        ra : float
            Apocenter distance of the orbit in kpc
            
        Returns:
        --------
        tuple : Residuals of the constraint equations that should equal zero 
                when the correct E and L are found
        """
        E, L = p
        return (2 * (E - phi_hq(rp,a,M)) - L**2./rp**2., 2 * (E - phi_hq(ra,a,M)) - L**2./ra**2.)

    E_fin, L_fin =  scipy.optimize.fsolve(equations, (5000., 5000.), args = (r_peri, r_apo,))
    
    return L_fin/r_apo #km/s


def make_grav_table(pot, boxsize, res, arepo_units=True):
    """
    Generate a table of gravitational forces for a given potential.
    
    This function computes the radial and vertical components of the gravitational
    force at a grid of positions, which can be used for external potential fields
    in simulation codes like AREPO.
    
    Parameters:
    -----------
    pot : galpy.potential.Potential
        Galpy potential object for which to compute forces
    boxsize : float
        Size of the box in kpc (assumes square grid with center at 0,0)
    res : float
        Resolution of the grid in kpc
    arepo_units : bool, optional
        If True, converts forces to AREPO code units (default: True)
        
    Returns:
    --------
    pandas.DataFrame : Table containing columns R, z, a_R, a_z with the 
                       positions and force components
    """
    
    Rs = np.linspace(1e-5, boxsize/2+1e-5,int((boxsize/2)/res)+1)
    zs = np.linspace(1e-5, boxsize/2+1e-5,int((boxsize/2)/res)+1)
    aR = []
    az = []
    final_R = []
    final_z = []
    count = 1
    for R in Rs:
        if (100*R)/max(Rs) - count >= 0:
            print(f'{count}% complete')
            count += 1
        final_z.extend(zs)
        final_R.extend(np.full(len(zs),R))
        aR.extend(gp.evaluateRforces(pot,R*u.kpc,zs*u.kpc))
        az.extend(gp.evaluatezforces(pot,R*u.kpc,zs*u.kpc))
    final_z = np.array(final_z)
    final_R = np.array(final_R)
    aR = np.array(aR)#*conversion.force_in_kmsMyr(vo=220.,ro=8.)
    az = np.array(az)#*conversion.force_in_kmsMyr(vo=220.,ro=8.)
    if arepo_units:
        aR /= int(365*3600*240)
        az /= int(365*3600*240)
    df = pd.DataFrame({'R':final_R,
                        'z':final_z,
                        'a_R':aR,
                        'a_z':az})
    return df



def get_softening_stellar(hdf5_path, c, f_bound = 0.556):
    with h5py.File(hdf5_path, 'r') as hf:
        raw_coords = np.array(hf['PartType3']['Coordinates'])
        raw_stel_coords = np.array(hf['PartType1']['Coordinates'])
    coords = [[],[],[]]
    stel_coords = [[],[],[]]
    # c_divider = int(len(raw_coords)/3)
    # sc_divider = int(len(raw_stel_coords)/3)
    coords[0].append(raw_coords[0::3])
    coords[1].append(raw_coords[1::3]) 
    coords[2].append(raw_coords[2::3])
    stel_coords[0].append(raw_stel_coords[0::3])
    stel_coords[1].append(raw_stel_coords[1::3]) 
    stel_coords[2].append(raw_stel_coords[2::3])
    coords = np.array(coords).squeeze()
    stel_coords = np.array(stel_coords).squeeze()
    recoords = np.rollaxis(coords, axis=1)
    stel_recoords = np.rollaxis(stel_coords, axis=1)
    radii_3d = np.sqrt(np.sum((recoords)**2, axis = 1))
    stel_radii_3d = np.sqrt(np.sum((stel_recoords)**2, axis = 1))
    pdf, bins_edges = np.histogram( 
                    np.append(radii_3d, stel_radii_3d),        
                    bins=1000,   
                    density=True)
    cdf = np.cumsum(pdf*np.diff(bins_edges)) # CMF in 2D LOS
    bin_centers = []
    for k in range(len(pdf)):
        bin_centers.append((bins_edges[k]+bins_edges[k+1])/2) 
    bin_centers = np.array(bin_centers)
    r_half = bin_centers[cdf >= max(cdf)/2][0]
    if f_bound is not None:
        r_half = r_half*0.7*f_bound**0.5
    def nfw_f(x):
        return np.log10(1+x) - x/(1+x)
    return r_half*nfw_f(c)/(0.62 * c**(1.26))
    

def transform_coords(raw_coords, raw_vels, SF_com):
    
    sqrt_rs = np.sqrt(np.sum(raw_coords*raw_coords, axis=1))
    unit_rs = raw_coords/np.rollaxis(np.array([sqrt_rs, sqrt_rs, sqrt_rs]),1,0)
    los_vels_3d = np.sum(raw_vels*unit_rs, axis =1)
    com_rad = np.sqrt(np.sum(SF_com*SF_com))
    unit_com = SF_com/com_rad
    com_proj_rs = raw_coords.dot(unit_com)
    los_vels_2d = raw_vels.dot(unit_com) 
    new_proj_rs = unit_com*com_proj_rs[:,np.newaxis]
    rejec_rs = raw_coords-new_proj_rs
    radii_2d = np.sqrt(np.sum((rejec_rs)**2, axis = 1)) # final 2d radii from dwarf COM (rejection from LOS at GC)
    radii_3d = np.sqrt(np.sum((raw_coords - SF_com)**2, axis = 1)) # final 3d radii from dwarf COM (true radii)
    
    return radii_2d, radii_3d, los_vels_2d, los_vels_3d




def get_rotation_matrix(los_dir):
    """
    Compute a rotation matrix that aligns the los_dir with the z-axis.
    """
    los_dir = los_dir / np.linalg.norm(los_dir)
    
    # If LOS is already aligned with z-axis, no need to rotate
    if np.allclose(los_dir, [0, 0, 1]):
        return np.eye(3)
    
    # Define the z-axis
    z_axis = np.array([0, 0, 1])
    
    # Compute the rotation axis (cross product of LOS and z-axis)
    rotation_axis = np.cross(los_dir, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # Compute the angle between LOS and z-axis
    cos_theta = np.dot(los_dir, z_axis)
    sin_theta = np.linalg.norm(rotation_axis)
    
    # Use Rodrigues' rotation formula to get the rotation matrix
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]])
    
    rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    return rotation_matrix



def sphericity3d(positions, masses_particle):
    """
    Calculate the axis ratios of a 3D mass distribution using the inertia tensor.
    
    This function computes the eigenvalues of the normalized 3D inertia tensor
    to determine the shape parameters (axis ratios) of the 3D mass distribution.
    A perfectly spherical system would have equal eigenvalues, while a flattened
    or elongated system will have one or two eigenvalues smaller than the others.
    
    Parameters:
    -----------
    positions : array_like
        3D array with shape (n, 3) containing x, y, z coordinates of particles
    masses_particle : float
        Mass per particle (assumed constant for all particles)
        
    Returns:
    --------
    array : Sorted eigenvalues of the normalized inertia tensor, which represent
           the axis ratios of the mass distribution (minor to major)
    """
    inertia_norm = np.empty((3, 3))

    totalmass = len(positions)*masses_particle

    # Compute the normalized inertia tensor
    for i in range(0, 3):
        for j in range(0, 3):
            inertia_norm[i, j] = (1.0/totalmass) * \
                np.sum(masses_particle * \
                positions[:, i] * positions[:, j] / (positions[:, 0]**2.0 + positions[:, 1]**2.0 + positions[:, 2]**2.0))

    # Compute eigenvalues and eigenvectors
    w_norm = np.linalg.eig(inertia_norm)

    # Store sorted eigenvectors and eigenvalues
    vector = np.zeros((3, 3))
    ellipsoid_norm = np.sqrt(w_norm[0])
    sort = np.argsort(ellipsoid_norm)

    vector[0, :] = w_norm[1][:, sort[0]]  # minor axis
    vector[1, :] = w_norm[1][:, sort[1]]  # intermediate axis
    vector[2, :] = w_norm[1][:, sort[2]]  # major axis

    return ellipsoid_norm[sort]


def support(positions, veldif, masses_particle):
    """
    Calculate rotational support metrics for a stellar/dark matter system.
    
    This function computes two important metrics of rotational support:
    1. fcorot: Fraction of particles co-rotating with the total angular momentum
    2. kapparot: Rotational kinetic energy divided by total kinetic energy
    
    Parameters:
    -----------
    positions : array_like
        3D array of particle positions relative to center
    veldif : array_like
        3D array of particle velocities relative to center
    masses_particle : float
        Mass per particle (assumed constant for all particles)
        
    Returns:
    --------
    tuple : (fcorot, kapparot)
            fcorot: Fraction of co-rotating particles (0-1)
            kapparot: Fraction of kinetic energy in rotation (0-1)
    """
    mass_star = masses_particle
    
    # Calculate total angular momentum
    totangmom = np.sum(mass_star * np.cross(positions, veldif), axis=0)
    
    # Calculate specific angular momentum for each particle
    specang = np.cross(positions, veldif)
    
    # Normalize the total angular momentum vector
    normaltotang = totangmom / np.sqrt(np.sum(totangmom**2))
    
    # Calculate angular momentum along the total angular momentum axis
    jz = np.dot(specang, normaltotang)
    
    # Calculate fraction of co-rotating particles (jz > 0)
    fcorot = float(np.size(np.where(jz > 0.0))) / float(np.size(jz))
    
    # Define axis aligned with total angular momentum
    Laxisup = normaltotang * 100.0
    Laxisdown = normaltotang * (-100.0)
    
    # Calculate cylindrical radius for each particle
    d = np.cross((positions - Laxisup), (positions - Laxisdown))
    dtop = np.sqrt(np.sum(d**2, axis=1))
    bot = Laxisdown - Laxisup
    bottom = np.sqrt(np.sum(bot**2))
    Rxy = dtop / bottom
    
    # Calculate total kinetic energy
    tot_kin_energ = np.sum(0.5 * mass_star * np.sum(veldif**2, axis=1))
    
    # Calculate rotational kinetic energy and divide by total
    kapparot = (np.sum((mass_star/2.0) * (jz/Rxy)**2.0)) / tot_kin_energ
    
    return fcorot, kapparot
    
    
def sphericity2d(positions, masses_particle):
    """
    Calculate the axis ratio of a 2D mass distribution using the inertia tensor.
    
    This function computes the eigenvalues of the normalized 2D inertia tensor
    to determine the shape parameters (axis ratios) of the 2D mass distribution.
    
    Parameters:
    -----------
    positions : array_like
        2D array with shape (n, 2) containing x, y coordinates of particles
    masses_particle : float
        Mass per particle (assumed constant for all particles)
        
    Returns:
    --------
    array : Sorted eigenvalues of the normalized inertia tensor, which represent
           the axis ratios of the mass distribution (minor to major)
    """
    inertia_norm = np.empty((2, 2))

    totalmass = len(positions) * masses_particle

    # Compute the normalized inertia tensor
    for i in range(0, 2):
        for j in range(0, 2):
            inertia_norm[i, j] = (1.0/totalmass) * \
                np.sum(masses_particle * \
                positions[:, i] * positions[:, j] / (positions[:, 0]**2.0 + positions[:, 1]**2.0))

    # Compute eigenvalues and eigenvectors
    w_norm = np.linalg.eig(inertia_norm)

    vector = np.zeros((2, 2))
    ellipsoid_norm = np.sqrt(w_norm[0])
    sort = np.argsort(ellipsoid_norm)

    # Store sorted eigenvectors
    vector[0, :] = w_norm[1][:, sort[0]]  # minor axis direction
    vector[1, :] = w_norm[1][:, sort[1]]  # major axis direction
    
    # Return sorted eigenvalues (axis ratios)
    return ellipsoid_norm[sort]


def calcmedquartnine(array):
    """
    Calculate various percentiles of a distribution.
    
    This function computes the median, 16th and 84th percentiles (1-sigma), 
    2.5th and 97.5th percentiles (2-sigma), and 0.15th and 99.85th percentiles (3-sigma)
    of the input array.
    
    Parameters:
    -----------
    array : array_like
        Input data array
        
    Returns:
    --------
    tuple : (median, 16%, 84%, 2.5%, 97.5%, 0.15%, 99.85%) percentiles
            representing the median and the various confidence intervals
    """
    index = np.argsort(array, axis=0)
    median = np.median(array)
    
    # 1-sigma (68% confidence interval)
    sixlow = np.percentile(array, 16)
    sixhigh = np.percentile(array, 84)
    
    # 2-sigma (95% confidence interval)
    ninelow = np.percentile(array, 2.5)
    ninehigh = np.percentile(array, 97.5)
    
    # 3-sigma (99.7% confidence interval)
    nineninelow = np.percentile(array, 0.15)
    nineninehigh = np.percentile(array, 99.85)

    return median, sixlow, sixhigh, ninelow, ninehigh, nineninehigh, nineninelow

def calc_real_vlos2_error(R, vz, vzerr, prob, nmonte=1000, Nbin=None):
    """
    Calculate the line-of-sight velocity dispersion and its error using Monte Carlo sampling.
    
    This function computes the line-of-sight velocity dispersion from radially binned 
    data, accounting for measurement errors through Monte Carlo simulations. It correctly
    handles probability weights for each point and subtracts measurement errors from the 
    observed dispersion.
    
    Parameters:
    -----------
    R : array_like
        Radial distances of stars/particles
    vz : array_like
        Line-of-sight velocities
    vzerr : array_like
        Measurement errors on line-of-sight velocities
    prob : array_like
        Probability weights for each star/particle
    nmonte : int, optional
        Number of Monte Carlo realizations (default: 1000)
    Nbin : int, optional
        Number of particles per radial bin. If None, estimated from data size.
        
    Returns:
    --------
    tuple : (radial_bins, velocity_dispersion, dispersion_error)
            containing the radial bin positions, velocity dispersion, and 
            uncertainties on the dispersion
    """
    # Determine bin size based on data size if not specified
    if Nbin is None:
        Nbin = int(len(prob)/np.sqrt(len(prob)))
        
    # Sort data by radius
    index = np.argsort(R)  # sort all kinematics positions by distance

    # Calculate the probability-weighted mean line-of-sight velocity
    vzmean = np.sum(vz * prob) / np.sum(prob)  # center-of-mass velocity
    vzmeanerr = 0.  # Not used but initialized for potential future use

    # Initialize arrays for radial binning
    cnt = 0
    jsum = 0.0
    norm = np.zeros(len(R))
    vlos2med = np.zeros(len(R))
    rbin_tmp = np.zeros(len(R))
    
    # First calculate and subtract the mean vz:  prob ~ number contribution
    for i in range(len(R)):  # fill bins with specified stars per bin
        if jsum < Nbin:
            vlos2med[cnt] = vlos2med[cnt] + (vz[index[i]] - vzmean) ** 2. * prob[index[i]]  # number weighted 2nd moment
            rbin_tmp[cnt] = R[index[i]]  # where is the furthest star in bin?
            jsum = jsum + prob[index[i]]
        if jsum >= Nbin:
            norm[cnt] = jsum
            jsum = 0.0
            cnt = cnt + 1
    vlos2med = vlos2med[:cnt]
    norm = norm[:cnt]  # normalize with tot stars
    vlos2med = vlos2med / norm
    rbin_tmp = rbin_tmp[:cnt]

    # And Monte-Carlo the errors:
    vlos2 = np.zeros((nmonte, len(R)))
    vlos2_pureerr = np.zeros((nmonte, len(R)))
    norm = np.zeros(len(R))
    for k in range(nmonte):
        cnt = 0
        jsum = 0.0
        for i in range(len(R)):
            vz_err = (vz[index[i]] - vzmean) + np.random.normal(0.0, vzerr[index[i]])  # perturb velocity with 2 km/s
            vz_pure_err = np.random.normal(0.0, vzerr[index[i]])
            if jsum < Nbin:
                vlos2[k, cnt] = vlos2[k, cnt] + vz_err ** 2. * prob[index[i]]
                vlos2_pureerr[k, cnt] = vlos2_pureerr[k, cnt] + vz_pure_err ** 2. * prob[index[i]]
                jsum = jsum + prob[index[i]]
            if jsum >= Nbin:
                norm[cnt] = jsum
                jsum = 0.0
                cnt = cnt + 1

    vlos2tmp = np.zeros((nmonte, cnt))
    vlos2tmp = vlos2[:, :cnt]
    vlos2_pe_tmp = np.zeros((nmonte, cnt))
    vlos2_pe_tmp = vlos2_pureerr[:, :cnt]
    norm = norm[:cnt]

    vlos2 = vlos2tmp / norm
    vlos2_pe = vlos2_pe_tmp / norm

    # And now estimate the full measurement error:
    vlos2err_meas = np.zeros(cnt)
    vlos2_pe_meas = np.zeros(cnt)

    for k in range(cnt):
        median, sixlow, sixhigh, ninelow, ninehigh, nineninehigh, nineninelow = calcmedquartnine(vlos2[:, k])
        vlos2err_meas[k] = (sixhigh - sixlow) / 2.0
        median, sixlow, sixhigh, ninelow, ninehigh, nineninehigh, nineninelow = calcmedquartnine(vlos2_pe[:, k])
        vlos2_pe_meas[k] = (sixhigh - sixlow) / 2.0

    # Combine with the Poisson error:
    vlos2err = np.sqrt(vlos2err_meas ** 2.0 + vlos2med ** 2.0 / Nbin)

    vlos2med = vlos2med - vlos2_pe_meas  # subtract pure error

    # Demand positive:
    vlos2err = vlos2err[vlos2med > 0]
    rbin_tmp = rbin_tmp[vlos2med > 0]
    vlos2med = vlos2med[vlos2med > 0]
    

    vlos2err = vlos2err/2.0/np.sqrt(vlos2med)
    vlos2med = np.sqrt(vlos2med)
    return rbin_tmp, vlos2med, vlos2err
