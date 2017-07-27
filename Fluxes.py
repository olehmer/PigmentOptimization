import numpy as np
import matplotlib.pyplot as plt
from math import exp, pi, floor
import sys

H = 6.626E-34 #Planck constant in [J s]
C = 2.99792458E8 #Speed of light [m s-1]                                          
KB = 1.38E-23 #Boltzmann constant in [m2 kg s-2 K-1]
AU = 1.496E11 #1 AU in [m]
SUN_RAD = 6.957E8 #solar radius [m]

def flux_at_nearest_freq(nu, freqs, fluxes):
    """
    This function will return the index of the nearest frequency in the freqs 
    array to the passed in frequency, nu.

    Inputs:
    nu - the frequency you're looking for [Hz]
    freqs - the array of frequencies [Hz]
    fluxes - the array of fluxes [Photons m-2 s-1 Hz-1]

    Returns:
    flux - the nearest measured flux to nu [Photons m-2 s-1 Hz-1]
    """

    ind = (np.abs(freqs-nu)).argmin()
    flux = fluxes[ind]
    return flux



def blackbody_photon_flux(T, nu):
    """
    Return the total photon flux in photons m-2 s-1 for a blackbody at temperature
    T for the given frequency, nu. 

    Parameters:
    T - the temperature of the blackbody [K]
    nu - the frequency at which to calculate the photon flux [Hz]

    Returns:
    flux - the photon flux at nu [Photons m-2 s-1 Hz-1]
    """

    flux = 2.0*H*nu**3/C**2/(exp(H*nu/(KB*T))-1.0)/(H*nu)
    return flux*pi

def get_blackbody_photon_flux_for_frequencies(T, freqs, \
        star_rad=1.0, orbital_rad=1.0):
    """
    Get the photon fluxes for an array of frequency values.

    NOTE: this function can be used for calculating the flux a planet receives
          at an orbital distance of orbital_rad. 

    Inputs:
    T - the temperature of the blackbody [K]
    freqs - array of frequencies to calculate fluxes for [Hz]
    star_rad - the radius of the star [m]
    orbital_rad - the radius of the orbit

    Returns:
    fluxes - an array of photon fluxes for the corresponding frequency values
             in [Photons m-2 s-1 Hz-1]
    """

    fluxes = np.zeros_like(freqs)
    for i in range(0,len(freqs)):
        fluxes[i] = blackbody_photon_flux(T, freqs[i])*(star_rad/orbital_rad)**2

    return fluxes


def total_flux(freqs, fluxes):
    """
    This function will take an array of frequencies and an array of fluxes then
    return the total flux of the fluxes array.

    Inputs:
    freqs - the array of frequency values [Hz]
    fluxes - the array of flux values [Photons m-2 s-1 Hz-1]

    Returns:
    t_flux - the summed fluxes from the fluxes array [Photons m-2 s-1]
    """

    t_flux = 0.0

    width = 0.0
    for i in range(0,len(freqs)-1):
        width = abs(freqs[i]-freqs[i+1])
        t_flux += width*fluxes[i]

    #add the last flux measurement using the width of the previous freq.
    t_flux += width*fluxes[-1]

    return t_flux

def get_earth_surface_flux():
    """
    Get the frequency and flux array for the surface of the Earth

    Returns:
    freqs - the array of frequency measurements [Hz]
    photons - the array of photon flux measurements [photons m-2 s-1 Hz-1]
    """

    wavelengths, toa_flux, sur_flux = read_solar_flux_data()
    freqs, fluxes, photons = convert_wavelength_flux_array_to_freq(wavelengths,\
            sur_flux)
    return (freqs, photons)


def convert_wavelength_flux_array_to_freq(wvs, wv_flux):
    """
    Convert an array of wavelengths in nm to frequency in Hz and the corresponding
    array of flux measurements in W m-2 nm-1 to W m-2 Hz-1. Also return the flux
    in Photons m-2 Hz-1
    """

    freqs = np.zeros_like(wvs)
    fluxes = np.zeros_like(wvs)
    photons = np.zeros_like(wvs)

    for i in range(0,len(wvs)):
        freqs[i] = C/(wvs[i]*1.0E-9)
        fluxes[i] = wv_flux[i]*(1.0E9)*(wvs[i]*1.0E-9)**2/C
        photons[i] = fluxes[i]/(H*freqs[i])

    return (freqs, fluxes, photons)


def read_solar_flux_data():
    """
    Plot the data from http://rredc.nrel.gov/solar/spectra/am1.5/

    See the wikipedia plot of the same data at: https://commons.wikimedia.org/wiki/File:Solar_Spectrum.png

    This function returns the wavelengths array, the array of corresponding 
    fluxes at TOA and the corresponding fluxes at the surface.
    """
    filename = "./ASTMG173.csv"
    data = np.loadtxt(filename,delimiter=",", skiprows=2)
    wavelengths = data[:,0]
    flux = data[:,1]
    flux_tilt = data[:,2]
    flux_circ = data[:,3]

    return (wavelengths, flux, flux_circ)

def plot_read_data():
    wavelengths, flux, flux_circ = read_solar_flux_data()

    f, axarr = plt.subplots(3,1)

    axarr[0].plot(wavelengths,flux, label="TOA Flux")
    #axarr[0].plot(wavelengths, flux_tilt, label="Global Tilt")
    axarr[0].plot(wavelengths, flux_circ, label="Surface Flux")
    axarr[0].legend()
    #axarr[0].set_xlim(200,2500)
    axarr[0].set_title("Power per wavelength interval")
    axarr[0].set_xlabel("Wavelength [nm]")
    axarr[0].set_xlim(np.min(wavelengths),np.max(wavelengths))

    freqs, freq_flux, photon_flux = convert_wavelength_flux_array_to_freq(wavelengths,flux)
    freqs, freq_flux_sur, photon_flux_sur = convert_wavelength_flux_array_to_freq(wavelengths,flux_circ)

    axarr[1].plot(freqs,freq_flux, label="TOA Flux")
    axarr[1].plot(freqs,freq_flux_sur, label="Surface Flux")
    axarr[1].set_title("Power per frequency interval")
    axarr[1].set_xlabel("Frequency [Hz]")
    axarr[1].legend()
    axarr[1].set_xlim(np.min(freqs), np.max(freqs))

    
    #test the blackbody function
    planck_flux_photons = get_blackbody_photon_flux_for_frequencies(5780.0,\
            freqs, star_rad=SUN_RAD, orbital_rad=AU)

    axarr[2].plot(freqs, photon_flux, label="TOA Flux")
    axarr[2].plot(freqs, photon_flux_sur, label="Surface Flux")
    axarr[2].plot(freqs, planck_flux_photons, label="Blackbody")
    axarr[2].set_title("Photons per frequency interval")
    axarr[2].set_xlabel("Frequency [Hz]")
    axarr[2].legend()
    axarr[2].set_xlim(np.min(freqs), np.max(freqs))

    plt.show()

