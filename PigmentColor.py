from math import floor, exp, pi, log
import numpy as np
import matplotlib.pyplot as plt

KB = 1.38E-23 #Boltzmann constant in [m2 kg s-2 K-1]
H = 6.626E-34 #Planck constant in [J s]
C = 2.99792458E8 #Speed of light [m s-1]                                          
SIGMA = 5.67E-8 #Stefan-Boltzmann constant
AU = 1.496E11 #1 AU in [m]
SUN_RAD = 6.957E8 #solar radius [m]

def planck_function(T, lambda1, lambda2):
    """
    Return the total flux in W/m2 for a blackbody at temperature T between 
    the wavelengths of lambda1 and lambda2. NOTE: lambda1 < lambda2 

    The returned flux is scaled by pi to account for the flux from a hemisphere
    of isotropic radiation.

    Parameters:
    T - the temperature of the body
    lambda1 - the first wavelength to consider given in nm
    lambda2 - the final wavelength to consider given in nm

    Returns:
    Bv - the total flux in the wavelength interval [W m-2]
    """

    v1 = floor(lambda1)
    v2 = floor(lambda2)
    if v2-v1 < 1:
        return 0

    vs = np.linspace(v1,v2,(v2-v1)+1) 
    Bv = np.zeros(len(vs))

    c1 = 2.0*H*C**2
    c2 = H*C/(KB*T)
    for i in range(0,len(Bv)):
        wavelength = vs[i]*(1.0E-9) #convert to meters
        Bv[i] = c1/wavelength**5*1.0/(exp(c2/wavelength)-1.0)*(1.0E-9)

    return pi*sum(Bv) #scale by pi to account for flux over hemisphere



def calc_u_naught(T_star, T_sur, R_star, R_orbit, wv, wv_width, phi=0.333):
    """
    Calculate the u0 value from Bjorn (1976). This value is calculated assuming
    a blackbody approximation here.
    """

    star_flux = planck_function(T_star, wv-wv_width, wv+wv_width)*\
            (R_star/R_orbit)**2
    plant_flux = planck_function(T_sur, wv-wv_width, wv+wv_width)

    u0 = KB*T_sur*log(phi*star_flux/plant_flux+1.0)

    return u0



def simple_pigment_calc(T_star, R_star, R_orbit):
    """
    Simple calculation of the optimal pigment color following the calculation 
    by Bjorn (1976). We assume the star emits as a blackbody.

    Input:
    T_star - the temperature of the star [K]
    R_star - the radius of the star [m]
    R_orbit - the orbital distance of the planet [m]

    Returns:
    pig - the optimal pigment absorption wavelength [nm]
    """

    #ORL TODO - this pigment width needs to be justified/explored as a parameter
    abs_width = 10.0 #the half-width of the pigment absorption line [nm]
    start_wavelength = 200.0 #start calculations at 200 [nm]
    end_wavelength = 1000.0 #end calculations at 1000 [nm]
    T_sur = 300.0 #planet surface temp [K]

    
    wavelengths = np.linspace(start_wavelength,end_wavelength,\
            end_wavelength-start_wavelength)
    powers = np.zeros_like(wavelengths)
    us = np.zeros_like(wavelengths)
    u0s = np.zeros_like(wavelengths)
    u_effs = np.zeros_like(wavelengths)
    for i, wv in enumerate(wavelengths):
        #calculate the flux at the planet, this has to be scaled by star size and
        #planet orbital distance
        val = planck_function(T_star,wv-abs_width, wv+abs_width)*\
                (R_star/R_orbit)**2
        u0 = calc_u_naught(T_star, T_sur, R_star, R_orbit, wv, abs_width)
        u = KB*T_sur*log(KB*T_sur/(u0+KB*T_sur)) #the approximation of equation 9 in Bjorn (1976)
        u_eff = u/(1.0+KB*T_sur/u) 
        us[i] = u
        u0s[i] = u0
        u_effs[i] = u_eff
        power = u_eff*val
        powers[i] = power

    print(wavelengths[powers.argmax()])
    plt.plot(wavelengths,us,label="u")
    plt.plot(wavelengths,u0s,label="$u_{0}$")
    plt.plot(wavelengths,u_effs,label="$u_{eff}$")
    plt.legend()
    #plt.plot(wavelengths,powers)
    plt.show()


simple_pigment_calc(5800.0, SUN_RAD, AU)



