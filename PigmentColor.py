from math import floor, exp, pi, log
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing as mp
import Fluxes

KB = 1.38E-23 #Boltzmann constant in [m2 kg s-2 K-1]
H = 6.626E-34 #Planck constant in [J s]
C = 2.99792458E8 #Speed of light [m s-1]                                          
SIGMA = 5.67E-8 #Stefan-Boltzmann constant
AU = 1.496E11 #1 AU in [m]
SUN_RAD = 6.957E8 #solar radius [m]
joule_to_eV_conv = 6.2415091E18 #convert from Joules to eV


def calc_u_naught(T_star, T_sur, R_star, R_orbit, nu, nu_width, phi=0.333):
    """
    Calculate the u0 value from Bjorn (1976). 
    """

    star_flux = planck_function_frequency_photons(T_star, nu+nu_width, nu-nu_width)*\
            (R_star/R_orbit)**2/4.0
    #print("at nu: %0.2f - star flux is: %2.3e"%(nu,star_flux))
    plant_flux = planck_function_frequency_photons(T_sur, nu+nu_width, nu-nu_width)

    #bjorn_u0 = KB*T_sur*log(phi*R_star**2/R_orbit**2)+H*nu*(1.0-T_sur/T_star)
    u0 = KB*T_sur*log(phi*star_flux/plant_flux+1.0)

    #print("nu: %2.3e - bjorn_u0=%2.3e, u0=%2.3e (error: %f)"%(nu,bjorn_u0,u0,(u0-bjorn_u0)/bjorn_u0))

    return u0

def get_plant_flux(T, start, end, resolution=100):
    """
    Get the flux emitted by the surface plants in a blackbody approximation.

    Inputs:
    T - the surface temperature (temperature of the plants) [K]
    start - the start frequency for the calculations [Hz]
    end - the end frequency for the calculations [Hz]
    resolution - the number of intervals between start and end

    Returns:
    flux - the total flux from the plants [Photons m-2 s-1]
    """

    freqs = np.linspace(start,end,resolution)

    fluxes = Fluxes.get_blackbody_photon_flux_for_frequencies(T,freqs)
    flux = Fluxes.total_flux(freqs, fluxes)

    return flux


def get_incident_flux(TYPE, start, end, star_rad=1.0, orbital_rad=1.0, T=-1, \
        resolution=100):
    """
    This function will return the flux incident on a planet in the range 
    start to end. The incident flux can be scaled by the orbital distance of the
    planet. This must be done in the blackbody approximation (TYPE=0), i.e. a
    star radius and orbital radius must be specified.

    Input:
    TYPE - the flux you'd like. The options are:
        0 - use a blackbody approximation for the flux
        1 - use the flux from the Sun at Earth's orbital distance at the surface
        2 - use the flux from the Sun at Earth's orbital distance at the TOA
    star_rad - the radius of the star [m]
    orbital_rad - the orbital radius of the planet [m]
    T - the temperature of the star [K]
    start - The starting frequency to use [Hz]
    end - the ending frequency to use [Hz]
    resolution - the number of steps to use between start and end

    Returns:
    flux - the total flux between start and end [Photons m-2 s-1] 
    """

    if end < start:
        print("Error in get_incident_flux(): end < start.")
        sys.exit(1)

    flux = 0.0
    freqs = np.linspace(start,end,resolution)

    if TYPE==0:
        #this is the blackbody approximation, make sure we have everything
        #needed to calculate the photon flux
        if star_rad==1.0 or orbital_rad==1.0 or T<0:
            print("Error in get_incident_flux(): invalid inputs for TYPE=0.")
            sys.exit(1)

        #the necessary parameters are provided, continue
        fluxes = Fluxes.get_blackbody_photon_flux_for_frequencies(T,freqs,\
                star_rad,orbital_rad)
        flux = Fluxes.total_flux(freqs, fluxes)

    elif TYPE==1:
        #get the total flux for the Earth's surface
        measured_freqs, measured_fluxes = Fluxes.get_earth_surface_flux()
        calc_fluxes = np.zeros_like(freqs)
        for i in range(0,len(freqs)):
            calc_fluxes[i] = Fluxes.flux_at_nearest_freq(freqs[i], \
                    measured_freqs, measured_fluxes)

    
    return flux










def simple_pigment_calc(T_star, R_star, R_orbit, PLOT=False):
    """
    Simple calculation of the optimal pigment color following the calculation 
    by Bjorn (1976). 

    Input:
    T_star - the temperature of the star [K]
    R_star - the radius of the star [m]
    R_orbit - the orbital distance of the planet [m]

    Returns:
    pig - the optimal pigment absorption wavelength [nm]
    """

    #ORL TODO - this pigment width needs to be justified/explored as a parameter
    abs_width = 1.0 #the half-width of the pigment absorption line [Hz-1]
    start_nu = 1.5E15 #start calculations at 200 nm [Hz-1]
    end_nu = 1.5E14 #end calculations at ~2000 nm [Hz-1]
    T_sur = 300.0 #planet surface temp [K]

    
    nus = np.linspace(end_nu, start_nu, 1000)
    powers = np.zeros_like(nus)
    us = np.zeros_like(nus)
    u0s = np.zeros_like(nus)
    u_effs = np.zeros_like(nus)
    for i, nu in enumerate(nus):
        #calculate the flux at the planet, this has to be scaled by star size and
        #planet orbital distance
        val = planck_function_frequency_photons(T_star,nu+abs_width, nu-abs_width)*\
                (R_star/R_orbit)**2/4.0
        photon_val = val/H*nu
        u0 = calc_u_naught(T_star, T_sur, R_star, R_orbit, nu, abs_width)
        u = u0+KB*T_sur*log(KB*T_sur/(u0+KB*T_sur)) #KB*T_sur*log(KB*T_sur/(u0+KB*T_sur)) #the approximation of equation 9 in Bjorn (1976)
        u_eff = u/(1.0+KB*T_sur/u) 
        us[i] = u
        u0s[i] = u0
        u_effs[i] = u_eff
        power = u_eff*val #u_eff*nu**2*exp(-H*nu/(KB*T_star))
        powers[i] = power


        bjorn_val = nu**2*exp(-H*nu/(KB*T_star))
        #print("%d: val=%2.3e (Bjorn val=%2.3e) (error: %f) ueff=%2.3e, photon_val=%2.3e"%(i,val,bjorn_val,(val-bjorn_val)/bjorn_val, u_eff, photon_val))

    peak = C/nus[powers.argmax()]/1.0E-6
    if PLOT:
        print("peak at: %0.4f microns"%(peak))
        #plt.plot(nus,u0s*joule_to_eV_conv,label="$u_{0}$")
        #plt.plot(nus,us*joule_to_eV_conv,label="$u$")
        #plt.plot(nus,u_effs*joule_to_eV_conv,label="$u_{eff}$")
        #plt.legend()
        plt.plot(C/nus/1.0E-6,powers)
        plt.show()

    return peak


def plot_pigments_over_temp(temps, pigs):

    height = 0.06
    base = np.min(pigs)-height

    plt.gca().add_patch(patches.Rectangle((2500, base),1200,height, alpha=0.5,\
            edgecolor="none", facecolor="red"))
    plt.gca().add_patch(patches.Rectangle((3700, base),1500,height, alpha=0.5,\
            edgecolor="none", facecolor="orange"))
    plt.gca().add_patch(patches.Rectangle((5200, base),800,height, alpha=0.5,\
            edgecolor="none", facecolor="yellow"))
    plt.gca().add_patch(patches.Rectangle((6000, base),1500,height, alpha=0.5,\
            edgecolor="none", facecolor="#ffff99"))

    plt.gca().text(3100,base+height/2.0,"M", verticalalignment="center", horizontalalignment="center")
    plt.gca().text(4450,base+height/2.0,"K", verticalalignment="center", horizontalalignment="center")
    plt.gca().text(5600,base+height/2.0,"G", verticalalignment="center", horizontalalignment="center")
    plt.gca().text(6750,base+height/2.0,"F", verticalalignment="center", horizontalalignment="center")

    plt.plot((2550,2550),(base+height,np.max(pigs)), "k:", linewidth=2, label="TRAPPIST-1")
    plt.plot((3042,3042),(base+height,np.max(pigs)), "k--", linewidth=2, label="Proxima Centauri")
    plt.plot((5778,5778),(base+height,np.max(pigs)), "k-.", linewidth=2, label="Sun")

    cutoff_pig = 0
    for i in range(0,len(pigs)):
        if pigs[i] < 1.0:
            cutoff_pig = temps[i]
            break

    plt.gca().add_patch(patches.Rectangle((np.min(temps),base+height),\
            cutoff_pig-np.min(temps), np.max(pigs)-base-height,alpha=0.5,\
            edgecolor='none',facecolor="purple"))


    plt.plot(temps,pigs, "k", linewidth=2)
    plt.xlim(np.min(temps),np.max(temps))
    plt.ylim(base,np.max(pigs))
    plt.xlabel("Stellar Temperature [K]")
    plt.ylabel(r"Optimal Pigment Absorption [$\mathrm{\mu}$m]")
    plt.legend()
    plt.show()

def plot_opt_pig_over_temp():
    num_temps = 40
    temps = np.linspace(2500, 7500, num_temps)
    opt_pigs = np.zeros(num_temps)

    for i in range(0,num_temps):
        star_rad = (0.00018647*temps[i]+0.00825597)*SUN_RAD
        orb_rad = (star_rad**2*SIGMA*temps[i]**4/1366.0)**0.5
        opt_pigs[i] = simple_pigment_calc(temps[i],star_rad,orb_rad)
        print("%d: T=%4.0f, Star=%1.2f, orb=%1.4f"%(i,temps[i],star_rad/SUN_RAD,orb_rad/AU))

    plt.plot(temps,opt_pigs)
    plt.show()


def mp_simple_pig(T):
    star_rad = (0.00018647*T+0.00825597)*SUN_RAD
    orb_rad = (star_rad**2*SIGMA*T**4/1366.0)**0.5
    print("T=%4.0f, Star=%1.2f, orb=%1.4f"%(T,star_rad/SUN_RAD,orb_rad/AU))
    return simple_pigment_calc(T,star_rad,orb_rad)


def multithreaded_plot_opt_pig_over_temp():
    if __name__ == '__main__':
        temps = np.linspace(2500,7500,20)
        pool = mp.Pool(mp.cpu_count())
        opt_pigs = pool.map(mp_simple_pig, temps)

        plot_pigments_over_temp(temps,opt_pigs)

#simple_pigment_calc(2500.0, SUN_RAD, AU, PLOT=True)
#plot_opt_pig_over_temp()
multithreaded_plot_opt_pig_over_temp()

def temp_rad_comp():
    """
    This is just a test function to fit the data from wikipedia because I was
    too lazy to do it by hand...
    """
    #data from https://en.wikipedia.org/wiki/Stellar_classification table
    temps = [7500.0,6000.0,5200.0,3700.0]
    rads = [1.4,1.15,0.96,0.7]

    res = np.polyfit(temps,rads,1)
    print(res)

    plt.plot(temps,rads)
    plt.show()


