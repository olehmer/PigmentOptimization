from math import floor, exp, pi, log
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing as mp

KB = 1.38E-23 #Boltzmann constant in [m2 kg s-2 K-1]
H = 6.626E-34 #Planck constant in [J s]
C = 2.99792458E8 #Speed of light [m s-1]                                          
SIGMA = 5.67E-8 #Stefan-Boltzmann constant
AU = 1.496E11 #1 AU in [m]
SUN_RAD = 6.957E8 #solar radius [m]
joule_to_eV_conv = 6.2415091E18 #convert from Joules to eV

def planck_function_frequency_photons(T, nu1, nu2, resolution=1000):
    """
    Return the total photon flux in photons/m^2 for a blackbody at temperature
    T between the frequencies nu1 and nu2. NOTE: nu1>nu2

    Parameters:
    T - the temperature of the star
    nu1 - the higher frequency limit
    nu2 - the lower frequency limit
    resolution - the number of frequencies per sum 

    Returns:
    Bv - the total photon flux in the frequency interval [photons m-2]
    """

    nu1 = floor(nu1)
    nu2 = floor(nu2)

    if nu2 > nu1:
        #this violates the requirements of the function, bad job.
        return 0

    #Reiman sum the Planck function per wavelength interval
    vs = np.linspace(nu2,nu1,resolution)

    Bv = np.zeros(resolution-1)

    for i in range(0,len(Bv)):
        width = vs[i+1]-vs[i]
        Bv[i] = 8.0*pi/C**2*vs[i]**2/(exp(H*vs[i]/(KB*T))-1.0)*width
    return sum(Bv)


def planck_function_frequency(T, nu1, nu2, resolution=1000):
    """
    Return the total flux in W/m^2 for a blackbody at temperature T between the
    frequencies nu1 and nu2. NOTE: nu1>nu2

    Parameters:
    T - the temperature of the star
    nu1 - the higher frequency limit
    nu2 - the lower frequency limit
    resolution - the number of frequencies per sum 

    Returns:
    Bv - the total flux in the frequency interval [W m-2]
    """

    nu1 = floor(nu1)
    nu2 = floor(nu2)

    if nu2 > nu1:
        #this violates the requirements of the function, bad job.
        return 0

    #Reiman sum the Planck function per wavelength interval
    vs = np.linspace(nu2,nu1,resolution)

    Bv = np.zeros(resolution-1)

    for i in range(0,len(Bv)):
        width = vs[i+1]-vs[i]
        Bv[i] = 2.0*H*vs[i]**3.0/C**2/(exp(H*vs[i]/(KB*T))-1.0)*width

    return pi*sum(Bv)




def planck_function_wavelength(T, lambda1, lambda2):
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



def calc_u_naught(T_star, T_sur, R_star, R_orbit, nu, nu_width, phi=0.333):
    """
    Calculate the u0 value from Bjorn (1976). This value is calculated assuming
    a blackbody approximation here.
    """

    star_flux = planck_function_frequency_photons(T_star, nu+nu_width, nu-nu_width)*\
            (R_star/R_orbit)**2/4.0
    #print("at nu: %0.2f - star flux is: %2.3e"%(nu,star_flux))
    plant_flux = planck_function_frequency_photons(T_sur, nu+nu_width, nu-nu_width)

    bjorn_u0 = KB*T_sur*log(phi*R_star**2/R_orbit**2)+H*nu*(1.0-T_sur/T_star)
    u0 = KB*T_sur*log(phi*star_flux/plant_flux+1.0)

    #print("nu: %2.3e - bjorn_u0=%2.3e, u0=%2.3e (error: %f)"%(nu,bjorn_u0,u0,(u0-bjorn_u0)/bjorn_u0))

    return u0



def simple_pigment_calc(T_star, R_star, R_orbit, PLOT=False):
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

    base = np.min(pigs)
    plt.gca().add_patch(patches.Rectangle((2500, base),1200,0.1, alpha=0.5,\
            edgecolor="none", facecolor="red"))
    plt.gca().add_patch(patches.Rectangle((3700, base),1500,0.1, alpha=0.5,\
            edgecolor="none", facecolor="orange"))
    plt.gca().add_patch(patches.Rectangle((5200, base),800,0.1, alpha=0.5,\
            edgecolor="none", facecolor="yellow"))
    plt.gca().add_patch(patches.Rectangle((6000, base),1500,0.1, alpha=0.5,\
            edgecolor="none", facecolor="#ffff99"))

    plt.gca().text(3100,base+0.05,"M", verticalalignment="center", horizontalalignment="center")


    plt.plot(temps,pigs)
    plt.xlim(np.min(temps),np.max(temps))
    plt.ylim(np.min(pigs),np.max(pigs))
    plt.show()

def plot_opt_pig_over_temp():
    num_temps = 20
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


