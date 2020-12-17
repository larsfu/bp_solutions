# This is not a reference solution for the task on sheet 5, as the ODE integration
# is not done “manually”. Instead, this serves as an introduction to scipy’s solve_ivp.

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.integrate import solve_ivp
import matplotlib.patches as patches
import matplotlib.colors as colors


# Representation of the ordinary differential equation system for a simplified linear
# accelerator. I slightly rearranged the equations from the sheet, as integrating over
# time seems more generalized and using momentum instead of energy allows simpler
# handling of negative velocities. (On the other hand force is dp/dt and this is nice)
def linac_momentum(t, y, m, E, q, k, L):
    # m, E, q, k are constant paramters.
    # y is the state vector of the current step, we unpack it for readability
    φ, p, s = y
    # calculate velocity from momentum
    v = p / np.sqrt(m**2 + p**2 / const.c**2)
    dφdt = k * (v - const.c)
    dpdt = q * E * np.cos(φ)
    dsdt = v
    return (dφdt, dpdt, dsdt)


# When using scipy.integrate.solve_ivp, you can define so-called events that may happen
# during the solving process. These are defined by the following functions’ roots.

# We want to terminate the integration process when the particle falls out of the “wrong
# side” of the cavity. Therefore we just return s, so that later the integration can be
# terminated when s crosses zero.
def leave_cavity_back(t, y, *_):
    φ, p, s = y
    return s

# Analogously, this function handles particles that cross the cavity’s other boundary.
def leave_cavity_front(t, y, m, E, q, k, L):
    φ, p, s = y
    return s - L

# Additionally, it might be interesting to see which orbits contain a velocity inversion
# (particles travelling backwards at any point of their path).
def inversion(t, y, *_):
    φ, p, s = y
    return p

# The first two events should terminate the integration, therefore we set this parameter.
leave_cavity_back.terminal = True
leave_cavity_front.terminal = True


@np.vectorize
def solve_one_particle(φ0, W0, m, E, q, k, L):
    # Because I transformed the problem to momentum, we need to calculate the inital momenta
    # from the given initial energies.
    E0 = m * const.c**2
    p0 = np.sqrt((W0**2 + 2 * W0 * E0) / const.c ** 2)

    # Now we call solve_ivp to solve our initial value problem.
    # The paremeters are:
    #   – Our ODE system
    #   – The time range of interest (which will not be exhausted because of our terminal events)
    #   – The initial values (np.nextafter is used so the integration will not terminate instantly)
    #   – A solving method, in this case a Runge-Kutta solver (the results vary a lot)
    #   – Error tolerances (in the case of atol for every quantity)
    #   – The additional, constant arguments of our ODE system
    #   – The event functions
    result = solve_ivp(linac_momentum, (0, 1e-3), (φ0, p0, np.nextafter(0, 1)), method='RK23', 
                       rtol=1e-5, atol=(1e-3, 1e3, 1e-3),
                       args=(m, E, q, k, L), 
                       events=(leave_cavity_front, leave_cavity_back, inversion))

    # If the solver was terminated by the particle leaving the cavity on the right side, take the last
    # momentum value and convert it back to energy. If not, return np.nan.
    if len(result.t_events[0]) == 1:
        return np.sqrt((result.y[1][-1] * const.c)**2 + E0**2) - E0
    else:
        return np.nan


if __name__ == '__main__':
    m = const.m_e  # kg
    E = 12e6  # V/m
    f = 2856e6  # Hz
    q = const.e  # C
    L = 3.05  # m

    k = 2 * np.pi * f / const.c
    φrange = np.array((-np.pi, np.pi))
    Wrange = np.array((0.1 * const.m_e*const.c**2, 15 * const.m_e*const.c**2))
    φ0, W0 = np.meshgrid(np.linspace(*φrange, 36), np.linspace(*Wrange, 10))

    # The @np.vectorize decorator allows us to call the function with numpy arrays
    result = solve_one_particle(φ0, W0, m, E, q, k, L)

    # Plotting begins here

    # We are plotting in polar coordinates, because we feel especially crazy today.
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    # As we want to plot the energy gain with a diverging color map, we need to use
    # a `TwoSlopeNorm` to keep the center of the color map at zero.
    norm = colors.TwoSlopeNorm(vmin=np.nanmin(result-W0) / const.e / 1e6, vcenter=0., vmax=np.nanmax(result-W0) / const.e / 1e6)
    
    cm = ax.pcolormesh(φ0, W0 / const.e / 1e6, (result-W0) / const.e / 1e6, shading='nearest', cmap='RdBu_r', norm=norm)

    # To add to our craziness we put our phi ticks on the inside of the plot
    ax.tick_params(labelleft=True, labelright=False, labeltop=True, labelbottom=False)
    ax.set_xticks((0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi))
    ax.set_xticklabels(('0', r'$\frac{\mathrm{\pi}}{4}$', 
                             r'$\frac{\mathrm{\pi}}{2}$', 
                             r'$\frac{3\mathrm{\pi}}{4}$', 
                             r'$\pm\mathrm{\pi}$', 
                             r'$-\frac{3\mathrm{\pi}}{4}$', 
                             r'$-\frac{\mathrm{\pi}}{2}$', 
                             r'$-\frac{\mathrm{\pi}}{4}$'))
    
    # Axis labels are not flexible enough for this plot, therefore we just put texts.
    plt.text(1.02, .45, '$W_0$ / MeV', transform=ax.transAxes)
    plt.text(0.49, .5, '$φ_0$', transform=ax.transAxes)

    # Draw an arrow along the r axis
    style = "Simple, tail_width=0.1, head_width=3, head_length=8"
    a1 = patches.FancyArrowPatch((0.0, 0), (0, 1.4*Wrange[1] / const.e / 1e6), color="k", clip_on=False, arrowstyle=style, zorder=10)
    plt.gca().add_patch(a1)

    # And add some r ticks
    ax.set_rticks(np.arange(1, 8))
    ax.set_rorigin(-0.5 * Wrange[1] / const.e / 1e6)
    ax.set_rlabel_position(0)
    ax.tick_params(axis='y', colors='white')

    ax.grid(alpha=0.5, color='k')

    # Finally a color bar on the left side
    cax = fig.add_axes([0.045, 0.05, 0.03, 0.9])
    cb = plt.colorbar(mappable=cm, label='energy gain / MeV', cax=cax)
    cax.yaxis.set_label_position('left')
    
    # tight_layout throws a warning because we use `fig.add_axes`, but is still necessary...
    plt.tight_layout(pad=0.5)
    plt.savefig('build/1_5/bild_gain_nnn.png', dpi=300)