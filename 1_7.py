import numpy as np
import scipy.constants as const
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
# This “backend” is faster sometimes, e.g. for sliders
matplotlib.use('GTK3Agg')


# Our ODE system for the longitudinal phase space of a storage ring,
# including quadratic and zeroth order momentum compaction contributions
def longitudinal(t, y, E, V0, T0, RF, α0, α1, α2, Ψs):
    δ, ΔΨ = y
    dδdt = const.e * V0 / (T0 * E) * (np.sin(Ψs + ΔΨ) - np.sin(Ψs))
    dΔΨdt = np.pi * 2 * RF * (α0 + α1 * δ + α2 * δ**2)
    return dδdt, dΔΨdt

# To calculate all trajectories on the fly, the code needed to be faster,
# therefore now all initial values are integrated at the same time (vectorized)
def rk4_vectorized(fun, t_span, y0, max_step=800e-9, args=None):
    h = max_step
    #Initialize result array with the right size, (steps, # initial values, dimension)
    y = np.empty((int((t_span[1] - t_span[0])/h), *y0.shape))
    y[0] = y0
    t = t_span[0]

    # This is a trick so we can use f(t, y) later without passing all the arguments
    f = lambda t, y, fun=fun: np.array(fun(t, y, *args))

    # The loop should be obvious if you look at the exercise sheet :)
    for i in range(y.shape[0] - 1):
        t = i * h
        k1 = h * f(t, y[i])
        k2 = h * f(t + h / 2, y[i] + k1 / 2)
        k3 = h * f(t + h / 2, y[i] + k2 / 2)
        k4 = h * f(t + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y

# We need to adjust some indices to the new result dimensions, see lines 46 and 51-62
def is_stable_vectorized(δ0, ΔΨ0, t_max, max_step, *args):
    # Use our own Runge-Kutta implementation to solve the system.
    y0 = np.array((δ0, ΔΨ0))
    result = rk4_vectorized(longitudinal, (0, t_max), y0, max_step=max_step, args=args)
    δ, ΔΨ = result[:,0].T, result[:,1].T
    
    # Here, an orbit is defined as stable, when at any point in the simulation,
    # the result gets close enoxugh to the initial values. Close enough is defined
    # by trial-and-error as 0.022 in the following norm:
    scale = np.array((np.minimum(np.max(ΔΨ, axis=1) - np.min(ΔΨ, axis=1), 2 * np.pi), 
                                 np.max( δ, axis=1) - np.min( δ, axis=1)))

    close_enough = np.sqrt(((ΔΨ - ΔΨ[:, 0, None]) / scale[0][:, None])**2 +
                           (( δ -  δ[:, 0, None]) / scale[1][:, None])**2) < 0.022
    
    # We need to skip until the particle has moved away far enough. np.arg{min,max} can
    # be used to find the first {False, True} value in a numpy array. If none are found,
    # it returns zero.
    skip = np.argmin(close_enough, axis=1)
    mask = np.arange(close_enough.shape[1]) <= skip[:, None]
    close_enough[mask] = False
    revolution = np.argmax(close_enough, axis=1)

    # If a close approach is found (`revolution` nonzero), we can calculate the synchrotron
    # frequency from the number of steps needed to reach the point. If not, we return np.nan.
    synchrotron_frequency = np.empty_like(revolution, dtype=float)
    synchrotron_frequency[revolution >  0] = 1 / (args[2] * revolution[revolution > 0])
    synchrotron_frequency[revolution == 0] = np.nan

    return (synchrotron_frequency, δ, ΔΨ)

if __name__ == '__main__':
    # BESSY data
    E  = 1.7e9   # GeV
    V0 = 1.2e6   # V
    T0 = 800e-9  # s
    RF = 500e6   # Hz
    W  = 170e3   # eV
    α1  = 8e-4
    α0  = 0
    α2  = 0

    turns = 400

    Ψs = np.pi - np.arcsin(W/V0)
    E *= const.e
    args = (E, V0, T0, RF, α0, α1, α2, Ψs)
    
    # Define the parameter space.
    # If the program is too slow for you, decrease the number of samples.
    number_of_samples = 1000
    np.random.seed(0)
    δ0  = np.random.uniform(low=-0.08, high=0.04, size=(number_of_samples,))
    ΔΨ0 = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=(number_of_samples,))

    fig = plt.figure(constrained_layout=True)
    # We need some space below the plot, we use GridSpec to make it
    # Look at https://matplotlib.org/3.1.1/tutorials/intermediate/gridspec.html to
    # to see what you can do with it.
    gs = gridspec.GridSpec(4, 1, height_ratios = [20, 1, 1, 1], figure=fig)
    ax1 = plt.subplot(gs[0])
    ax1.set_xlim(-2*np.pi, 2*np.pi)
    ax1.set_ylim(-0.08, 0.04)
    ax1.set_xlabel(r'Δ$\Psi$ / rad')
    ax1.set_ylabel(r'$\frac{\mathrm{ΔE}}{E}$')

    # First calculation of all trajectories at once
    result = is_stable_vectorized(δ0, ΔΨ0, turns*T0, T0, *args)
    f = result[0]
    stable = ~np.isnan(f) # ~ inverts the mask
    # For plotting we also need to improve speed, therefore we use two line collections
    # LineCollection needs a list of x-y tuples, so we need to restructure our array a bit
    segments = np.rollaxis(np.array(result[1:3][::-1]), 0, 3)
    lc_stable = LineCollection(segments[stable], alpha=1, array=f[stable]/1e3, cmap='plasma')
    lc_unstable = LineCollection(segments[~stable], alpha=0.1, color='C0')
    ax1.add_collection(lc_stable)
    ax1.add_collection(lc_unstable) 
    cb = plt.colorbar(lc_stable, label='synchrotron frequency / kHz', format="%2i")

    # We define a function that is called when a slider is moved
    def update(val):
        # Change arguments and calculate new trajectories
        args = (E, V0, T0, RF, s0.val, s1.val, s2.val, Ψs)
        result = is_stable_vectorized(δ0, ΔΨ0, turns*T0, T0, *args)
        f = result[0]
        stable = ~np.isnan(f)
        segments = np.rollaxis(np.array(result[1:3][::-1]), 0, 3)

        # Update the LineCollection with the new data
        lc_stable.set_segments(segments[stable])
        # Check if there are any stable trajectories - if we dont check it breaks
        if sum(stable) > 0:
            lc_stable.set_array(f[stable]/1e3)
            lc_stable.autoscale()
        lc_unstable.set_segments(segments[~stable])
    
    # Create three slider axes to modify α0-α2 on the fly
    slax0, slax1, slax2 = plt.subplot(gs[1]), plt.subplot(gs[2]), plt.subplot(gs[3])
    s0 = Slider(slax0, r'$\alpha_0$', 0,           3.8e-5,    valfmt='%.1e', valinit=α0)
    #           ^^^^^  ^^^^^^^^^^^^^  ^^           ^^^^^^^    ^^^^^^^^^^^^^  ^^^^^^^^^^
    #           axis       label      start value  end value   label format    initial
    s1 = Slider(slax1, r'$\alpha_1$', 0, 1e-2, valfmt='%.1e', valinit=α1)
    s2 = Slider(slax2, r'$\alpha_2$', 0, 2e-2, valfmt='%.1e', valinit=α2)
    for s in (s0, s1, s2):
        s.valtext.set_fontfamily('monospace')
        s.on_changed(update)
    
    plt.show()
