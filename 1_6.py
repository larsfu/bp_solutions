import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from scipy.signal import argrelmin

# This method implements the classic Runge–Kutta method of order 4.
# Its parameters are chosen in this way to be a drop-in replacement for solve_ivp
# fun is the ODE system to integrate, t_span the t variable interval, 
# max_step the (fixed) step size, args is a list of arguments that are passed to fun.
def rk4(fun, t_span, y0, max_step=800e-9, args=None):
    h = max_step
    #Initialize result array with the right size, (steps, dimension)
    y = np.empty((int((t_span[1] - t_span[0])/h), len(y0)))
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
    # This is again to be compatible with solve_ivp. Basically equivalent to `return y.T`.
    return OptimizeResult(y=y.T)

# Our ODE system for the longitudinal phase space of a storage ring
def longitudinal(t, y, E, V0, T0, RF, α, Ψs):
    δ, ΔΨ = y
    dδdt = const.e * V0 / (T0 * E) * (np.sin(Ψs + ΔΨ) - np.sin(Ψs))
    dΔΨdt = np.pi * 2 * RF * α * δ
    return dδdt, dΔΨdt

# Simulate an orbit and find out whether it is stable
@np.vectorize
def is_stable(δ0, ΔΨ0, t_max, max_step, *args, return_orbit=False):
    # Use our own Runge-Kutta implementation to solve the system.
    # You can replace `rk4` by `solve_ivp` to see if we did everything correctly,
    # but it will be a lot slower.
    δ, ΔΨ = rk4(longitudinal, (0, t_max), (δ0, ΔΨ0), max_step=max_step, args=args).y
    
    # Here, an orbit is defined as stable, when at any point in the simulation,
    # the result gets close enough to the initial values. Close enough is defined
    # by trial-and-error as 0.01 in the following norm:
    scale = (min(ΔΨ.max() - ΔΨ.min(), 2 * np.pi), δ.max() - δ.min())
    close_enough = np.sqrt(((ΔΨ - ΔΨ[0]) / scale[0])**2 + ((δ - δ[0]) / scale[1])**2) < 0.01
    
    # We need to skip until the particle has moved away far enough. np.arg{min,max} can
    # be used to find the first {False, True} value in a numpy array. If none are found,
    # it returns zero.
    skip = np.argmin(close_enough)
    revolution = np.argmax(close_enough[skip:])

    # If a close approach ist found (`revolution` nonzero), we can calculate the synchrotron
    # frequency from the number of steps needed to reach the point. If not, we return np.nan.
    synchrotron_frequency = 1/(args[2] * (revolution + skip)) if revolution > 0 else np.nan

    # We do not always need the trajectories.
    if return_orbit:
        return synchrotron_frequency, δ, ΔΨ
    else:
        return synchrotron_frequency

if __name__ == '__main__':
    # BESSY data
    E  = 1.7e9   # GeV
    V0 = 1.2e6   # V
    T0 = 800e-9  # s
    RF = 500e6   # Hz
    W  = 170e3   # eV
    α  = 8e-4
    turns = 500

    Ψs = np.pi - np.arcsin(W/V0)
    E *= const.e
    args = (E, V0, T0, RF, α, Ψs)
    
    # Define the parameter space.
    δ0 = np.linspace(-0.04, 0.04, 1*15)
    ΔΨ0 = np.linspace(-np.pi, np.pi, 1*20)

    A, B = np.meshgrid(δ0, ΔΨ0)
    fsync = is_stable(A, B, turns*T0, T0, *args)
    # Shift the imshow extent by half a unit, so the ticks are centered
    d = ((ΔΨ0[1] - ΔΨ0[0])/2, (δ0[1] - δ0[0])/2)
    extent = (ΔΨ0[0] - d[0], ΔΨ0[-1] + d[0], δ0[0] - d[1], δ0[-1] + d[1])
    plt.imshow(fsync.T/1000, origin='lower', aspect='auto', extent=extent, interpolation='none')
    plt.colorbar(label='synchrotron frequency / kHz')
    plt.xlabel(r'Δ$\Psi$ / rad')
    plt.ylabel(r'$\frac{\mathrm{ΔE}}{E}$')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-0.04, 0.04)
    plt.tight_layout(pad=0)
    plt.savefig('build/1_6/stable.pdf')

    # Functions that are needed only locally can also be defined in a place like this
    def plot_one_trajectory(δ0, ΔΨ0):
        fsync, δ, ΔΨ = is_stable(δ0, ΔΨ0, turns*T0, T0, *args, return_orbit=True)
        plt.plot(ΔΨ, δ, '-', color='C3' if fsync > 0 else 'C0', alpha=0.5)

    for δ0 in np.linspace(0.002, 0.038, 10):
        plot_one_trajectory(δ0, 0)

    # To add some fun, we want to be able to add trajectories on the fly by clicking
    # somewhere in the plot and using the position as initial values.
    def onclick(event):
        # event.{x,y}data are the click coordinates in actual plot units.
        plot_one_trajectory(event.ydata, event.xdata)
        plt.draw()
    # matplotlib allows us to tie our function to an event, here for example clicking.
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    plt.savefig('build/1_6/stable_with_trajectories.pdf')
    plt.show()
