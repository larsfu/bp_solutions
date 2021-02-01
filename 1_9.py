import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider

# Calculates the magnetix flux density inside any multipole.
# Order is defined by n. n = 1: dipole, n = 2: quadrupole,...
def multipole_field(X, Y, n, C):
    B = C * (X + 1j * Y) ** (n-1)
    # Extract real and imaginary parts
    # (This is just how the formula works)
    B = np.array((np.imag(B), np.real(B)))
    # Field magnitude
    Bt = np.hypot(*B)
    return B, Bt


# Calculate the ideal magnetic yoke shape for any multipole.
def multipole_yoke(ϑ, n, r0):
    # We will encounter numeric problems as the function has poles.
    # I am too lazy to determine where these poles are, therefore
    # I just disable numpy warnings for this function and let
    # it do its thing. Values for poles will be np.nan or np.inf.
    np.seterr(all='ignore')
    # Yoke shape in polar coordinates, r0 is the minimum distance.
    # See https://cds.cern.ch/record/1333874/files/1.pdf (p. 15)
    r = r0 *(1 / np.sin(n * ϑ))**(1/n)
    # Transform to cartesian
    x = r * np.cos(ϑ)
    y = r * np.sin(ϑ)
    return x, y


if __name__ == "__main__":
    # Simulation 
    limits = (-30e-3, 30e-3) # m
    n = 2 # (quadrupole)
    C = 75 # T/m for quadrupole
    r0 = 0.02 # m

    x = np.linspace(*limits, 61)
    X, Y = np.meshgrid(x, x)

    pole_ϑ = np.linspace(-np.pi, np.pi, 2001)
    B, Bt = multipole_field(X, Y, n, C)
    pole_x, pole_y = multipole_yoke(pole_ϑ, n, r0)

    # Plotting

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, height_ratios = [30, 1], width_ratios=[20,1], figure=fig)
    ax1 = plt.subplot(gs[0,0])

    # Magnitude in the background
    im = ax1.imshow(Bt, origin="lower", extent=(*limits,*limits), cmap='viridis')
    # Quiver in the foreground
    stride = int(x.shape[0] / 20)
    q = ax1.quiver(X[::stride,::stride], 
                   Y[::stride,::stride], 
                  *B[:,::stride,::stride], 
                   color='white')
    
    cax = plt.subplot(gs[0,1])
    cb = fig.colorbar(im, cax=cax, label="B / T")

    poles1, poles2, = ax1.plot(pole_x, pole_y, pole_x, -pole_y, color='white')

    ax1.set_xlim(*limits)
    ax1.set_ylim(*limits)

    ax1.set_xlabel('x / m')
    ax1.set_ylabel('y / m')

    # A function that is called when the slider is moved.
    def update(new_n):
        global q, im, poles1, poles2
        C_const = C * (np.sqrt(2) * limits[1])**(1-new_n + (n-1))
        B, Bt = multipole_field(X, Y, new_n, C_const)
        pole_x, pole_y = multipole_yoke(pole_ϑ, new_n, r0)
        im.set_array(Bt)
        im.autoscale()
        # If someone knows a way to update a quiver plot that also rescales the
        # arrow lengths, please send pull request. Could not get set_UVC to work
        q.remove()
        q = ax1.quiver(X[::stride,::stride], 
                       Y[::stride,::stride], 
                      *B[:,::stride,::stride], 
                       color='white')
        poles1.set_data(pole_x, pole_y)
        poles2.set_data(pole_x, -pole_y)

    # Add the slider and connect the update function.
    slax0 = plt.subplot(gs[1,:])
    s0 = Slider(slax0, r'n', 1, 7, valstep=1, valinit=n)
    s0.valtext.set_fontfamily('monospace')
    s0.on_changed(update)
    plt.show()