import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('GTK3Agg')

# Function to generate the wire pattern for a given number of poles.
# The parameter spread defines how far the wires are spread out,
# between 0 (^= 0°) and 1 (^= pi/poles angular spread)
def generate_wires(spread, radius, wirecount, poles):
    # Make sure that spread is a numpy array.
    spread = np.array(spread)

    # Central angle for each pole (e.g. 0 and pi for a dipole)
    pole_angles = np.linspace(0, 2 * np.pi, poles + 1)[:-1]

    # Relative angles for every pole
    wire_angles = np.linspace(-spread * np.pi / poles, spread * np.pi / poles, wirecount)

    # Apply relative angles to every pole by broadcasting
    phi = pole_angles[:,None,None] + wire_angles

    # We do not need a separate axis for each pole, thus we collapse this dimension.
    # You can pass -1 to reshape, meaning “the rest”
    phi = phi.reshape((-1, phi.shape[-1]))

    # Convert to cartesian
    wires = np.stack((radius*np.cos(phi), radius*np.sin(phi)))
    return wires

# Calculate magnetic field for all parameters at once
# wires has 3 axes: (position, wire, spread)
# grid  has 3 axes: (position, x index, y index)
def magnetic_field(wires, currents, grid):
    # To calculate distances and angles for all combinations, we offset 
    # all grids by all wire positions.
    # 
    # As they only have the first axis in common, we need to create new 
    # axes on them so they can be “broadcast” together, spanning a larger space.
    # If you use None (or np.newaxis) like below, an axis of size 1 is created.
    # Now for each array, each axis is either of the same size (for the position),
    # one of the axis sizes is 1 – in this case broadcasting works.
    offset_grids = grid[:,:,:,None,None] - wires[:,None,None,:,:]
    # offset_grids now has 5 axes: (position, x index, y index, wire, spread)

    # Now calculating the distance is trivial, as we just need to do the norm over
    # the first axis (position). This reduces our array to 4 axes.
    distance = np.linalg.norm(offset_grids, axis=0)
    # Also we need the angle between each position and each wire. For this we use
    # the first and zeroth element of offset_grids, i.e. all x and y components.
    # arctan2 helps us to find the right branch of the tangens.
    # https://de.wikipedia.org/wiki/Arctan2
    theta = np.arctan2(offset_grids[1], offset_grids[0])

    # To calculate the B field of each wire, again need to broadcast the currents,
    # (as their array has too few axes) and multiply it by the unit vector in phi direction.
    B = const.mu_0 * currents[:,None] / (2 * np.pi * distance) * np.array((-np.sin(theta), np.cos(theta)))
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                        magnitude                                           direction
    
    # Our approximation of infinitely small wires becomes a problem when we are too close to a wire.
    B[:,distance < 4e-3] = np.nan
    
    # Finally we sum up all B fields along axis 3 (the wire axis), to get the total field.
    return np.sum(B, axis=3)


def homogeneity(B, offset):
    # We need trim off everything outside our region of interest
    B_roi = B[:, offset:-offset, offset:-offset]
    # Calculate direction and magnitude of the B field at every point
    B_theta = np.arctan2(B_roi[1], B_roi[0])
    B_mag = np.hypot(*B_roi)
    # For the dipole case, homogeneity can be defined as a low standard
    # deviaton of magnitude and direction inside the region of interest.
    # This does not work for more than two poles, as the expecte field
    # is not flat. Feel free to submit a pull request for more multipoles!
    theta_hom = np.std(B_theta, axis=(0, 1))
    mag_hom  = np.std(B_mag, axis=(0, 1))
    return theta_hom, mag_hom

if __name__ == '__main__':
    resolution = 51 # number of pixels
    limits = (-0.05, 0.05) # m
    radius = 0.04 # m
    poles = 2 # only even numbers are valid
    wirecount = int(40 / poles) # per pole
    currents = np.array(wirecount * [-24000] + wirecount * [24000]) # A
    # Repeat the current pattern as often as we need it.
    currents = np.tile(currents, poles // 2)

    sp = np.linspace(limits[0], limits[1], resolution)
    grid = np.array(np.meshgrid(sp, sp))
    roi = 0.5 * radius #region of interest for homogeneity

    # See generate_wires for the definition of spread.
    spread = np.linspace(0, 1, 50)
    wires = generate_wires(spread, radius, wirecount, poles)
    B = magnetic_field(wires, currents, grid)
    
    # Number of pixels we want to cut off for the homogeneity calculation.
    offset = (resolution - int(resolution * (2 * roi / (limits[1] - limits[0]))))//2
    hom = homogeneity(B, offset)
    
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(spread, hom[0], 'C0-', label='angular homogeneity')
    ax2 = ax1.twinx() # two y axes (paradoxically called twinx...)
    ln2 = ax2.plot(spread, hom[1], 'C1-', label='magnitudinal homogeneity')
    ax1.set_xlabel(r'relative spread')
    ax1.set_ylabel('standard deviation of angle')
    ax2.set_ylabel('standard deviation of magnitude')
    # We need to to some legend gymnastics to get both in one.
    labels = [l.get_label() for l in ln1 + ln2]
    ax2.legend(ln1 + ln2, labels, loc='best')
    plt.tight_layout(pad=0)
    plt.savefig('build/1_8/homogeneity.png', dpi=300)
    plt.close()

    # We want to know the spread with best homogeneity, but we have two
    # ways of measuring it (magnitude and angle), so we just average those.
    # np.argmin gives you the index of the lowest array entry.
    best_spread = int(np.mean(np.argmin(hom, axis=1)))

    # Once more we need the magnitude of the field for the color.
    # *B does the same as putting B[0], B[1], it unpacks the first axis of B.
    Bt = np.hypot(*B)

    # constrained_layout is in some way a replacement for tight_layout that works
    # better in conjunction with gridspec.
    fig = plt.figure(constrained_layout=True)
    # We need space for our Slider, so we make some.
    gs = gridspec.GridSpec(2, 1, height_ratios = [30, 1], figure=fig)
    ax1 = plt.subplot(gs[0])

    # Shift the imshow extent by half a unit, so the ticks are centered
    d = ((limits[1] - limits[0]) / resolution / 2)
    extent = 2 * [limits[0] - d, limits[1] + d]
    im = ax1.imshow(Bt[:,:,best_spread], origin="lower", extent=extent)

    # It does not make sense to show an arrow for every pixel we calculated,
    # therefore we define a stride to skip some.
    stride = int(resolution / 20)
    # a[::2] takes every second element of a, for example.
    q = ax1.quiver(*grid[:,::stride,::stride], *B[:,::stride,::stride,best_spread], color='white')
    # Make more room for the colorbar – we could also do this in the gridspec.
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # As we want to fix the colorbar, the colormap is not able to display 
    # every field value that might come up at different spreads chosen
    # by the slider. We indicate this throught the extend="both" argument.
    # Another variant would be to choose the colormap maximum at the global
    # maximum field value, but that would reduce the contrast at the best spread.
    cb = fig.colorbar(im, cax=cax, format="%4.1f", extend="both")
    cb.set_label('$B$/T')
    
    # Plot the wires and the roi for reference.
    wire_plot = ax1.scatter(*wires[:,:,best_spread], c=currents, cmap='coolwarm', s=5)
    ax1.plot((-roi, roi, roi, -roi, -roi), (-roi, -roi, roi, roi, -roi), 'C3-', alpha=0.5)
    
    ax1.set_xlim(*limits)
    ax1.set_ylim(*limits)
    ax1.set_xlabel('$x$/m')
    ax1.set_ylabel('$y$/m')

    # A function that is called when the slider is moved.
    def update(val):
        # Find the spread index from given spread value.
        idx = int(spread.shape[0] * val / spread[-1])
        # Update imshow, quiver and wire plot.
        im.set_array(Bt[:,:,idx])
        q.set_UVC(*B[:,::stride,::stride,idx])
        wire_plot.set_offsets(wires[:,:,idx].T)

    # Add the slider and connect the update function.
    slax0 = plt.subplot(gs[1])
    s0 = Slider(slax0, r'spread', 0, np.nextafter(1, 0), valfmt='%.3f', valinit=spread[best_spread])
    s0.valtext.set_fontfamily('monospace')
    s0.on_changed(update)
    plt.show()