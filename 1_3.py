import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
# from p_tqdm import p_map
# import matplotlib


# This function takes a potential array and resets the electrodes to their initial values
def set_electrodes(potential, o, w):
    potential[o:o+w,    o:o+w] = potential[-o-w:-o, -o-w: -o] =  1000
    potential[o:o+w, -o-w: -o] = potential[-o-w:-o,    o:o+w] = -1000


# This function takes the current potential array and does one iteration step on it
def iterate_manual(old):
    # We need two arrays, old and new values, therefore we copy the old one
    copied = np.copy(old)

    # To calculate the averages, the whole field is shifted in four different ways
    # instead of manually looping through the array in two four loops. This is
    # much faster.

    # Central pixels:
    copied[1:-1, 1:-1] = (old[0:-2, 1:-1] + old[2:, 1:-1] + old[1:-1, 0:-2] + old[1:-1, 2:]) / 4
    #      ^^^^^^^^^^
    #      Everything but the outermost pixels is considered here

    # For the edges and corners, there still need to be special cases.
    # Edges:
    copied[ 0,     1:-1] = (old[ 1,     1:-1] + old[ 0,    0:-2] + old[ 0,  2:]) / 3
    copied[-1,     1:-1] = (old[-2,     1:-1] + old[-1,    0:-2] + old[-1,  2:]) / 3
    copied[ 1:-1,  0   ] = (old[ 1:-1,  1   ] + old[ 0:-2,  0  ] + old[ 2:, 0 ]) / 3
    copied[ 1:-1, -1   ] = (old[ 1:-1, -2   ] + old[ 0:-2, -1  ] + old[ 2:, -1]) / 3
    # Corners:
    copied[0,   0] = (old[ 0,  1] + old[ 1,  0]) / 2
    copied[0,  -1] = (old[ 0, -2] + old[ 1, -1]) / 2
    copied[-1, -1] = (old[-1, -2] + old[-2, -1]) / 2
    copied[-1,  0] = (old[-1,  1] + old[-2,  0]) / 2

    return copied


# This is a drop-in replacement for the function above and does almost the same
def iterate_convolve(potential):
    # An alternative approach is to consider the iteration a convolution with the following
    # convolution kernel:
    weights = np.array(((0,    0.25, 0   ),
                        (0.25, 0,    0.25),
                        (0,    0.25, 0   )))
    # For each pixel in the resulting convolved array, the weights define the contribution of
    # its surrounding pixels to the result. (The pixel itself being the center element (1,1))

    # scipy.ndimage.convolve now does everything for us. Unfortunately it does not support
    # the required boundary condition. The next best mode is 'nearest', see
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
    # fore more details and other variants.
    return convolve(potential, weights, mode='nearest')


# This function does the whole iteration
def do_iteration(init, field_size, electrode_offset, electrode_width, stop):
    # Initialize empty potential array (with initial value)
    potential = np.zeros(field_size) + init

    # This variable will be used to store the maximum difference to the last iteration step.
    # It is initialized with infinity to be larger than any other number in the beginning.
    convergence = np.inf

    # The evolution of the maximum difference may be interesting, therefore we store it.
    c_list = []

    # Iterate with an additional stop criterion to avoid infinite loops
    while convergence > stop and len(c_list) < 60000:
        # Do the iteration (you can put iterate_convolve here to try it)
        new_potential = iterate_manual(potential)
        # Reset the electrode values
        set_electrodes(new_potential, electrode_offset, electrode_width)
        # Calculate absolute maximum difference
        convergence = np.max(np.abs(potential - new_potential))
        # Store value
        c_list.append(convergence)
        potential = new_potential
    return c_list, potential


# This checks whether you executed this Python script as a program (e.g. with `python 1_3.py`)
# and is False when you imported it as a module from another script. It is not that useful
# here, but may be in the future for projects with multiple files.
if __name__ == '__main__':
    field_size = (100, 100)
    electrode_offset = 20  # from boundary
    electrode_width = 8
    # Stop criterion. The iteration is stopped when the maximum absolute difference between two
    # iteration steps is less than this
    ε = 1e-5
    c_list, potential = do_iteration(1, field_size, electrode_offset, electrode_width, ε)

    # Plot the evolution of the convergence
    plt.plot(c_list)
    plt.yscale('log')
    plt.xlim(-300, len(c_list) + 300)
    plt.ylabel('Maximum absolute difference to previous iteration')
    plt.xlabel('Iteration')
    plt.tight_layout(pad=.5)
    plt.savefig('build/1_3/convergence.pdf')
    plt.close()

    # Plot the resulting potential array
    fig, ax = plt.subplots()
    # As we have potential values above and below zero and zero is a distinguished value,
    # a 'divergent' colormap like RdBu or coolwarm should be used.
    plt.imshow(potential, origin='lower', cmap='RdBu')
    plt.title(f'{len(c_list)} iterations')
    plt.colorbar(label='Potential / V')
    # Add some contour lines (optional)
    CS = ax.contour(potential, colors='k', levels=np.arange(-750, 1000, 250))
    ax.clabel(CS, inline=1, fontsize=10, fmt='%1.0f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout(pad=.5)
    plt.savefig('build/1_3/potential.pdf')
    plt.close()

    # Very, very optional:
    # It may be interesting to make a convergence analysis for different initial conditions.
    # The following code does that. If you want to try it, uncomment it and line 4+5.
    # Additionally you have to install the p-tqdm package (pip install --user p-tqdm), if
    # it is not already installed. The package provides the p_map command that allows us to
    # spread out the computation over multiple processes (and makes a nice progress bar!),
    # which gives a large performance boost on modern processors.

    # inits = np.append(np.geomspace(-1, -1e-4, 9), np.geomspace(1e-4, 1, 9))
    # convergences = p_map(lambda init:
    #   do_iteration(init, field_size, electrode_offset, electrode_width, 0)[0], inits)
    #
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=inits.shape[0])
    # s_m = matplotlib.cm.ScalarMappable(cmap='Spectral', norm=norm)
    #
    # for idx, (c, i) in enumerate(zip(convergences, inits)):
    #     plt.plot(c, alpha=1, label=f'{i:.1e}', color=s_m.to_rgba(idx))
    # plt.yscale('log')
    # #plt.xlim(-300, count+300)
    # plt.ylabel('Mean difference to previous iteration')
    # plt.xlabel('Iteration')
    # plt.title('Convergence for different initial values')
    # plt.legend(loc='best', ncol=2, prop={'family': 'monospace'})
    # plt.tight_layout(pad=.5)
    # plt.savefig('build/1_3/conv_all.pdf')
