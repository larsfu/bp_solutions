import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as const
from mpl_toolkits.mplot3d import Axes3D

######
# a) #
######
# Load electron mass from scipy constants, then divide by 1000 to convert to GeV/c^2
m_e = const.value('electron mass energy equivalent in MeV') * 1e-3

# Create 150 equidistant points from 511keV to 6GeV
E = np.linspace(m_e, 6, 150)
# Meshgrid takes the 1D energy values and creates two 2D arrays that allow getting
# all combinations of the 1D arrays when used in element-wise operations:
#                                    ((a, b, c),      ((d, d, d)
# (a, b, c) and (d, e, f)   become    (a, b, c),  and  (e, e, e)
#                                     (a, b, c))       (f, f, f))
E1, E2 = np.meshgrid(E, E)
#                                                       ((ad, bd, cd)
# Now, when you do E1 * E2 for example, it results in    (ae, be, ce)
#                                                        (af, bf, cf))
# This allows sampling a large parameter space without making the formula any more
# complex than for one value each. In a head-on collision, the invariant mass formula is:
M = np.sqrt(2 * (m_e**2 + E1 * E2 + np.sqrt(E1**2 - m_e**2) * np.sqrt(E2**2 - m_e**2)))

######
# b) #
######
plt.title(r'Invariant mass of head-on electron collision with $E_2\,=\,0$')
# Plot in 1D for E_2 = 0 (takes just the first row of the grid)
plt.plot(E, M[0])
# You can use some LaTeX commands in labels if you enclose them in $…$
# The string prefix `r` means these are “raw strings”, normally python would interpret
# everything starting with a backslash as a control character, now it does not.
plt.xlabel(r'$E_1$ / GeV')
plt.ylabel(r'$\sqrt{s}$ / GeV/c$^2$')
# Set plot limits
plt.xlim(E[0], E[-1])
plt.ylim(M[0][0], M[0][-1])
# `plt.tight_layout` reorders the graphical elements optimally for smaller margins.
plt.tight_layout(pad=0.5)
# When you export an image, the `dpi` (dots per inch) parameter sets the resolution.
# Most of the time, it is better to save plots as PDF files (vector) and not raster images.
plt.savefig('build/1_2/1d.png', dpi=150)
# Close the plot so nothing of this shows up in the next ones.
plt.close()


######
# c) #
######
# Plot in 3D
fig = mpl.figure.Figure()
fig.suptitle('Invariant mass of head-on electron collision')
ax = Axes3D(fig)
# The stride parameters allow you to skip values if plotting is too slow.
# `cmap` chooses a color map for the z value. Good color maps are viridis, plasma and magma.
ax.plot_surface(E1, E2, M, rstride=1, cstride=1, cmap='viridis')
# Default is a left-handed coordinate system.
ax.invert_yaxis()
ax.set_xlabel(r'$E_1$ / GeV')
ax.set_ylabel(r'$E_2$ / GeV')
ax.set_zlabel(r'$\sqrt{s}$ / GeV/c$^2$')
fig.savefig('build/1_2/3d.png', dpi=150)

# Plot in 2D (optional)
plt.title('Invariant mass of head-on electron collision')
# A 2D color plot is nothing else than an image. There fore we use `imshow` to make one.
# Computer images traditionally have their origin on the upper left, we need to change that.
# `extent` tells matplotlib the size of the image in units of energy for the tick labels.
# Note, 2 * [a, b] == [a, b, a, b]. We do not need a quadratic image, therefore `aspect='auto'`.
plt.imshow(M, origin='lower', extent=2 * (E[0], E[-1]), aspect='auto')
plt.xlabel(r'$E_1$ / GeV')
plt.ylabel(r'$E_2$ / GeV')
plt.colorbar(label=r'$\sqrt{s}$ / GeV/c$^2$')
plt.tight_layout(pad=0.5)
plt.savefig('build/1_2/2d.png', dpi=150)
plt.close()
