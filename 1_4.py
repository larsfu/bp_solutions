import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const


# This function simulates a cyclotron for the given parameters.
# One „turn“ is equal to a half-turn in real life.
def simulate(turns, E0, f, U, m, q, φ, relativistic):
    # Proton mass
    E0p = m * const.c**2 / const.e  # eV
    B = 2 * np.pi * f * m / q
    E = E0 * np.ones(turns)
    T, r = np.zeros((2, turns))

    for turn in range(turns):
        if not relativistic:
            r[turn] = np.sqrt(2 * m * E[turn - 1] * const.e) / (q * B)
            T[turn] = np.pi * m / (q * B)
        else:
            r[turn] = np.sqrt((E0p + E[turn - 1])**2 - E0p**2) * const.e / (const.c * q * B)
            T[turn] = (E[turn - 1] / E0p + 1) * np.pi * m / (q * B)

        gain = (-1)**(turn + 1) * U * np.cos(2 * np.pi * f * np.sum(T) + φ)
        #      ^^^^^^^^^^^^^^^^
        #      Invert E field for every half turn, as the transition direction changes.

        # Do not continue calculation if the kinetic energy would be below zero.
        # It is debatable what happens in this case – acceleration in the other direction
        # would be a possibility, for that we would need to simulate velocities instead
        # of energies. My guess is that the bunch would just drift a part due to space charge.
        if gain + E[turn - 1] < 0 or E[turn - 1] == np.nan:
            # A numerical operation containing np.nan (not a number) always results in np.nan.
            E[turn] = np.nan
        else:
            E[turn] = E[turn - 1] + gain
    return E, T, r


if __name__ == '__main__':
    turns = 400  # half turns to be precise
    E0 = 50e3  # eV
    f = 20e6  # Hz
    U = 50e3  # V
    m = const.m_p
    q = const.e

    ###########
    # a) + b) #
    ###########

    # Simulate nonrelativistic and relativistic case with phase zero.
    E_nonrel, T_nonrel, r_nonrel = simulate(turns, E0, f, U, m, q, 0, False)
    E_rel, T_rel, r_rel = simulate(turns, E0, f, U, m, q, 0, True)

    fig = plt.figure()
    ax1 = plt.subplot(311)
    ax1.set_ylabel(r'$E_\mathrm{kin}$ / MeV')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.plot(1e-6 * E_nonrel, label='nonrelativistic')
    plt.plot(1e-6 * E_rel, label='relativistic')
    ax1.legend()
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.set_ylabel(r'$T$ / ns')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.plot(1e9 * T_nonrel)
    plt.plot(1e9 * T_rel)
    ax3 = plt.subplot(313, sharex=ax1)
    ax3.set_ylabel(r'$r$ / m')
    ax3.set_xlabel('half turns')
    plt.plot(r_nonrel)
    plt.plot(r_rel)
    fig.align_ylabels((ax1, ax2, ax3))
    plt.tight_layout(pad=1)
    plt.savefig('build/1_4/plot.pdf')
    plt.close()

    ######
    # c) #
    ######

    # Vary the RF phase to achieve the maximal proton energy.
    phis = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 45)

    # Define a ScalarMappable with a colormap to plot every energy curve in a different color
    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=phis[0], vmax=phis[-1]))

    # I am mixing calculation and plotting here, but I consider it fine as the simulated data
    # is not needed in any other place.
    for idx, phi in enumerate(phis):
        E_rel = simulate(500, E0, f, U, m, q, phi, True)[0]
        plt.plot(E_rel / 1e6, color=sm.to_rgba(phi), alpha=0.9)
    # Create color bar with ticks in units of pi/2
    cbar = plt.colorbar(sm, ticks=(-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi))
    cbar.ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.ylabel(r'$E_\mathrm{kin} / $MeV')
    plt.xlabel('Half turns')
    plt.tight_layout(pad=0.5)
    plt.savefig('build/1_4/phases.pdf')
    plt.close()
