import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# Create phase space transfer matrix for quadropole or drift
def create_matrix(kind, length, strength):
    t, s, k = kind, length, strength
    # Identitiy
    mat = np.eye(6)
    if t == 0:
        mat[0, 1] = s
        mat[2, 3] = s
    if t == 4:
        Ω = np.sqrt(np.abs(k)) * s
        sqk = np.sqrt(np.abs(k))
        # Indices for focussing and defocussing submatrices
        idx_f = (0, 2) if k < 0 else (2, 4)
        idx_d = (0, 2) if k > 0 else (2, 4)
        mat[idx_f[0] : idx_f[1], idx_f[0] : idx_f[1]] = (
            (np.cos(Ω), 1 / sqk * np.sin(Ω)),
            (-sqk * np.sin(Ω), np.cos(Ω)),
        )
        mat[idx_d[0] : idx_d[1], idx_d[0] : idx_d[1]] = (
            (np.cosh(Ω), 1 / sqk * np.sinh(Ω)),
            (sqk * np.sinh(Ω), np.cosh(Ω)),
        )
    return mat

def track(i_phasespace, kind, length, strength, steps):
    """
    Tracks particles from an initial phase space through a magnet lattice using
    transfer matrices.
    """
    # Allocate space for all phase space results
    phasespace = np.empty((strength.shape[0] * steps + 1, 6, i_phasespace.shape[1]))
    phasespace[0] = i_phasespace
    # Accumulate step length
    total_s = np.zeros(phasespace.shape[0])
    i = 1
    for k, l, s in zip(kind, length, strength):
        mat = create_matrix(k, l/steps, s)
        for j in range(steps):
            # Apply transfer matrix for every step
            phasespace[i] = np.dot(mat, phasespace[i - 1])
            total_s[i] = total_s[i - 1] + l / steps
            i += 1
    return phasespace, total_s

def plot_lattice(ax, kind, length, strength):
    """
    Plots a magnet lattice symbolically.
    """
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    cumulative_l = 0
    max_quad = np.abs(strength[kind == 4]).max()
    max_sext = np.abs(strength[kind == 6]).max() if len(strength[kind == 6]) > 0 else 1
    for t, l, k in zip(kind, length, strength):
        rel_k = 0.5 * k / (max_quad if t == 4 else max_sext)
        if t == 2:
            rect = patches.Rectangle(
                (cumulative_l, 0.1),
                l,
                0.8,
                edgecolor=None,
                facecolor="C2",
                linewidth=1,
                alpha=0.6,
                transform=trans,
            )
            ax.add_patch(rect)
        if t == 4:
            rect = patches.Rectangle(
                (cumulative_l, 0.5),
                l,
                rel_k,
                edgecolor=None,
                facecolor="C1",
                linewidth=1,
                alpha=0.6,
                transform=trans,
            )
            ax.add_patch(rect)
        if t == 6:
            rect = patches.Rectangle(
                (cumulative_l, 0.5 - rel_k / 2),
                l,
                rel_k,
                edgecolor=None,
                facecolor="C3",
                linewidth=1,
                alpha=0.6,
                transform=trans,
            )
            ax.add_patch(rect)
        if t != 1:
            cumulative_l += l
    ax.set_xlim(0, cumulative_l)
    ax.plot((0, cumulative_l), (0, 0), "k", lw=0.75)

if __name__ == "__main__":
    n_particles = 1000
    steps = 10
    angle_sigma = 1e-3
    position_simga = 1e-3
    lattice = "input_files/FODO1.opt"
    
    with open(lattice) as f:
        # Read first line (number of repetitions)
        repetitions = int(f.readline())
        # Give genfromtxt the rest of the file
        o = np.genfromtxt(f, dtype=None, skip_footer=1)
    
    # Repeat lattice as often as necessary
    kind = np.tile(o["f0"], repetitions)
    length = np.tile(o["f1"], repetitions)
    strength = np.tile(o["f2"], repetitions)

    # Check for invalid magnet types
    if not np.in1d(kind, (0, 4)).all():
        raise ValueError("Invalid element type.")

    # Define initial phase space
    i_phasespace = np.array(
        (
            np.random.normal(0, position_simga, n_particles),
            np.random.normal(0, angle_sigma, n_particles),
            np.random.normal(0, position_simga, n_particles),
            np.random.normal(0, angle_sigma, n_particles),
            np.zeros(n_particles),
            np.zeros(n_particles),
        )
    )

    result, s = track(i_phasespace, kind, length, strength, steps)

    # Plotting

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios = [1, 3, 3], figure=fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)
    plot_lattice(ax1, kind, length, strength)
    ax2.plot(s, result[:, 0], 'C0', alpha=0.1, lw=0.5)
    ax3.plot(s, result[:, 2], 'C0', alpha=0.1, lw=0.5)
    ax2.set_ylabel('x / m')
    ax3.set_xlabel('s / m')
    ax3.set_ylabel('y / m')
    ax1.set_yticklabels([])
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.savefig('build/1_10/xy.png', dpi=200)
    plt.close()

    fig = plt.figure(figsize=(kind.shape[0] * 3.2, 4.8))
    gs = gridspec.GridSpec(2, kind.shape[0], figure=fig)
    for i in range(kind.shape[0]):
        ax1 = plt.subplot(gs[0, i])
        ax2 = plt.subplot(gs[1, i])
        ax1.set_xlim(-12, 12)
        ax2.set_xlim(-12, 12)
        ax1.set_ylim(-5, 5)
        ax2.set_ylim(-5, 5)
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('x\' / mrad')
        ax2.set_xlabel('y / mm')
        ax2.set_ylabel('y\' / mrad')
        ax1.set_title(f's = {s[i*steps]:.2f}m')
        ax1.plot(result[i*steps, 0] * 1e3, result[i*steps, 1] * 1e3, 'C0.', alpha=0.5)
        ax2.plot(result[i*steps, 2] * 1e3, result[i*steps, 3] * 1e3, 'C0.', alpha=0.5)
    plt.tight_layout(pad=0.2)
    plt.savefig('build/1_10/scatter.pdf')
