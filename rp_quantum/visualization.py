"""
Visualization for Quantum Circuit RP Simulation
================================================
Publication-quality figures comparing quantum circuit vs classical.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette (Nature-style)
C_QUANTUM = '#2166AC'   # Blue
C_CLASSICAL = '#B2182B'  # Red
C_TROTTER = '#4D4D4D'   # Gray
C_ACCENT = '#762A83'    # Purple
C_FILL = '#D1E5F0'      # Light blue


def plot_evolution_comparison(t_values, P_S_quantum, P_S_classical,
                              n_trotter=15, B=0.0, savepath=None):
    """
    Figure 1: Singlet probability evolution — quantum vs classical.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): P_S(t) comparison
    ax = axes[0]
    ax.plot(t_values * 1e6, P_S_classical, '-', color=C_CLASSICAL,
            linewidth=2.5, label='Exact (matrix exponential)')
    ax.plot(t_values * 1e6, P_S_quantum, 'o', color=C_QUANTUM,
            markersize=5, label=f'Quantum circuit ({n_trotter} Trotter steps)')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('$P_S(t) = \\langle S | \\rho(t) | S \\rangle$')
    ax.set_title(f'(a) Singlet Probability Evolution  [B = {B:.1f} mT]')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Panel (b): Error analysis
    ax = axes[1]
    error = np.abs(P_S_quantum - P_S_classical)
    ax.plot(t_values * 1e6, error, '-', color=C_TROTTER, linewidth=2)
    ax.fill_between(t_values * 1e6, error, alpha=0.3, color=C_TROTTER)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('$|P_S^{quantum} - P_S^{classical}|$')
    ax.set_title('(b) Trotterization Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_trotter_convergence(t_values, P_S_classical, trotter_results, savepath=None):
    """
    Figure 2: Trotter step convergence — how many steps needed?
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Classical reference
    ax.plot(t_values * 1e6, P_S_classical, '-', color=C_CLASSICAL,
            linewidth=3, label='Exact', zorder=5)

    # Different Trotter steps
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(trotter_results)))
    for (n_steps, P_S_q), color in zip(trotter_results.items(), colors):
        ax.plot(t_values * 1e6, P_S_q, '--', color=color, linewidth=1.5,
                label=f'{n_steps} steps', alpha=0.8)

    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('$P_S(t)$')
    ax.set_title('Trotterization Convergence: Steps vs Accuracy')
    ax.legend(framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_mfe_comparison(B_values, phi_S_quantum, phi_S_classical, savepath=None):
    """
    Figure 3: Magnetic field effect — quantum circuit vs classical.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Φ_S vs B
    ax = axes[0]
    ax.plot(B_values, phi_S_classical, '-', color=C_CLASSICAL,
            linewidth=2.5, label='Classical (exact)')
    ax.plot(B_values, phi_S_quantum, 's', color=C_QUANTUM,
            markersize=6, label='Quantum circuit')
    ax.set_xlabel('Magnetic Field B (mT)')
    ax.set_ylabel('Singlet Yield $\\Phi_S$')
    ax.set_title('(a) Magnetic Field Effect')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel (b): MFE %
    ax = axes[1]
    mfe_q = (phi_S_quantum - phi_S_quantum[0]) / phi_S_quantum[0] * 100
    mfe_c = (phi_S_classical - phi_S_classical[0]) / phi_S_classical[0] * 100
    ax.plot(B_values, mfe_c, '-', color=C_CLASSICAL, linewidth=2.5)
    ax.plot(B_values, mfe_q, 's', color=C_QUANTUM, markersize=6)
    ax.set_xlabel('Magnetic Field B (mT)')
    ax.set_ylabel('MFE (%)')
    ax.set_title('(b) Relative MFE')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_circuit_diagram(qc, savepath=None, max_depth=40):
    """
    Figure 4: Visualize the quantum circuit (truncated for readability).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    qc.draw('mpl', ax=ax, fold=max_depth)
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)


def plot_publication_composite(t_values, P_S_q, P_S_c, B_values,
                                phi_S_q, phi_S_c, trotter_results, savepath=None):
    """
    Figure 5: Publication-quality 2×2 composite figure.
    Uses natural units (no μs scaling).
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (a) P_S evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_values, P_S_c, '-', color=C_CLASSICAL, linewidth=2.5,
             label='Exact')
    ax1.plot(t_values, P_S_q, 'o', color=C_QUANTUM, markersize=5,
             label=f'Quantum (15 Trotter)')
    ax1.set_xlabel('Time (natural units)')
    ax1.set_ylabel('$P_S(t)$')
    ax1.set_title('(a) Singlet Evolution', fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # (b) Trotter convergence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_values, P_S_c, '-', color=C_CLASSICAL, linewidth=3,
             label='Exact', zorder=5)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(trotter_results)))
    for (n_steps, P_S), color in zip(trotter_results.items(), colors):
        ax2.plot(t_values, P_S, '--', color=color, linewidth=1.5,
                 label=f'{n_steps} steps', alpha=0.8)
    ax2.set_xlabel('Time (natural units)')
    ax2.set_ylabel('$P_S(t)$')
    ax2.set_title('(b) Trotter Convergence', fontweight='bold')
    ax2.legend(framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # (c) MFE
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(B_values, phi_S_c, '-', color=C_CLASSICAL, linewidth=2.5,
             label='Exact')
    ax3.plot(B_values, phi_S_q, 's', color=C_QUANTUM, markersize=6,
             label='Trotter')
    ax3.set_xlabel('$\\omega$ (Zeeman, natural units)')
    ax3.set_ylabel('$\\Phi_S$')
    ax3.set_title('(c) Magnetic Field Effect', fontweight='bold')
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # (d) Error vs Trotter steps
    ax4 = fig.add_subplot(gs[1, 1])
    errors = []
    n_steps_list = sorted(trotter_results.keys())
    for n in n_steps_list:
        err = np.mean(np.abs(trotter_results[n] - P_S_c))
        errors.append(err)
    ax4.semilogy(n_steps_list, errors, 'o-', color=C_ACCENT, linewidth=2,
                 markersize=8)
    ax4.set_xlabel('Number of Trotter Steps')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('(d) Error Convergence', fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        fig.savefig(savepath.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {savepath}")
    plt.close(fig)
