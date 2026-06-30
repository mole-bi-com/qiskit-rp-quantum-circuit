"""
===============================================================
  RADICAL PAIR SPIN DYNAMICS — DIGITAL QUANTUM SIMULATION (DQS)
  Qiskit 2.4 | Trotterization | AerSimulator
  Reference: arXiv:2406.12986 (APL Quantum, 2024)
===============================================================

  Key finding: ~15 Trotter steps sufficient for faithful RP simulation
  (mean error < 0.02, matching the APL Quantum paper)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from rp_quantum.circuit_builder import (
    build_hamiltonian, evolve_exact, evolve_trotter, evolve_circuit_shots,
    singlet_yield, build_singlet_prep, singlet_state, singlet_proj
)
from rp_quantum.visualization import (
    plot_evolution_comparison, plot_trotter_convergence,
    plot_mfe_comparison, plot_circuit_diagram, plot_publication_composite
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Natural units: HFC_A = 1.0 (reference scale)
HFC_A = [1.0]    # FADH• (anisotropic, strong)
HFC_B = [0.3]    # Trp• (weaker)
K_S = 0.5        # Recombination rate (natural units)
K_T = 0.5


def banner():
    print("=" * 64)
    print("  RADICAL PAIR SPIN DYNAMICS — DIGITAL QUANTUM SIMULATION")
    print("  Qiskit 2.4 | Trotterization | AerSimulator")
    print("  Reference: arXiv:2406.12986 (APL Quantum, 2024)")
    print("=" * 64)


def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  ⏱  {func.__name__}: {elapsed:.1f}s")
        return result
    wrapper.__name__ = func.__name__
    return wrapper


# ═══════════════════════════════════════════════════════════════
@timer
def phase_1_circuit():
    """Phase 1: Build and visualize quantum circuit."""
    print("\n" + "=" * 50)
    print("   PHASE 1: Quantum Circuit Construction")
    print("=" * 50)

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from scipy.linalg import expm
    from rp_quantum.circuit_builder import build_hamiltonian, embed_op, embed_2q, SZ, SP, SM

    n_qubits = 4
    omega, J = 0.1, 0.01

    # Build one Trotter step unitary
    H_Z = omega * (embed_op(SZ, 0, n_qubits) + embed_op(SZ, 1, n_qubits)) / 2
    H_Ex = J * (embed_2q(SP, 0, SM, 1, n_qubits) + embed_2q(SM, 0, SP, 1, n_qubits)) / 2
    H_Ex += J * embed_2q(SZ, 0, SZ, 1, n_qubits) / 4
    H_HFC = build_hamiltonian(n_qubits, 0, 0, HFC_A, HFC_B) - H_Z - H_Ex

    dt = 1.0  # natural unit time
    U_step = expm(-1j * H_HFC * dt) @ expm(-1j * H_Ex * dt) @ expm(-1j * H_Z * dt)

    qc = QuantumCircuit(n_qubits)
    prep = build_singlet_prep(n_qubits)
    qc.compose(prep, inplace=True)
    qc.barrier()
    step_gate = UnitaryGate(U_step, label='U_step')
    qc.append(step_gate, range(n_qubits))
    qc.barrier()
    qc.compose(prep.inverse(), inplace=True)

    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Circuit depth (1 step): {qc.depth()}")
    print(f"  Total gates: {qc.size()}")

    plot_circuit_diagram(qc, savepath=os.path.join(FIG_DIR, 'phase1_circuit.png'))
    return qc


@timer
def phase_2_evolution():
    """Phase 2: Quantum (Trotter) vs Exact evolution comparison."""
    print("\n" + "=" * 50)
    print("   PHASE 2: Trotter vs Exact Evolution")
    print("=" * 50)

    t_values = np.linspace(0, 50, 30)
    omega = 0.1

    print(f"  Time points: {len(t_values)}")
    print(f"  omega = {omega} (Zeeman, natural units)")
    print(f"  J = 0.01 (Exchange)")
    print(f"  HFC_A = {HFC_A}, HFC_B = {HFC_B}")

    P_exact = evolve_exact(t_values, omega=omega, J=0.01, hfc_A=HFC_A, hfc_B=HFC_B)
    P_trotter = evolve_trotter(t_values, n_trotter=15, omega=omega, J=0.01,
                                hfc_A=HFC_A, hfc_B=HFC_B)

    mean_err = np.mean(np.abs(P_exact - P_trotter))
    max_err = np.max(np.abs(P_exact - P_trotter))
    print(f"  Mean |error| (15 Trotter): {mean_err:.6f}")
    print(f"  Max  |error|: {max_err:.6f}")
    print(f"  P_exact[0] = {P_exact[0]:.4f} (should be 1.0)")

    plot_evolution_comparison(
        t_values, P_trotter, P_exact, n_trotter=15, B=omega,
        savepath=os.path.join(FIG_DIR, 'phase2_evolution.png')
    )

    return {'t': t_values, 'P_S_q': P_trotter, 'P_S_c': P_exact}


@timer
def phase_3_trotter_convergence():
    """Phase 3: Trotter step convergence — how many steps needed?"""
    print("\n" + "=" * 50)
    print("   PHASE 3: Trotterization Convergence")
    print("=" * 50)

    t_values = np.linspace(0, 50, 30)
    omega = 0.1
    trotter_steps = [1, 3, 5, 10, 15, 25, 40]

    P_exact = evolve_exact(t_values, omega=omega, J=0.01, hfc_A=HFC_A, hfc_B=HFC_B)

    results = {}
    for n in trotter_steps:
        P_q = evolve_trotter(t_values, n_trotter=n, omega=omega, J=0.01,
                              hfc_A=HFC_A, hfc_B=HFC_B)
        err = np.mean(np.abs(P_q - P_exact))
        print(f"    Trotter n={n:3d}: mean error = {err:.6f}")
        results[n] = P_q

    plot_trotter_convergence(
        t_values, P_exact, results,
        savepath=os.path.join(FIG_DIR, 'phase3_trotter_convergence.png')
    )

    return {'t': t_values, 'P_S_c': P_exact, 'results': results}


@timer
def phase_4_mfe():
    """Phase 4: Magnetic field effect via quantum circuit."""
    print("\n" + "=" * 50)
    print("   PHASE 4: MFE — Singlet Yield vs Field")
    print("=" * 50)

    omega_values = np.linspace(0, 0.5, 15)
    t_values = np.linspace(0, 50, 30)

    phi_exact = np.zeros(len(omega_values))
    phi_trotter = np.zeros(len(omega_values))

    for i, omega in enumerate(omega_values):
        P_e = evolve_exact(t_values, omega=omega, J=0.01, hfc_A=HFC_A, hfc_B=HFC_B)
        P_t = evolve_trotter(t_values, n_trotter=15, omega=omega, J=0.01,
                              hfc_A=HFC_A, hfc_B=HFC_B)
        phi_exact[i] = singlet_yield(P_e, t_values, k_S=K_S, k_T=K_T)
        phi_trotter[i] = singlet_yield(P_t, t_values, k_S=K_S, k_T=K_T)
        print(f"    omega={omega:.3f}: Φ_S(exact)={phi_exact[i]:.4f}  Φ_S(trotter)={phi_trotter[i]:.4f}")

    plot_mfe_comparison(
        omega_values, phi_trotter, phi_exact,
        savepath=os.path.join(FIG_DIR, 'phase4_mfe_quantum_vs_classical.png')
    )

    return {'B': omega_values, 'phi_q': phi_trotter, 'phi_c': phi_exact}


@timer
def phase_5_noise():
    """Phase 5: NISQ noise model — realistic quantum device simulation."""
    print("\n" + "=" * 50)
    print("   PHASE 5: NISQ Noise Effect")
    print("=" * 50)

    from qiskit_aer.noise import NoiseModel, depolarizing_error

    t_values = np.linspace(0, 30, 12)

    # Noise levels to test
    noise_configs = [
        ('Ideal (no noise)', None),
        ('Low noise (1Q=0.001, 2Q=0.01)', {'p1': 0.001, 'p2': 0.01}),
        ('High noise (1Q=0.01, 2Q=0.05)', {'p1': 0.01, 'p2': 0.05}),
    ]

    results = {}
    for label, noise_cfg in noise_configs:
        print(f"    {label}...", end='', flush=True)
        noise_model = None
        if noise_cfg:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(noise_cfg['p1'], 1),
                ['x', 'h', 'z', 'rx', 'ry', 'rz', 'sx']
            )
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(noise_cfg['p2'], 2), ['cx']
            )

        P_S = evolve_circuit_shots(
            t_values, n_trotter=15, omega=0.1, J=0.01,
            hfc_A=HFC_A, hfc_B=HFC_B, shots=4096, noise_model=noise_model
        )
        results[label] = P_S
        print(f" P_S[-1]={P_S[-1]:.4f}")

    # Plot noise comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    P_exact = evolve_exact(t_values, omega=0.1, J=0.01, hfc_A=HFC_A, hfc_B=HFC_B)
    ax.plot(t_values, P_exact, '-', color='#B2182B', linewidth=2.5, label='Exact')
    colors = ['#2166AC', '#4D4D4D', '#D6604D']
    for (label, _), color in zip(noise_configs, colors):
        ax.plot(t_values, results[label], 'o--', color=color, markersize=5,
                label=label, alpha=0.8)
    ax.set_xlabel('Time (natural units)')
    ax.set_ylabel('$P_S(t)$')
    ax.set_title('NISQ Noise Effect on RP Quantum Simulation (15 Trotter steps)')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, 'phase5_noise_effect.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'phase5_noise_effect.pdf'), bbox_inches='tight')
    print(f"  Saved: phase5_noise_effect.png/.pdf")
    plt.close(fig)


@timer
def phase_6_publication(phase2, phase3, phase4):
    """Phase 6: Publication-quality 2×2 composite figure."""
    print("\n" + "=" * 50)
    print("   PHASE 6: Publication Composite Figure")
    print("=" * 50)

    plot_publication_composite(
        phase2['t'], phase2['P_S_q'], phase2['P_S_c'],
        phase4['B'], phase4['phi_q'], phase4['phi_c'],
        phase3['results'],
        savepath=os.path.join(FIG_DIR, 'phase6_publication.png')
    )
    print(f"  Published to: {FIG_DIR}/phase6_publication.png/.pdf")


# ═══════════════保管══════════════════════════════════════════════
if __name__ == '__main__':
    banner()

    t_start = time.time()

    qc = phase_1_circuit()
    p2 = phase_2_evolution()
    p3 = phase_3_trotter_convergence()
    p4 = phase_4_mfe()
    phase_5_noise()
    phase_6_publication(p2, p3, p4)

    total = time.time() - t_start
    print("\n" + "=" * 64)
    print(f"  ✅ ALL PHASES COMPLETE — Total time: {total:.0f}s ({total/60:.1f} min)")
    print("=" * 64)

    print("\n  Figures saved to: figures/")
    for f in sorted(os.listdir(FIG_DIR)):
        path = os.path.join(FIG_DIR, f)
        size = os.path.getsize(path) / 1024
        print(f"    {f:<45s} {size:.1f} KB")
