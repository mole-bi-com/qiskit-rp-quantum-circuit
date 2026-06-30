"""
Radical Pair Spin Dynamics on Digital Quantum Computer
=======================================================
Uses natural (dimensionless) units where HFC = 1.
This is the standard approach for Trotterized quantum simulation.

Hamiltonian (dimensionless):
  H = ω(S_z^A + S_z^B) + J(S_A·S_B) + Σ_k A_k(S_e·I_k)

All parameters are ratios relative to the reference HFC strength.
Physical values recovered through dimensional analysis.

Reference: arXiv:2406.12986 (APL Quantum, 2024)
"""

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


# ═══════════════════════════════════════════════════════════════
#  Pauli matrices
# ═══════════════════════════════════════════════════════════════
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SP = np.array([[0, 1], [0, 0]], dtype=complex)
SM = np.array([[0, 0], [1, 0]], dtype=complex)


def tensor(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def embed_op(op, target, total):
    ops = [I2] * total
    ops[target] = op
    return tensor(*ops)


def embed_2q(op_A, t_A, op_B, t_B, total):
    ops = [I2] * total
    ops[t_A] = op_A
    ops[t_B] = op_B
    return tensor(*ops)


# ═══════════════════════════════════════════════════════════════
#  Hamiltonian (dimensionless units)
# ═══════════════════════════════════════════════════════════════

def build_hamiltonian(n_qubits, omega, J, hfc_A=None, hfc_B=None):
    """
    H = ω(S_z^A + S_z^B) + J·S_A·S_B + Σ A_k(S_e·I_k)

    All parameters dimensionless (HFC = 1 scale).
    S = σ/2 (spin-1/2 operators)

    Parameters
    ----------
    omega : float — Zeeman frequency (ratio to HFC reference)
    J : float — Exchange coupling
    hfc_A, hfc_B : list — HFC strengths on each radical
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # Zeeman: ω(S_z^A + S_z^B)
    H += omega * embed_op(SZ, 0, n_qubits) / 2
    H += omega * embed_op(SZ, 1, n_qubits) / 2

    # Exchange: J·S_A·S_B = J/2(S+S- + S-S+) + J·S_z·S_z
    H += J * embed_2q(SP, 0, SM, 1, n_qubits) / 2
    H += J * embed_2q(SM, 0, SP, 1, n_qubits) / 2
    H += J * embed_2q(SZ, 0, SZ, 1, n_qubits) / 4

    # Hyperfine on radical A (electron 0)
    nA = len(hfc_A) if hfc_A else 0
    nB = len(hfc_B) if hfc_B else 0
    if hfc_A:
        for k, a in enumerate(hfc_A):
            qn = 2 + k
            H += a * embed_2q(SP, 0, SM, qn, n_qubits) / 2
            H += a * embed_2q(SM, 0, SP, qn, n_qubits) / 2
            H += a * embed_2q(SZ, 0, SZ, qn, n_qubits) / 4

    # Hyperfine on radical B (electron 1)
    if hfc_B:
        for k, a in enumerate(hfc_B):
            qn = 2 + nA + k
            H += a * embed_2q(SP, 1, SM, qn, n_qubits) / 2
            H += a * embed_2q(SM, 1, SP, qn, n_qubits) / 2
            H += a * embed_2q(SZ, 1, SZ, qn, n_qubits) / 4

    return H


# ═══════════════════════════════════════════════════════════════
#  State preparation
# ═══════════════════════════════════════════════════════════════

def singlet_state(n_qubits):
    """|S⟩ = (|01⟩ - |10⟩)/√2 ⊗ |00...0⟩"""
    v = np.zeros(2**n_qubits, dtype=complex)
    v[2**(n_qubits-2)] = 1/np.sqrt(2)    # |01⟩ on electrons
    v[2**(n_qubits-1)] = -1/np.sqrt(2)   # |10⟩ on electrons
    return v


def singlet_proj(n_qubits):
    v = singlet_state(n_qubits)
    return np.outer(v, v.conj())


def build_singlet_prep(n_qubits):
    """Circuit: |00...0⟩ → |S⟩"""
    qc = QuantumCircuit(n_qubits)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(0)
    return qc


# ═══════════════════════════════════════════════════════════════
#  Evolution (exact + Trotterized)
# ═══════════════════════════════════════════════════════════════

def evolve_exact(t_values, omega=0.1, J=0.01, hfc_A=None, hfc_B=None):
    """Exact statevector evolution. Returns P_S(t)."""
    nA = len(hfc_A) if hfc_A else 0
    nB = len(hfc_B) if hfc_B else 0
    n_qubits = 2 + nA + nB

    H = build_hamiltonian(n_qubits, omega, J, hfc_A, hfc_B)
    psi0 = singlet_state(n_qubits)
    P_S = singlet_proj(n_qubits)

    result = np.zeros(len(t_values))
    for i, t in enumerate(t_values):
        U = expm(-1j * H * t)
        psi_t = U @ psi0
        result[i] = np.real(np.conj(psi_t) @ P_S @ psi_t)
    return result


def evolve_trotter(t_values, n_trotter=15, omega=0.1, J=0.01,
                    hfc_A=None, hfc_B=None):
    """
    Trotterized evolution: U ≈ [exp(-iH_Z·dt)·exp(-iH_Ex·dt)·exp(-iH_HFC·dt)]^n

    Returns P_S(t) with Trotter error only (no shot noise).
    """
    nA = len(hfc_A) if hfc_A else 0
    nB = len(hfc_B) if hfc_B else 0
    n_qubits = 2 + nA + nB

    # Build component Hamiltonians
    dim = 2**n_qubits
    H_Z = np.zeros((dim, dim), dtype=complex)
    H_Z += omega * embed_op(SZ, 0, n_qubits) / 2
    H_Z += omega * embed_op(SZ, 1, n_qubits) / 2

    H_Ex = np.zeros((dim, dim), dtype=complex)
    H_Ex += J * embed_2q(SP, 0, SM, 1, n_qubits) / 2
    H_Ex += J * embed_2q(SM, 0, SP, 1, n_qubits) / 2
    H_Ex += J * embed_2q(SZ, 0, SZ, 1, n_qubits) / 4

    H_HFC = np.zeros((dim, dim), dtype=complex)
    if hfc_A:
        for k, a in enumerate(hfc_A):
            qn = 2 + k
            H_HFC += a * embed_2q(SP, 0, SM, qn, n_qubits) / 2
            H_HFC += a * embed_2q(SM, 0, SP, qn, n_qubits) / 2
            H_HFC += a * embed_2q(SZ, 0, SZ, qn, n_qubits) / 4
    if hfc_B:
        for k, a in enumerate(hfc_B):
            qn = 2 + nA + k
            H_HFC += a * embed_2q(SP, 1, SM, qn, n_qubits) / 2
            H_HFC += a * embed_2q(SM, 1, SP, qn, n_qubits) / 2
            H_HFC += a * embed_2q(SZ, 1, SZ, qn, n_qubits) / 4

    psi0 = singlet_state(n_qubits)
    P_S = singlet_proj(n_qubits)

    result = np.zeros(len(t_values))
    for i, t in enumerate(t_values):
        dt = t / n_trotter
        U_Z = expm(-1j * H_Z * dt)
        U_Ex = expm(-1j * H_Ex * dt)
        U_HFC = expm(-1j * H_HFC * dt)

        psi = psi0.copy()
        for _ in range(n_trotter):
            psi = U_HFC @ U_Ex @ U_Z @ psi

        result[i] = np.real(np.conj(psi) @ P_S @ psi)
    return result


def evolve_circuit_shots(t_values, n_trotter=15, omega=0.1, J=0.01,
                          hfc_A=None, hfc_B=None, shots=8192, noise_model=None):
    """
    Run the actual quantum circuit on AerSimulator with shot noise.
    Includes state preparation, Trotterized unitary gates, and measurement.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    nA = len(hfc_A) if hfc_A else 0
    nB = len(hfc_B) if hfc_B else 0
    n_qubits = 2 + nA + nB
    dim = 2**n_qubits

    # Precompute component Hamiltonians and unitaries
    H_Z = omega * (embed_op(SZ, 0, n_qubits) + embed_op(SZ, 1, n_qubits)) / 2

    H_Ex = J * (embed_2q(SP, 0, SM, 1, n_qubits) + embed_2q(SM, 0, SP, 1, n_qubits)) / 2
    H_Ex += J * embed_2q(SZ, 0, SZ, 1, n_qubits) / 4

    H_HFC = np.zeros((dim, dim), dtype=complex)
    if hfc_A:
        for k, a in enumerate(hfc_A):
            qn = 2 + k
            H_HFC += a * (embed_2q(SP, 0, SM, qn, n_qubits) + embed_2q(SM, 0, SP, qn, n_qubits)) / 2
            H_HFC += a * embed_2q(SZ, 0, SZ, qn, n_qubits) / 4
    if hfc_B:
        for k, a in enumerate(hfc_B):
            qn = 2 + nA + k
            H_HFC += a * (embed_2q(SP, 1, SM, qn, n_qubits) + embed_2q(SM, 1, SP, qn, n_qubits)) / 2
            H_HFC += a * embed_2q(SZ, 1, SZ, qn, n_qubits) / 4

    sim = AerSimulator(noise_model=noise_model)
    all_zeros = '0' * n_qubits
    P_S = np.zeros(len(t_values))

    for i, t in enumerate(t_values):
        dt = t / n_trotter
        U_Z = expm(-1j * H_Z * dt)
        U_Ex = expm(-1j * H_Ex * dt)
        U_HFC = expm(-1j * H_HFC * dt)
        U_step = U_HFC @ U_Ex @ U_Z

        # Build circuit
        qc = QuantumCircuit(n_qubits)
        prep = build_singlet_prep(n_qubits)
        qc.compose(prep, inplace=True)
        qc.barrier()
        step_gate = UnitaryGate(U_step, label=f'U(t={dt:.3f})')
        for _ in range(n_trotter):
            qc.append(step_gate, range(n_qubits))
        qc.barrier()
        qc.compose(prep.inverse(), inplace=True)
        qc.measure_all()

        qc_t = transpile(qc, sim, optimization_level=1)
        counts = sim.run(qc_t, shots=shots).result().get_counts()
        P_S[i] = counts.get(all_zeros, 0) / shots

    return P_S


# ═══════════════════════════════════════════════════════════════
#  Yield calculation
# ═══════════════════════════════════════════════════════════════

def singlet_yield(P_S, t_values, k_S=0.5, k_T=0.5):
    """
    Φ_S = k_S ∫₀^∞ P_S(t) · exp(-(k_S+k_T)·t) dt

    Parameters k_S, k_T in natural units.
    """
    k_total = k_S + k_T
    weights = np.exp(-k_total * t_values)
    return k_S * np.trapezoid(P_S * weights, t_values)
