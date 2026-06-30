"""
Radical Pair Simulation — Quantum Circuit Interface
====================================================
Wraps the dimensionless circuit builder for the demo.
"""

import numpy as np
from rp_quantum.circuit_builder import (
    evolve_exact, evolve_trotter, evolve_circuit_shots, singlet_yield
)

__all__ = [
    'simulate_singlet_evolution',
    'simulate_exact',
    'simulate_with_shots',
    'singlet_yield',
]


def simulate_singlet_evolution(t_values, omega=0.1, J=0.01, n_trotter=15,
                               hfc_A=None, hfc_B=None):
    """Trotterized statevector evolution (no shot noise)."""
    return evolve_trotter(t_values, n_trotter=n_trotter, omega=omega, J=J,
                          hfc_A=hfc_A, hfc_B=hfc_B)


def simulate_exact(t_values, omega=0.1, J=0.01, hfc_A=None, hfc_B=None):
    """Exact evolution (matrix exponential)."""
    return evolve_exact(t_values, omega=omega, J=J, hfc_A=hfc_A, hfc_B=hfc_B)


def simulate_with_shots(t_values, omega=0.1, J=0.01, n_trotter=15,
                        hfc_A=None, hfc_B=None, shots=8192, noise_model=None):
    """Quantum circuit on AerSimulator with shot noise."""
    return evolve_circuit_shots(t_values, n_trotter=n_trotter, omega=omega, J=J,
                                 hfc_A=hfc_A, hfc_B=hfc_B, shots=shots,
                                 noise_model=noise_model)
