"""
Coherent noise models for Qiskit Aer simulation.

Coherent errors are unitary (systematic) mismatch errors, as opposed to incoherent
depolarizing/dephasing noise. They matter because:
  - Pauli Twirling converts them to incoherent noise, so they test the PT pipeline.
  - GDD and ODR behave differently under coherent vs incoherent noise.
  - IBM hardware has significant coherent components (gate miscalibration, ZZ crosstalk).
"""

from qiskit_aer.noise import NoiseModel, coherent_unitary_error
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Unitary building blocks
# ──────────────────────────────────────────────────────────────────────────────

def _rz_matrix(theta):
    return np.array([[np.exp(-0.5j * theta), 0.0],
                     [0.0,                   np.exp( 0.5j * theta)]])


def _rx_matrix(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c,      -1j * s],
                     [-1j * s, c     ]])


def _zz_matrix(theta):
    """exp(-i θ/2 Z⊗Z)"""
    phases = np.exp(np.array([-0.5j, 0.5j, 0.5j, -0.5j]) * theta)
    return np.diag(phases)


def _ecr_overrotation_matrix(epsilon):
    """
    Perturbative coherent error for an ECR gate:  exp(-i ε/2 ZX) after the ideal ECR.
    ZX = Z⊗X, eigenvalues ±1 → exp(∓iε/2) on each sector.
    """
    ZX = np.kron(np.diag([1., -1.]), np.array([[0., 1.], [1., 0.]]))
    from scipy.linalg import expm
    return expm(-0.5j * epsilon * ZX)


# ──────────────────────────────────────────────────────────────────────────────
#  Public noise model builders
# ──────────────────────────────────────────────────────────────────────────────

def coherent_overrotation_model(
    epsilon_1q: float = 0.01,
    epsilon_2q: float = 0.01,
    gates_1q: list[str] | None = None,
    gates_2q: list[str] | None = None,
) -> NoiseModel:
    """
    Noise model with coherent over-rotation on single- and two-qubit gates.

    Each gate G is followed by a small unitary error U(ε):
      - Single-qubit: RZ(ε)  (pure phase error, most common IBM miscalibration)
      - Two-qubit:    exp(-iε/2 ZX)  (dominant coherent component of ECR/CX)

    Parameters
    ----------
    epsilon_1q : over-rotation angle (rad) for single-qubit gates
    epsilon_2q : over-rotation angle (rad) for two-qubit gates
    gates_1q   : gate names to apply 1q error to (default: IBM basis set)
    gates_2q   : gate names to apply 2q error to (default: ['ecr', 'cx'])

    Returns
    -------
    qiskit_aer.noise.NoiseModel
    """
    if gates_1q is None:
        gates_1q = ['rz', 'sx', 'x', 'id']
    if gates_2q is None:
        gates_2q = ['ecr', 'cx']

    noise_model = NoiseModel()

    if epsilon_1q != 0.0:
        err_1q = coherent_unitary_error(_rz_matrix(epsilon_1q))
        noise_model.add_all_qubit_quantum_error(err_1q, gates_1q)

    if epsilon_2q != 0.0:
        err_2q = coherent_unitary_error(_ecr_overrotation_matrix(epsilon_2q))
        noise_model.add_all_qubit_quantum_error(err_2q, gates_2q)

    return noise_model


def zz_crosstalk_model(
    coupling_map: list[tuple[int, int]],
    zz_angle: float = 0.005,
) -> NoiseModel:
    """
    Residual ZZ-crosstalk noise model for coupled qubit pairs.

    Adds exp(-i ζ/2 Z⊗Z) after every two-qubit gate on each coupled pair.
    This is always-on coherent coupling present in IBM transmon chips.

    Parameters
    ----------
    coupling_map : list of (control, target) qubit pairs
    zz_angle     : ZZ coupling angle ζ (rad) per gate layer

    Returns
    -------
    qiskit_aer.noise.NoiseModel
    """
    noise_model = NoiseModel()
    err = coherent_unitary_error(_zz_matrix(zz_angle))
    for q0, q1 in coupling_map:
        noise_model.add_quantum_error(err, ['cz'], [q0, q1])
    return noise_model


def rx_overrotation_model(
    epsilon_1q: float = 0.01,
    epsilon_2q: float = 0.01,
    gates_1q: list[str] | None = None,
    gates_2q: list[str] | None = None,
) -> NoiseModel:
    """
    Coherent over-rotation noise model with RX-type errors on physical pulse gates.

    On IBM hardware, `rz` is a virtual gate (frame update, no physical pulse) and
    accumulates no coherent error. The dominant 1q coherent error lives on `sx` and
    `x`, which are physical microwave pulses subject to RX over-rotation miscalibration.

    Each gate G is followed by a small unitary error U(ε):
      - Single-qubit (sx, x): RX(ε)  — physical pulse over-rotation
      - Two-qubit   (ecr, cx): exp(-iε/2 ZX)  — ZX over-rotation

    Parameters
    ----------
    epsilon_1q : over-rotation angle (rad) for sx/x gates
    epsilon_2q : over-rotation angle (rad) for two-qubit gates
    gates_1q   : gate names for 1q error (default: ['sx', 'x'])
    gates_2q   : gate names for 2q error (default: ['ecr', 'cx'])

    Returns
    -------
    qiskit_aer.noise.NoiseModel
    """
    if gates_1q is None:
        gates_1q = ['sx', 'x']
    if gates_2q is None:
        gates_2q = ['ecr', 'cx']

    noise_model = NoiseModel()

    if epsilon_1q != 0.0:
        err_1q = coherent_unitary_error(_rx_matrix(epsilon_1q))
        noise_model.add_all_qubit_quantum_error(err_1q, gates_1q)

    if epsilon_2q != 0.0:
        err_2q = coherent_unitary_error(_ecr_overrotation_matrix(epsilon_2q))
        noise_model.add_all_qubit_quantum_error(err_2q, gates_2q)

    return noise_model


def combined_coherent_model(
    epsilon_1q: float = 0.01,
    epsilon_2q: float = 0.01,
    zz_angle: float = 0.005,
    coupling_map: list[tuple[int, int]] | None = None,
    gates_1q: list[str] | None = None,
    gates_ecr: list[str] | None = None,
    gates_zz: list[str] | None = None,
) -> NoiseModel:
    """
    Combined coherent noise: RZ over-rotation (1q) + ZX over-rotation (ECR/CX) + ZZ crosstalk (CZ).

    The ECR/ZX error and the ZZ crosstalk are kept on separate gate types and are
    never composed together — ECR over-rotation belongs to ECR/CX gates, ZZ crosstalk
    belongs to CZ gates.

    Parameters
    ----------
    epsilon_1q   : RZ over-rotation angle (rad) for single-qubit gates
    epsilon_2q   : ZX over-rotation angle (rad) for ECR/CX gates
    zz_angle     : ZZ crosstalk angle ζ (rad) per gate layer on CZ gates
    coupling_map : list of (q0, q1) pairs for ZZ crosstalk; None disables it
    gates_1q     : gate names for 1q error (default: ['rz', 'sx', 'x', 'id'])
    gates_ecr    : gate names for ZX over-rotation (default: ['ecr', 'cx'])
    gates_zz     : gate names for ZZ crosstalk (default: ['cz'])
    """
    if gates_1q is None:
        gates_1q = ['rz', 'sx', 'x', 'id']
    if gates_ecr is None:
        gates_ecr = ['ecr', 'cx']
    if gates_zz is None:
        gates_zz = ['cz']

    noise_model = NoiseModel()

    if epsilon_1q != 0.0:
        err_1q = coherent_unitary_error(_rz_matrix(epsilon_1q))
        noise_model.add_all_qubit_quantum_error(err_1q, gates_1q)

    if epsilon_2q != 0.0:
        err_ecr = coherent_unitary_error(_ecr_overrotation_matrix(epsilon_2q))
        noise_model.add_all_qubit_quantum_error(err_ecr, gates_ecr)

    if zz_angle != 0.0 and coupling_map is not None:
        zz_err = coherent_unitary_error(_zz_matrix(zz_angle))
        for q0, q1 in coupling_map:
            noise_model.add_quantum_error(zz_err, gates_zz, [q0, q1])

    return noise_model
