"""Fidelity of a chain of GHZ blocks from populations and parity oscillations.

For a block of N qubits and the target (|0...0> + e^{i theta} |1...1>) / sqrt(2) (Javadi-Abhari et al., arXiv:2510.09520;
Guhne et al., PRA 76, 030305 (2007); Wei et al., PRA 101, 032343 (2020)):

    F = (P + chi) / 2,    P = <0|rho|0> + <1|rho|1>,    chi = 2 Re(e^{i theta} rho_01),    rho_01 = <0...0|rho|1...1>.

P comes from a Z-basis measurement. rho_01 comes from the parity M_phi = prod_j (cos(phi) X_j + sin(phi) Y_j), measured
with Rz(-phi) Ry(-pi/2) on every qubit before a Z measurement, at phi_k = k pi / (n + 1), k = 0, ..., 2n + 1:

    <M_phi> = 2 Re(e^{i N phi} rho_01) + (frequencies |q| < N),    I_q = 1 / (2n + 2) sum_k e^{i q phi_k} <M_phi_k>,

so that I_{-N} = rho_01 for every block of N <= n qubits. The blocks of a chain are measured at once: the grid of the
largest block also resolves the smaller remainder block. The paper uses chi_abs = |I_N| + |I_{-N}| = 2 |rho_01|, which
ignores the relative phase and gives the fidelity with the closest GHZ state; chi is the coherence with the target.

Readout (TREX, van den Berg et al., PRA 105, 032620 (2022)): with measurement twirling the readout channel only rescales
the Z strings. The parity of a block is divided by its value <Z...Z> on |0...0> (readout calibration circuit). The
population uses the factors f_q = <Z_q> of each qubit (uncorrelated readout errors; the paper uses M3 here): every shot
contributes 2^{-N} [prod_q (1 + z_q / f_q) + prod_q (1 - z_q / f_q)].

Errors are first order in the shot noise of the job circuits and of the calibration circuit, whose factors enter P and
chi of a block together.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit

import hardware.utils.circs as circutils


def chain_blocks(nqubits, block_size):
    """[(start, stop)] of the GHZ blocks of stateprep.GHZBlockChainStatePrepBSize(nqubits, block_size), remainder last."""
    nblocks = nqubits // block_size
    blocks = [(i * block_size, (i + 1) * block_size) for i in range(nblocks)]
    if nqubits % block_size != 0:
        blocks.append((nblocks * block_size, nqubits))
    return blocks


def parity_phases(block_size):
    return np.arange(2 * block_size + 2) * np.pi / (block_size + 1)


def parity_basis_change(nqubits, phi):
    """Rz(-phi) Ry(-pi/2) on every qubit: a Z measurement afterwards measures cos(phi) X + sin(phi) Y."""
    circuit = QuantumCircuit(nqubits)
    circuit.rz(-phi, range(nqubits))
    circuit.ry(-np.pi / 2, range(nqubits))
    return circuit


def fidelity_circuits(state_prep, block_size):
    """[Z-basis circuit, parity circuits at parity_phases(block_size)], all measured, and their labels."""
    nqubits = state_prep.num_qubits
    circuits = [circutils.measure_all(state_prep, add_barrier=False)]
    labels = [{"block_size": int(block_size), "basis": "Z", "phase": None}]
    for phi in parity_phases(block_size):
        circuits.append(circutils.measure_all(state_prep.compose(parity_basis_change(nqubits, phi)), add_barrier=False))
        labels.append({"block_size": int(block_size), "basis": "parity", "phase": float(phi)})
    return circuits, labels


def readout_calibration_circuit(nqubits):
    return circutils.measure_all(QuantumCircuit(nqubits), add_barrier=False)


def counts_to_z(counts, nqubits):
    """Z value (+-1) of every qubit for each bitstring of counts (qubit q is character -1 - q), and the counts."""
    keys = list(counts)
    bits = np.frombuffer("".join(keys).encode(), dtype=np.uint8).reshape(len(keys), -1) - ord("0")
    if bits.shape[1] != nqubits:
        raise ValueError(f"Bitstrings have {bits.shape[1]} bits, expected {nqubits}")
    z = 1 - 2 * bits[:, ::-1].astype(np.int8)
    return z, np.array(list(counts.values()), dtype=float)


def _mean_and_error(values, weights):
    shots = weights.sum()
    mean = values @ weights / shots
    variance = (values - mean) ** 2 @ weights / shots
    return mean, np.sqrt(variance / shots)


def _exclusive_products(x):
    """prod_{r != q} x[:, r] for every column q, without divisions."""
    ones = np.ones((len(x), 1))
    left = np.cumprod(np.hstack([ones, x[:, :-1]]), axis=1)
    right = np.cumprod(np.hstack([ones, x[:, :0:-1]]), axis=1)[:, ::-1]
    return left * right


def block_fidelities(z_counts, parity_counts, block_size, nqubits, cal_counts=None, target_phases=None):
    """Population, coherence and fidelity of every block of chain_blocks(nqubits, block_size).

    z_counts: counts of the Z-basis circuit. parity_counts: counts of the parity circuits, at parity_phases(block_size).
    cal_counts: counts of the readout calibration circuit for TREX, None for the raw estimate.
    target_phases: theta of every block; default N pi / 2, the block GHZ state followed by S on every qubit
        (circs_old.TStabilizerBlockStateBSize).
    Returns a dict of arrays over the blocks: blocks, sizes, P, chi, chi_abs, F, F_abs, rho01, parity (blocks x phases),
    phases, the readout factors f_block, and the errors *_err (parity_err: shot noise of the parity circuits only).
    """
    blocks = chain_blocks(nqubits, block_size)
    nblocks = len(blocks)
    sizes = np.array([b - a for a, b in blocks])
    phases = parity_phases(block_size)
    if len(parity_counts) != len(phases):
        raise ValueError(f"Expected {len(phases)} parity circuits for block size {block_size}, got {len(parity_counts)}")
    theta = sizes * np.pi / 2 if target_phases is None else np.asarray(target_phases, dtype=float)
    if cal_counts is None:
        f_qubit, f_block = np.ones(nqubits), np.ones(nblocks)
    else:
        z_cal, w_cal = counts_to_z(cal_counts, nqubits)
        f_qubit = w_cal @ z_cal / w_cal.sum()
        f_block = np.array([w_cal @ np.prod(z_cal[:, a:b], axis=1) / w_cal.sum() for a, b in blocks])

    # Population and its derivatives with respect to the readout factors of the block qubits
    P, P_err, dP = np.zeros(nblocks), np.zeros(nblocks), []
    z, w = counts_to_z(z_counts, nqubits)
    for i, (a, b) in enumerate(blocks):
        zf = z[:, a:b] / f_qubit[a:b]
        P[i], P_err[i] = _mean_and_error((np.prod(1 + zf, axis=1) + np.prod(1 - zf, axis=1)) / 2 ** (b - a), w)
        dv = z[:, a:b] / f_qubit[a:b] ** 2 * (_exclusive_products(1 - zf) - _exclusive_products(1 + zf)) / 2 ** (b - a)
        dP.append(w @ dv / w.sum())

    parity, parity_err = np.zeros((nblocks, len(phases))), np.zeros((nblocks, len(phases)))
    for k, counts in enumerate(parity_counts):
        z, w = counts_to_z(counts, nqubits)
        for i, (a, b) in enumerate(blocks):
            parity[i, k], parity_err[i, k] = _mean_and_error(np.prod(z[:, a:b], axis=1).astype(float), w)
    parity, parity_err = parity / f_block[:, None], parity_err / np.abs(f_block[:, None])

    # rho_01 = I_{-N} and the derivatives of chi and |rho_01| with respect to each parity
    weights = np.exp(-1j * sizes[:, None] * phases[None, :]) / len(phases)
    rho01 = np.sum(weights * parity, axis=1)
    chi = 2 * np.real(np.exp(1j * theta) * rho01)
    chi_abs = 2 * np.abs(rho01)
    dchi = 2 * np.real(np.exp(1j * theta)[:, None] * weights)
    dchi_abs = 2 * np.real(np.conj(rho01)[:, None] * weights) / np.where(chi_abs > 0, chi_abs / 2, np.inf)[:, None]
    chi_err = np.sqrt(np.sum((dchi * parity_err) ** 2, axis=1))
    chi_abs_err = np.sqrt(np.sum((dchi_abs * parity_err) ** 2, axis=1))
    F_err = np.sqrt(P_err ** 2 + chi_err ** 2) / 2
    F_abs_err = np.sqrt(P_err ** 2 + chi_abs_err ** 2) / 2

    if cal_counts is not None:
        # Calibration shot noise: every quantity depends on the factors through sum_q g_q f_q + g_b f_block, and its
        # variance is that of sum_q g_q z_q + g_b z_block over the calibration shots
        for i, (a, b) in enumerate(blocks):
            u = np.hstack([z_cal[:, a:b], np.prod(z_cal[:, a:b], axis=1, keepdims=True)]).astype(float)
            zero = np.zeros(b - a)
            g_P = np.append(dP[i], 0)
            g_chi = np.append(zero, -chi[i] / f_block[i])
            g_chi_abs = np.append(zero, -chi_abs[i] / f_block[i])
            cal_err = {name: _mean_and_error(u @ g, w_cal)[1] for name, g in (
                ("P", g_P), ("chi", g_chi), ("chi_abs", g_chi_abs), ("F", (g_P + g_chi) / 2), ("F_abs", (g_P + g_chi_abs) / 2))}
            P_err[i] = np.hypot(P_err[i], cal_err["P"])
            chi_err[i] = np.hypot(chi_err[i], cal_err["chi"])
            chi_abs_err[i] = np.hypot(chi_abs_err[i], cal_err["chi_abs"])
            F_err[i] = np.hypot(F_err[i], cal_err["F"])
            F_abs_err[i] = np.hypot(F_abs_err[i], cal_err["F_abs"])

    return {
        "blocks": blocks, "sizes": sizes, "phases": phases, "f_block": f_block,
        "P": P, "P_err": P_err, "chi": chi, "chi_err": chi_err, "chi_abs": chi_abs, "chi_abs_err": chi_abs_err,
        "F": (P + chi) / 2, "F_err": F_err, "F_abs": (P + chi_abs) / 2, "F_abs_err": F_abs_err,
        "rho01": rho01, "parity": parity, "parity_err": parity_err,
    }


def product_with_error(values, errors):
    """Product of independent estimates and its first-order error."""
    values, errors = np.asarray(values), np.asarray(errors)
    product = np.prod(values)
    return product, np.abs(product) * np.sqrt(np.sum((errors / values) ** 2))


def analyze_run(job_counts, cal_counts, labels, nqubits):
    """block_fidelities (raw and TREX) of every block size of a run, from the counts of its circuits in submission order."""
    results = {}
    for block_size in sorted({lab["block_size"] for lab in labels}):
        indices = [i for i, lab in enumerate(labels) if lab["block_size"] == block_size]
        z_counts = [job_counts[i] for i in indices if labels[i]["basis"] == "Z"][0]
        parity_counts = [job_counts[i] for i in indices if labels[i]["basis"] == "parity"]
        results[block_size] = {
            "raw": block_fidelities(z_counts, parity_counts, block_size, nqubits),
            "trex": block_fidelities(z_counts, parity_counts, block_size, nqubits, cal_counts=cal_counts),
        }
    return results


def two_qubit_gate_count(circuit):
    return sum(1 for instruction in circuit.data if instruction.operation.num_qubits == 2)


def exact_check(state_prep, block_size, cx_depolarizing=0.02, readout_flip=0.03, phase_error=0.0):
    """Estimator against the exact block fidelities on a small chain (density matrix, no shot noise).

    The state is prepared with depolarizing noise on every CX and, if phase_error != 0, an Rz(phase_error) on qubit 0
    (part of the prepared state, seen by chi but not by chi_abs). The measurement rotations are exact and every qubit
    flips its readout with probability readout_flip.
    """
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nqubits = state_prep.num_qubits
    prep = state_prep.copy()
    if phase_error != 0:
        prep.rz(phase_error, 0)
    prep.save_density_matrix()
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(cx_depolarizing, 2), ["cx"])
    simulator = AerSimulator(method="density_matrix", noise_model=noise_model, basis_gates=["cx", "h", "s", "rz"])
    rho = DensityMatrix(simulator.run(prep).result().data()["density_matrix"])

    flip = np.array([[1 - readout_flip, readout_flip], [readout_flip, 1 - readout_flip]])

    def measured_counts(state):
        probabilities = state.probabilities().reshape([2] * nqubits)      # axis -1 - q is qubit q
        for axis in range(nqubits):
            probabilities = np.moveaxis(np.tensordot(flip, probabilities, axes=([1], [axis])), 0, axis)
        return {format(i, f"0{nqubits}b"): p for i, p in enumerate(probabilities.reshape(-1)) if p > 1e-15}

    z_counts = measured_counts(rho)
    parity_counts = [measured_counts(rho.evolve(parity_basis_change(nqubits, phi))) for phi in parity_phases(block_size)]
    cal_counts = measured_counts(DensityMatrix.from_label("0" * nqubits))

    blocks = chain_blocks(nqubits, block_size)
    exact = []
    for a, b in blocks:
        n = b - a
        target = Statevector((np.eye(2 ** n)[0] + np.exp(1j * n * np.pi / 2) * np.eye(2 ** n)[-1]) / np.sqrt(2))
        reduced = partial_trace(rho, [q for q in range(nqubits) if not a <= q < b])
        exact.append(state_fidelity(reduced, target))
    return {
        "exact": np.array(exact),
        "raw": block_fidelities(z_counts, parity_counts, block_size, nqubits),
        "trex": block_fidelities(z_counts, parity_counts, block_size, nqubits, cal_counts=cal_counts),
    }
