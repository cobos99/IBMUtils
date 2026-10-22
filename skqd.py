"""Sample-based Krylov quantum diagonalization (SKQD), arXiv:2501.09702.

Workflow:
    1. Sample the Krylov states |psi_k> = e^{-i k H dt}|psi_0>, k = 0, ..., d-1, in the computational basis.
    2. Build the subspace spanned by the union of all sampled bitstrings B_{d,M}.
    3. Project H onto span(B_{d,M}) and diagonalize classically (qiskit-addon-sqd).

Bitstring convention: counts keys and bitstring-matrix rows follow Qiskit's ordering (leftmost
character = highest qubit index), which is the convention expected by qiskit_addon_sqd.qubit
together with a Qiskit SparsePauliOp.
"""
from qiskit_addon_sqd.qubit import solve_qubit, project_operator_to_subspace, sort_and_remove_duplicates
from qiskit_addon_sqd.counts import counts_to_arrays
from hardware.utils.samples import combine_samples_dicts
import numpy as np

def krylov_dt_from_spectral_range(hamiltonian, spectral_range=None):
    """dt = pi / (E_max - E_0) (Sec. II of the paper). Without spectral_range, the bound
    E_max - E_0 <= 2 * sum_i |c_i| is used, which gives a conservative (small) dt."""
    if spectral_range is None:
        spectral_range = 2 * np.sum(np.abs(hamiltonian.coeffs))
    return np.pi / spectral_range

def krylov_trotter_steps(krylov_dt, krylov_dim, trotter_dt):
    """Number of Trotter layers and actual Trotter step for each Krylov state k*krylov_dt."""
    times = krylov_dt * np.arange(krylov_dim)
    nsteps = np.ceil(np.round(times / trotter_dt, 10)).astype(int)
    step_dts = np.divide(times, nsteps, out=np.zeros_like(times), where=nsteps > 0)
    return times, nsteps, step_dts

def counts_to_bitstring_matrix(counts):
    """Bool bitstring matrix (one row per distinct bitstring) and empirical probabilities."""
    if len(counts) == 0:
        return np.zeros((0, 0), dtype=bool), np.zeros(0)
    return counts_to_arrays(counts)

def krylov_subspace(counts_per_krylov_state, krylov_dim=None, recovery_function=None, min_counts=1):
    """Union B_{d,M} of the bitstrings sampled from the first krylov_dim Krylov states.

    Args:
        counts_per_krylov_state: list of counts dicts, one per Krylov state (index k).
        krylov_dim: number d of Krylov states included. Defaults to all of them.
        recovery_function: optional map counts -> counts applied to each Krylov state before the
            union (e.g. Gauss Sector Recovery, the analogue of the paper's
            configuration recovery).
        min_counts: drop bitstrings observed fewer than min_counts times in the union.

    Returns:
        Sorted, de-duplicated bool bitstring matrix.
    """
    if krylov_dim is None:
        krylov_dim = len(counts_per_krylov_state)
    counts_list = list(counts_per_krylov_state[:krylov_dim])
    if recovery_function is not None:
        counts_list = [recovery_function(c) for c in counts_list]
    counts_list = [c for c in counts_list if len(c) > 0]
    if len(counts_list) == 0:
        raise ValueError("No bitstrings left to build the subspace")
    union = combine_samples_dicts(counts_list)
    if min_counts > 1:
        union = {k: v for k, v in union.items() if v >= min_counts}
    bsm, _ = counts_to_bitstring_matrix(union)
    bsm = sort_and_remove_duplicates(bsm)
    return bsm

def diagonalize(bitstring_matrix, hamiltonian, nstates=1, **scipy_kwargs):
    """Lowest nstates eigenpairs of H projected onto span(bitstring_matrix).

    Uses the qiskit-addon-sqd projection and falls back to dense diagonalization for subspaces
    too small for ARPACK.
    """
    bitstring_matrix = sort_and_remove_duplicates(bitstring_matrix)
    dim = bitstring_matrix.shape[0]
    if dim <= max(2 * nstates + 1, 64):
        hproj = project_operator_to_subspace(bitstring_matrix, hamiltonian).toarray()
        evals, evecs = np.linalg.eigh(hproj)
        return evals[:nstates], evecs[:, :nstates], bitstring_matrix
    scipy_kwargs = {"k": nstates, "which": "SA"} | scipy_kwargs
    evals, evecs = solve_qubit(bitstring_matrix, hamiltonian, **scipy_kwargs)
    order = np.argsort(evals)
    return evals[order], evecs[:, order], bitstring_matrix

def skqd_energies(counts_per_krylov_state, hamiltonian, krylov_dims=None, recovery_function=None, min_counts=1, nstates=1, verbose=False, **scipy_kwargs):
    """SKQD energy estimates as a function of the Krylov dimension d (convergence curve).

    Returns:
        krylov_dims, energies (shape (len(krylov_dims), nstates)), subspace dimensions.
    """
    if krylov_dims is None:
        krylov_dims = np.arange(1, len(counts_per_krylov_state) + 1)
    if recovery_function is not None:
        counts_per_krylov_state = [recovery_function(c) for c in counts_per_krylov_state]
    energies = np.full((len(krylov_dims), nstates), np.nan)
    subspace_dims = np.zeros(len(krylov_dims), dtype=int)
    for i, d in enumerate(krylov_dims):
        bsm = krylov_subspace(counts_per_krylov_state, d, min_counts=min_counts)
        subspace_dims[i] = bsm.shape[0]
        evals, _, _ = diagonalize(bsm, hamiltonian, nstates=min(nstates, bsm.shape[0]), **scipy_kwargs)
        energies[i, :len(evals)] = evals
        if verbose: print(f"\rd={d}, |B|={bsm.shape[0]}, E0={evals[0]:.8f}".ljust(60), end="")
    if verbose: print()
    return np.asarray(krylov_dims), energies, subspace_dims

def subspace_state_amplitudes(bitstring_matrix, eigenvector, top=None):
    """Bitstrings (as Qiskit-ordered strings) of the SKQD eigenvector sorted by |amplitude|^2."""
    probs = np.abs(eigenvector)**2
    order = np.argsort(probs)[::-1][:top]
    strings = ["".join("1" if b else "0" for b in row) for row in bitstring_matrix[order]]
    return strings, probs[order]

def exact_counts(statevector, shots=None, seed=None, threshold=1e-14):
    """Exact (shots=None) or sampled counts from a statevector, in Qiskit key ordering.
    Useful for noiseless benchmarking of the classical part of SKQD."""
    probs = np.abs(np.asarray(statevector))**2
    nqubits = int(np.log2(len(probs)))
    if shots is None:
        idx = np.nonzero(probs > threshold)[0]
        return {format(i, f"0{nqubits}b"): probs[i] for i in idx}
    rng = np.random.default_rng(seed)
    idx, cnts = np.unique(rng.choice(len(probs), size=shots, p=probs / probs.sum()), return_counts=True)
    return {format(i, f"0{nqubits}b"): int(c) for i, c in zip(idx, cnts)}
