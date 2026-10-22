# from juliacall import Main as jl, convert as jlconvert
# from collections.abc import Iterable

# jl.seval("include('DEC.jl')")
# jl.seval("using .DEC")

# DEC = jl.DEC

# def dec_samples_dicts(job_samples_dicts, cal_samples_dicts, alphas=0.1, normatols=1e-8, max_iters=1000, unsafe=False):
#     if len(job_samples_dicts) != len(cal_samples_dicts):
#         raise ValueError("job_samples_dicts and cal_samples_dicts must have the same length")
#     if isinstance(alphas, (float, int)):
#         alphas = [alphas]
#     elif isinstance(alphas, Iterable):
#         if len(alphas) != len(job_samples_dicts):
#             raise ValueError("Length of alphas must match length of job_samples_dicts")
#     else:
#         raise TypeError("alphas must be a float, int, or an iterable of floats/ints")
    
#     jl.seval("snoisy_count_vectors = DEC.SparseVecDict.(DEC.counts_vector.(samples_dicts); atol=1e-16)")
#     jl.seval("scal_count_vectors = DEC.SparseVecDict.(DEC.counts_vector.(cal_samples_dicts); atol=1e-16)")
#     snoisy_count_vectors = jl.snoisy_count_vectors
#     scal_count_vectors = jl.scal_count_vectors

from qiskit.result import QuasiDistribution
import numpy as np

try:  # sibling module inside the hardware.utils package (normal import path)
    from hardware.utils.sparse_wht_deconvolution import sparse_deconvolve
except ImportError:  # allow importing/running dec.py as a bare module too
    from sparse_wht_deconvolution import sparse_deconvolve

def first_col_update_label(col_label, key):
    num_qubits=len(col_label)
    return bin(int(col_label, 2) ^ int(key, 2))[2:].zfill(num_qubits)

def convert_to_first_col(counts: dict, col_label: str):
    '''The label is the associated column of tilde A. Counts is the associated column values.
    We should convert these to the first column by the symmetry rule described in the paper.'''
    counts={first_col_update_label(col_label, key):value for (key, value) in counts.items()}
    return counts

def counts_vector(samples_dict, col_label_int=0):
    n = len(list(samples_dict.keys())[0])
    samples_dict = convert_to_first_col(samples_dict, f"{col_label_int:0{n}b}")
    cv = np.zeros(2**n, dtype=int)
    for bitstring, count in samples_dict.items():
        index = int(bitstring, base=2)
        cv[index] += count
    return cv

def prob_vector(samples_dict, col_label_int=0):
    cv = counts_vector(samples_dict, col_label_int)
    total_counts = np.sum(cv)
    if total_counts == 0:
        return cv
    return cv / total_counts

def hadamard_transform(data, inverse=False):
    # In order to maintain the code vectorized a second dimension must be added.
    original_shape = data.shape
    if not 1 <= len(original_shape) <= 2:
        raise ValueError("The input data must be a one/two-dimensional array.")
    if len(original_shape) == 1:
        x = data[None]
    else:
        x = data
    batch_dim, d = x.shape

    batch_size2 = 2
    while batch_size2 <= d:
        batch_size = batch_size2 // 2
        x = x.reshape(batch_dim, d // batch_size2, batch_size2)
        half1, half2 = x[:, :, :batch_size], x[:, :, batch_size:]
        x = np.concatenate((half1 + half2, half1 - half2), axis=-1)
        batch_size2 = batch_size2 * 2
    
    if inverse:
        x = x / d
    # x = x / np.sqrt(d)
    
    return x.reshape(original_shape)

def quasiprobability_dict_dense(job_samples, cal_samples, cal_basis_state_int=0):
    nqb = len(list(job_samples.keys())[0])
    job_pv = counts_vector(job_samples)
    cal_pv = prob_vector(cal_samples, col_label_int=cal_basis_state_int)
    fhwt_job_pv = hadamard_transform(job_pv)
    fhwt_cal_pv = hadamard_transform(cal_pv)
    nonzero_mask = fhwt_cal_pv != 0
    fhwt_noiseless_pv = np.zeros_like(job_pv)
    fhwt_noiseless_pv[nonzero_mask] = fhwt_job_pv[nonzero_mask] / fhwt_cal_pv[nonzero_mask]
    noiseless_pv = hadamard_transform(fhwt_noiseless_pv, inverse=True)
    noiseless_pv = noiseless_pv / np.sum(noiseless_pv)
    return {f"{i:0{nqb}b}":prob for i, prob in enumerate(noiseless_pv) if prob > 0}

#============================================================================#
#   Sparse, full-register DEM: the same z = c * x  =>  x = ifhwt(fhwt(z)/    #
#   fhwt(c)) inversion as quasiprobability_dict_dense, but evaluated with    #
#   the SPARSE Walsh-Hadamard deconvolution (sparse_wht_deconvolution.py)    #
#   so it scales to n ~ 100 qubits, where the dense 2**n transform is        #
#   intractable (2**100 floats can neither be stored nor transformed).      #
#============================================================================#

def _truncate_counts(samples, min_count=None, max_support=None):
    '''Drop the shot-noise tail of a counts dict before feeding it to the sparse solver. The
    per-round cost of sparse_deconvolve is O(|support| * n * 2**m), and the smallest entries of a
    measured/calibration distribution are overwhelmingly single- or double-shot outcomes that carry
    almost no probability mass yet alias into the decoder's bins as noise - so trimming them both
    speeds the solve up and *denoises* it (empirically it improves, not just preserves, the mitigated
    result). `min_count` keeps only outcomes seen at least that many times (scale-free, the principled
    choice - e.g. 3 removes ones/twos); `max_support` additionally caps the dict to its top entries.
    Both default to None (no truncation, exactly the full-support behaviour).'''
    if min_count is not None:
        samples = {b: c for b, c in samples.items() if c >= min_count}
    if max_support is not None and len(samples) > max_support:
        top = sorted(samples.items(), key=lambda bc: bc[1], reverse=True)[:max_support]
        samples = dict(top)
    if not samples:
        raise ValueError("truncation removed every outcome; relax min_count/max_support")
    return samples

def _samples_to_sparse_signal(samples, shift_int=0, normalize=True):
    '''Turn a counts dict {bitstring: count} into the sparse {index: value} representation
    consumed by sparse_deconvolve, using the exact index convention of counts_vector/prob_vector
    (index = int(bitstring, 2), so bitstring[0] is the most significant bit). `shift_int` XORs
    every index (the sparse equivalent of convert_to_first_col: it shifts the calibration samples
    by the known noise-estimation output binit so they become the noise channel's first column).
    If `normalize`, values are divided by the total counts, turning the dict into a probability
    vector. Returns (signal_dict, n_qubits).'''
    if not samples:
        raise ValueError("samples dict is empty")
    n = len(next(iter(samples)))
    total = sum(samples.values()) if normalize else 1.0
    if total == 0:
        raise ValueError("samples dict has zero total counts")
    signal = {}
    for bitstring, count in samples.items():
        idx = int(bitstring, 2) ^ shift_int
        signal[idx] = signal.get(idx, 0.0) + count / total
    return signal, n

def quasiprobability_dict_sparse(job_samples, cal_samples, cal_basis_state_int=0,
                                 expected_k=None, m=None, max_rounds=200, stall_rounds=8,
                                 tol=1e-6, seed=None, verbose=False, verify_points=3,
                                 mag_tol=0.1, atom_rel_tol=0.05,
                                 min_count=None, max_support=None):
    '''Sparse counterpart of quasiprobability_dict_dense that scales Distribution Error
    Mitigation (DEC/DEM) to large registers (n ~ 100 qubits).

    Full DEM relates the noisy and noiseless output distributions z, x of a circuit through a
    bit-flip noise channel c that is a circulant (XOR-)convolution, z = c * x, and is inverted
    with the fast Walsh-Hadamard transform: x = ifhwt(fhwt(z) / fhwt(c)). quasiprobability_dict_dense
    does this by materializing dense length-2**n vectors and running a 2**n-point transform, which is
    O(2**n * n) - fine for ~20 qubits but hopeless at n = 100 (2**100 entries). Here the very same
    inversion is carried out by sparse_wht_deconvolution.sparse_deconvolve, which never materializes a
    length-2**n array: it evaluates fhwt(z)/fhwt(c) only at the O(n * 2**m) query points needed by a
    random-subsampling / peeling decoder and recovers x directly as a sparse dict, assuming (as DEM
    already does for the readout of a physically concentrated state) that the mitigated distribution
    has few significant outcomes. See sparse_wht_deconvolution.py and arXiv:2501.01953 / 2505.20387.

    job_samples, cal_samples : dict {bitstring: count}
        Noisy samples of, respectively, the circuit of interest (z) and the Clifford noise-estimation
        circuit (whose noiseless output is the basis state binit); same convention and bit ordering as
        quasiprobability_dict_dense. The calibration counts, XOR-shifted by binit, form c's first column.
    cal_basis_state_int : int
        Integer encoding of binit (digits ordered as the qubit axis, most-significant first). 0 (all
        zeros) is the standard echo calibration.
    expected_k : int, optional
        Rough upper bound on the number of SIGNIFICANT outcomes of the mitigated distribution (its
        sparsity) - typically tens to a few hundred for a physically concentrated state. Sizes m and is
        the early-stopping target. IMPORTANT: this is NOT the shot count and NOT len(job_samples); the
        per-round cost is O(|support| * n * 2**m) and m ~ log2(expected_k)+4, so passing the shot count
        (e.g. sum(job_samples.values()) = 2**18) makes 2**m explode into the millions - the usual cause
        of a multi-minute run. Set it to how many bitstrings actually carry weight, e.g. ~50-200.
    m : int, optional
        Subsampling dimension of the decoder (bins = 2**m). Auto-sized (and capped) from expected_k if
        None; the single dominant runtime knob since cost doubles with each unit of m.
    min_count, max_support : int, optional
        Trim the shot-noise tail of both distributions before solving (see _truncate_counts). This is
        the single most effective speed knob: since cost is O(|support| * n * 2**m), and the smallest
        outcomes are near-pure shot noise, trimming shrinks the solve and typically *improves* the
        result. Benchmark (n=100, 30 significant outcomes, 1% readout, 8192 shots): full support
        43s / TVD 0.058; capping both supports to ~1000 gives 10s / TVD 0.037. Recommended when
        speed matters: max_support ~ 30-50x expected_k, or min_count ~ 3. Default None (no trimming).
    max_rounds, stall_rounds, tol, seed, verbose, verify_points, mag_tol, atom_rel_tol :
        Passed straight through to sparse_deconvolve (see its docstring). mag_tol / atom_rel_tol
        are the noise-robustness knobs of the peeling decoder: the defaults tolerate the finite-shot
        noise floor of real measured distributions; raise mag_tol (toward ~0.3) for noisier data.

    Returns
    -------
    dict
        {bitstring: quasiprobability} over the recovered outcomes (n-bit strings, most-significant digit
        first), normalized to sum to 1 - the sparse analogue of quasiprobability_dict_dense's output,
        directly usable with QuasiDistribution(...).nearest_probability_distribution(). Unlike the dense
        version it also keeps small negative quasiprobabilities (it never enumerates 2**n, so there is no
        positivity scan); nearest_probability_distribution projects them away downstream.
    '''
    # Optionally trim the shot-noise tail of both distributions first (big speed-up, and usually a
    # quality gain too - see _truncate_counts). No-op when min_count and max_support are both None.
    job_samples = _truncate_counts(job_samples, min_count=min_count, max_support=max_support)
    cal_samples = _truncate_counts(cal_samples, min_count=min_count, max_support=max_support)

    # job -> x (kept as raw counts, like counts_vector in the dense version); cal -> c's first column,
    # shifted by binit and normalized to a probability vector (like prob_vector in the dense version).
    x_signal, n = _samples_to_sparse_signal(job_samples, shift_int=0, normalize=False)
    a_signal, n_cal = _samples_to_sparse_signal(cal_samples, shift_int=cal_basis_state_int, normalize=True)
    if n_cal != n:
        raise ValueError("job_samples and cal_samples must act on the same number of qubits")

    y = sparse_deconvolve(x_signal, a_signal, n, m=m, expected_k=expected_k,
                          max_rounds=max_rounds, stall_rounds=stall_rounds, tol=tol,
                          seed=seed, verbose=verbose, verify_points=verify_points,
                          mag_tol=mag_tol, atom_rel_tol=atom_rel_tol)
    total = sum(y.values())
    if total == 0:
        return {}
    return {f"{i:0{n}b}": v / total for i, v in y.items()}

def quasiprobability_dict(job_samples, cal_samples, cal_basis_state_int=0, dense_max_qubits=20,
                          **sparse_kwargs):
    '''Dispatch DEM to the dense O(2**n) solver for small registers and to the sparse solver
    (quasiprobability_dict_sparse) once n exceeds `dense_max_qubits`, so the same call site works from a
    handful of qubits up to n ~ 100. Extra keyword arguments (expected_k, m, seed, ...) are forwarded to
    the sparse solver and ignored in the dense branch.'''
    n = len(next(iter(job_samples)))
    if n <= dense_max_qubits:
        return quasiprobability_dict_dense(job_samples, cal_samples, cal_basis_state_int)
    return quasiprobability_dict_sparse(job_samples, cal_samples, cal_basis_state_int, **sparse_kwargs)

#============================================================================#
#   Marginal DEM (mDEM): DEC restricted to a subset of qubits, O(2^k * k)    #
#   instead of O(2^n * n), for a subset of size k around an observable of   #
#   interest. Gonzales, arXiv:2501.01953 / arXiv:2606.19601 (full DEM);     #
#   Schuhmacher et al., arXiv:2505.20387 Supp. Mat. (marginal DEM / mDEM).  #
#============================================================================#

def bits_to_indices(bits):
    '''Pack the last axis of a boolean array into integers via positional binary encoding,
    matching the convention used by counts_vector/prob_vector above: the first entry along the
    packed axis is the most significant bit, i.e. bits[..., 0] carries weight 2**(k-1).'''
    bits = np.asarray(bits)
    k = bits.shape[-1]
    weights = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
    return bits.astype(np.int64) @ weights

def counts_vector_from_bits(bits):
    '''Dense counts vector (length 2**k) of a boolean sample array of shape (shots, k), the
    array-based equivalent of counts_vector for samples given as dictionaries.'''
    bits = np.asarray(bits, dtype=bool)
    if bits.ndim != 2:
        raise ValueError("bits must be a two-dimensional (shots, qubits) array")
    k = bits.shape[-1]
    indices = bits_to_indices(bits)
    return np.bincount(indices, minlength=2**k).astype(float)

def samples_num_qubits(samples):
    '''Number of qubits (the length of the bitstring/last axis) of a samples object given either as a
    boolean sample array (last axis = qubits) or as a counts dictionary {bitstring: count}.'''
    if isinstance(samples, dict):
        if not samples:
            raise ValueError("samples dictionary is empty")
        return len(next(iter(samples)))
    return np.asarray(samples).shape[-1]

def marginal_counts_vector(samples, qubit_indices, flip_bits=None):
    '''Length-2**k marginal counts vector over `qubit_indices` for samples given either as a boolean
    sample array (any leading axes, last axis = qubits) or as a counts dictionary {bitstring: count};
    the two representations are histogrammed with the same most-significant-first bit ordering as
    counts_vector/counts_vector_from_bits, so qubit_indices[0] carries weight 2**(k-1) in the result.
    `flip_bits`, if given, is a length-k boolean array XORed into each marginal bitstring before it is
    histogrammed - used to shift the calibration samples by binit restricted to C (the array/dict
    equivalent of convert_to_first_col/first_col_update_label of the full-register DEM above).'''
    qubit_indices = np.asarray(qubit_indices, dtype=int)
    k = qubit_indices.size
    if isinstance(samples, dict):
        qi = qubit_indices.tolist()
        flip_int = 0 if flip_bits is None else int(bits_to_indices(np.asarray(flip_bits, dtype=bool)))
        cv = np.zeros(2**k, dtype=float)
        for bitstring, count in samples.items():
            sub_int = int("".join(bitstring[j] for j in qi), 2) ^ flip_int
            cv[sub_int] += count
        return cv
    arr = np.asarray(samples, dtype=bool)
    arr = arr.reshape(-1, arr.shape[-1])
    sub = arr[:, qubit_indices]
    if flip_bits is not None:
        sub = sub ^ np.asarray(flip_bits, dtype=bool)
    return counts_vector_from_bits(sub)

def marginal_dec(job_samples, cal_samples, qubit_indices, cal_basis_state_int=0, eps=None):
    '''Approximate, reduced-cost Distribution Error Mitigation (DEC/DEM) that only reconstructs the
    noiseless *marginal* distribution over a chosen subset of qubits `qubit_indices` (e.g. the qubits
    acted on by an observable of interest), instead of the full 2**n-dimensional distribution.

    Full DEM relates the noisy and noiseless output distributions of a circuit, z and x, via a Pauli
    (bit-flip) noise channel c that is block-circulant under XOR, and is efficiently inverted with the
    fast Walsh-Hadamard transform (fwht): z = c * x  =>  x = ifwht(fwht(z) / fwht(c)), at a cost of
    O(2**n * n) [Gonzales, arXiv:2501.01953; see also arXiv:2606.19601 App. B]. The noise channel c is
    itself learned from noisy samples z0 of a Clifford "noise-estimation" circuit whose noiseless output
    is a known computational basis state binit, since z0 = c * delta_binit is just c shifted by binit.

    Marginalizing both z and x to a subset of qubits C preserves this XOR-convolution/block-circulant
    structure with a *marginal* noise channel c_C (itself the marginal of c over C), so the very same
    fwht-based inversion can be run entirely in the 2**k-dimensional subspace, k = len(qubit_indices):
    z_C = c_C * x_C => x_C = ifwht(fwht(z_C) / fwht(c_C)), at a cost of O(2**k * k) << O(2**n * n). This
    is the "marginal DEM" (mDEM) scheme of Schuhmacher et al., arXiv:2505.20387, Supplementary Material.

    Parameters
    ----------
    job_samples : array_like of bool or dict
        Noisy samples of the circuit of interest, given in either of two interchangeable forms: (i) a
        boolean sample array whose last axis indexes the qubits, with all leading axes (shots,
        parameter-randomization repeats, ...) flattened away, i.e. reshaped as (-1, n_qubits); or
        (ii) a counts dictionary {bitstring: count} mapping length-n_qubits bitstrings to (integer or
        float) counts, using the same most-significant-first bit ordering as counts_vector/prob_vector
        (bitstring[j] is qubit j). job_samples and cal_samples may independently be arrays or dicts.
    cal_samples : array_like of bool or dict
        Noisy samples of the Clifford noise-estimation circuit used to characterize the noise channel,
        in either of the same two forms and with the same number of qubits as job_samples.
    qubit_indices : array_like of int
        Indices into the shared last axis of job_samples/cal_samples of the qubit subset C the DEC
        inversion is restricted to (e.g. the qubits surrounding an observable of interest).
    cal_basis_state_int : int, optional
        Integer encoding (same convention as counts_vector/prob_vector, i.e. binary digits ordered as
        the qubit axis itself, most-significant first) of binit, the known noiseless output bitstring
        of the noise-estimation circuit. Defaults to 0, the standard all-zeros "echo" calibration.
    eps : float, optional
        If given, floors the reconstructed noise eigenvalues fwht(c_C) at eps before inversion, an
        eigenvalue regularization that keeps the learned noise map numerically stable and projects it
        onto a completely positive map (arXiv:2505.20387, Eq. (S4)). If None (default), entries where
        the eigenvalue is exactly zero are left un-mitigated, matching quasiprobability_dict_dense.

    Returns
    -------
    dict
        Mapping from k-bit marginal bitstrings over qubit_indices (in the given order, most-significant
        digit first) to the mitigated quasi-probabilities.
    '''
    n = samples_num_qubits(job_samples)
    if samples_num_qubits(cal_samples) != n:
        raise ValueError("job_samples and cal_samples must act on the same number of qubits")

    qubit_indices = np.asarray(qubit_indices, dtype=int)
    if qubit_indices.ndim != 1 or qubit_indices.size == 0:
        raise ValueError("qubit_indices must be a non-empty one-dimensional array of qubit indices")
    if len(set(qubit_indices.tolist())) != qubit_indices.size:
        raise ValueError("qubit_indices must not contain repeated entries")
    if np.any(qubit_indices < 0) or np.any(qubit_indices >= n):
        raise ValueError("qubit_indices must index qubits within [0, n_qubits)")
    k = qubit_indices.size

    cal_basis_state = np.array([c == "1" for c in f"{cal_basis_state_int:0{n}b}"])

    job_cv = marginal_counts_vector(job_samples, qubit_indices)
    # Marginalizing the calibration samples to C and shifting by binit restricted to C directly gives
    # the (marginal) first column of the noise matrix, c_C - the array/dict equivalent of what
    # convert_to_first_col/first_col_update_label do for the full-register, dict-based DEM above.
    cal_cv = marginal_counts_vector(cal_samples, qubit_indices, flip_bits=cal_basis_state[qubit_indices])
    total_cal_counts = cal_cv.sum()
    if total_cal_counts == 0:
        raise ValueError("cal_samples contains no shots after restricting to qubit_indices")
    cal_pv = cal_cv / total_cal_counts

    fhwt_job = hadamard_transform(job_cv)
    fhwt_cal = hadamard_transform(cal_pv)
    if eps is not None:
        fhwt_cal = np.maximum(fhwt_cal, eps)
        nonzero_mask = np.ones_like(fhwt_cal, dtype=bool)
    else:
        nonzero_mask = fhwt_cal != 0

    fhwt_mitig = np.zeros_like(job_cv, dtype=float)
    fhwt_mitig[nonzero_mask] = fhwt_job[nonzero_mask] / fhwt_cal[nonzero_mask]
    mitig_pv = hadamard_transform(fhwt_mitig, inverse=True)
    mitig_pv = mitig_pv / np.sum(mitig_pv)
    return {f"{i:0{k}b}": prob for i, prob in enumerate(mitig_pv) if prob != 0}