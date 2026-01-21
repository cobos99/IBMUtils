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