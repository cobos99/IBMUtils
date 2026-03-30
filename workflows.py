from juliacall import Main as jl, convert as jlconvert
from qiskit.quantum_info import Pauli, PauliList
from qiskit.result import QuasiDistribution
import hardware.utils.postselection as ps
import hardware.utils.measure as measure
import hardware.utils.samples as samples
import hardware.utils.zne as znemod
import hardware.utils.misc as miscutils
import hardware.utils.odr as odr
import hardware.utils.dec as dec
import multiprocessing as mp
import numpy as np
import pickle
import os

class WorkflowError(Exception):
    pass

def diag_obs_postselection_and_odr(job_samples, cal_samples, job_data, postselection_genfunc, observables_genfuncs, 
                                   flip_threshold=None, flipped_syndromes=None,
                                   minimum_shots=None, confidence=None, 
                                   return_samples=False, return_postselected_samples=False,
                                   postselect_odr=True, odr_expected_observable=1,
                                   verbose=False):

    if (jshape := job_samples.shape) != cal_samples.shape:
        raise ValueError("job_samples and cal_samples must have the same shape")
    if len(jshape) == 2:
        # Combine samples dicts from different layouts
        if verbose: print("\rCombining samples dicts from different layouts...".ljust(100), end="")
        job_samples = samples.multilayout_combine_samples_dicts(job_samples)
        cal_samples = samples.multilayout_combine_samples_dicts(cal_samples)
    elif len(jshape) == 1:
        job_samples = np.array(job_samples, dtype=object).flatten()
        cal_samples = np.array(cal_samples, dtype=object).flatten()
    elif len(job_samples.shape) != 1:
        raise ValueError("job_samples and cal_samples must be at most a matrix of samples dicts")
    nqubits = len(list(job_samples.flatten()[0].keys())[0])
    postselection_ops, diag_observables, nondiag_observables, basis = ps.initialize_postselection(nqubits, postselection_genfunc, observables_genfuncs)
    if basis != "Z":
        raise ValueError("The provided observables are not diagonal in the Z basis")

    if verbose: print("\rGenerating recovered samples dicts...".ljust(100), end="")
    recovered_job_samples = ps.get_recovered_samples(job_samples, postselection_ops, 
                                                     flip_threshold=flip_threshold,
                                                     flipped_syndromes=flipped_syndromes,
                                                     minimum_shots=minimum_shots,
                                                     recovery_function=ps.gauge_recovery)
    recovered_cal_samples = ps.get_recovered_samples(cal_samples, postselection_ops, 
                                                     flip_threshold=flip_threshold,
                                                     flipped_syndromes=flipped_syndromes,
                                                     minimum_shots=minimum_shots,
                                                     recovery_function=ps.gauge_recovery)
    
    if verbose: print("\rMeasuring diagonal observables...".ljust(100), end="")
    raw_observables, raw_observables_confidence = measure.many_measure_diagonal_observables(job_samples, diag_observables, confidence=0 if confidence is None else confidence)
    recovered_observables, recovered_observables_confidence = measure.many_measure_diagonal_observables(recovered_job_samples, diag_observables, confidence=0 if confidence is None else confidence)
    if postselect_odr:
        cal_observables, cal_observables_confidence = measure.many_measure_diagonal_observables(recovered_cal_samples, diag_observables, confidence=0 if confidence is None else confidence)
    else:
        cal_observables, cal_observables_confidence = measure.many_measure_diagonal_observables(cal_samples, diag_observables, confidence=0 if confidence is None else confidence)
    clean_observables = np.zeros_like(recovered_observables)
    clean_observables_confidence = np.zeros_like(recovered_observables_confidence)
    for i in range(len(recovered_observables)):
        odr_rate = odr.compute_depolarizing_probability(cal_observables[i], odr_expected_observable, 0, nqubits)
        clean_observables[i], clean_observables_confidence[i] = odr.renormalize_observable(recovered_observables[i], odr_rate, 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observable, observable_confidence=recovered_observables_confidence[i])

    if verbose: print(f"\rLoaded {len(job_samples.flatten()) + len(cal_samples.flatten())} jobs".ljust(100))
    to_return = [[clean_observables, recovered_observables, cal_observables, raw_observables]]
    if confidence is not None:
        to_return.append([clean_observables_confidence, recovered_observables_confidence, cal_observables_confidence, raw_observables_confidence])
    if return_samples:
        to_return.append([job_samples, cal_samples])
    if return_postselected_samples:
        to_return.append([recovered_job_samples, recovered_cal_samples])
    return to_return if len(to_return) > 1 else to_return[0]

def diag_obs_odr(job_samples, cal_samples, job_data, observables_genfuncs, confidence=None,
                 return_samples=False, odr_expected_observable=1, verbose=False):
    if (jshape := job_samples.shape) != cal_samples.shape:
        raise ValueError("job_samples and cal_samples must have the same shape")
    if not isinstance(observables_genfuncs, (tuple, list)):
        observables_genfuncs = [observables_genfuncs]
    if 0 < len(jshape) < 3:
        if len(jshape) == 2 and jshape[0] == 1:
            observables_final_shape = (job_samples.shape[0], len(observables_genfuncs))
            observables_confidence_final_shape = (job_samples.shape[0], len(observables_genfuncs), 2)
            jshape = (jshape[1],)
        elif len(jshape) == 2:
            observables_final_shape = (job_samples.shape[0], job_samples.shape[1], len(observables_genfuncs))
            observables_confidence_final_shape = (job_samples.shape[0], job_samples.shape[1], len(observables_genfuncs), 2)
        else:
            observables_final_shape = (job_samples.shape[0], len(observables_genfuncs))
            observables_confidence_final_shape = (job_samples.shape[0], len(observables_genfuncs), 2)
        job_samples = np.array(job_samples, dtype=object).flatten()
        cal_samples = np.array(cal_samples, dtype=object).flatten()
    else:
        raise ValueError("job_samples and cal_samples must be at most a matrix of samples dicts")
    nqubits = len(list(job_samples.flatten()[0].keys())[0])
    diagonal_observbles_ops, non_diagonal_observables_ops = measure.generate_observables(nqubits, observables_genfuncs)
    if len(non_diagonal_observables_ops) > 0:
        raise ValueError("The provided observables are not diagonal in the Z basis")

    if verbose: print("\rMeasuring diagonal observables...".ljust(100), end="")
    raw_observables, raw_observables_confidence = measure.many_measure_diagonal_observables(job_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence)
    cal_observables, cal_observables_confidence = measure.many_measure_diagonal_observables(cal_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence)
    clean_observables = np.zeros_like(raw_observables)
    clean_observables_confidence = np.zeros_like(raw_observables_confidence)
    for i in range(len(raw_observables)):
        odr_rate = odr.compute_depolarizing_probability(cal_observables[i], odr_expected_observable, 0, nqubits)
        clean_observables[i], clean_observables_confidence[i] = odr.renormalize_observable(raw_observables[i], odr_rate, 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observable, observable_confidence=raw_observables_confidence[i])
    if len(jshape) == 2:
        final_raw_observables = np.zeros(observables_final_shape)
        final_cal_observables = np.zeros(observables_final_shape)
        final_clean_observables = np.zeros(observables_final_shape)
        final_raw_observables_confidence = np.zeros(observables_confidence_final_shape)
        final_cal_observables_confidence = np.zeros(observables_confidence_final_shape)
        final_clean_observables_confidence = np.zeros(observables_confidence_final_shape)
        for i in range(observables_final_shape[0]):
            for j in range(observables_final_shape[1]):
                for k in range(observables_final_shape[2]):
                    final_raw_observables[i, j, k] = raw_observables[i*observables_final_shape[1]+j, k]
                    final_cal_observables[i, j, k] = cal_observables[i*observables_final_shape[1]+j, k]
                    final_clean_observables[i, j, k] = clean_observables[i*observables_final_shape[1]+j, k]
                    final_raw_observables_confidence[i, j, k, :] = raw_observables_confidence[i*observables_final_shape[1]+j, k, :]
                    final_cal_observables_confidence[i, j, k, :] = cal_observables_confidence[i*observables_final_shape[1]+j, k, :]
                    final_clean_observables_confidence[i, j, k, :] = clean_observables_confidence[i*observables_final_shape[1]+j, k, :]
        raw_observables = final_raw_observables
        cal_observables = final_cal_observables
        clean_observables = final_clean_observables
        raw_observables_confidence = final_raw_observables_confidence
        cal_observables_confidence = final_cal_observables_confidence
        clean_observables_confidence = final_clean_observables_confidence
        job_samples = job_samples.reshape(jshape)
        cal_samples = cal_samples.reshape(jshape)
    if verbose: print(f"\rLoaded {len(job_samples.flatten()) + len(cal_samples.flatten())} jobs".ljust(100))
    to_return = [[clean_observables, cal_observables, raw_observables]]
    if confidence is not None:
        to_return.append([clean_observables_confidence, cal_observables_confidence, raw_observables_confidence])
    if return_samples:
        to_return.append([job_samples, cal_samples])
    return to_return if len(to_return) > 1 else to_return[0]

def diag_obs_odr_byrows(job_samples, cal_samples, job_data, confidence=None,
                        return_samples=False, verbose=False):
    jobs_shape = job_samples.shape
    cal_shape = cal_samples.shape
    if jobs_shape != cal_shape:
        raise ValueError("job_samples and cal_samples must have the same number of columns")
    
    for samples in job_samples.flatten():
        try:
            nqubits = len(list(samples.keys())[0])
            break
        except IndexError:
            continue
    else:
        raise ValueError("Every job_samples entry is empty, unvalid batch")
    
    try:
        sim_observables_str = job_data["sim_observables"]
    except KeyError:
        raise WorkflowError("job_data must contain 'sim_observables' key")
    diagonal_observables_ops = [[measure.invert_pauli_order(s) for s in group] for group in sim_observables_str]
    if any(not measure.is_diagonal(op) for op in miscutils.iterate_nested_list(diagonal_observables_ops)):
        raise WorkflowError("The observables in 'sim_observables' are not all diagonal in the Z basis")
    if any(len(op) != nqubits for op in miscutils.iterate_nested_list(diagonal_observables_ops)):
        raise WorkflowError("The observables in 'sim_observables' do not all match the number of qubits in the samples")
    diagonal_observables_ops = miscutils.to_padded_array(diagonal_observables_ops, fill_value=np.nan, dtype=object)

    try:
        cal_observables_str = job_data["cal_observables"]
    except KeyError:
        raise WorkflowError("job_data must contain 'cal_observables' key")
    cal_diagonal_observables_ops = [[measure.invert_pauli_order(s) for s in group] for group in cal_observables_str]
    if any(not measure.is_diagonal(op) for op in miscutils.iterate_nested_list(cal_diagonal_observables_ops)):
        raise WorkflowError("The observables in 'cal_observables' are not all diagonal in the Z basis")
    if any(len(op) != nqubits for op in miscutils.iterate_nested_list(cal_diagonal_observables_ops)):
        raise WorkflowError("The observables in 'cal_observables' do not all match the number of qubits in the samples")
    cal_diagonal_observables_ops = miscutils.to_padded_array(cal_diagonal_observables_ops, fill_value=np.nan, dtype=object)

    try:
        odr_expected_observables = np.array(job_data["odr_expected_observables"])
    except KeyError:
        raise WorkflowError("job_data must contain 'odr_expected_observables' key")
    if odr_expected_observables.shape != diagonal_observables_ops.shape:
        raise WorkflowError("The shape of 'odr_expected_observables' does not match the shape of 'sim_observables'")

    # Measure raw job and calibration observables
    if verbose: print("\rMeasuring job observables...".ljust(100), end="")
    raw_observables = np.full((*job_samples.shape, diagonal_observables_ops.shape[1]), np.nan)
    raw_observables_confidence = np.full((*job_samples.shape, diagonal_observables_ops.shape[1], 2), np.nan)
    cal_observables = np.full(cal_samples.shape, np.nan)
    cal_observables_confidence = np.full((*cal_samples.shape, 2), np.nan)
    for i in range(jobs_shape[0]):
        last_obs_ind = len([obs for obs in diagonal_observables_ops[i, :] if obs is not np.nan])
        raw_observables[i, :, :last_obs_ind], raw_observables_confidence[i, :, :last_obs_ind, :] = measure.many_measure_diagonal_observables(job_samples[i, :], diagonal_observables_ops[i, :last_obs_ind], confidence=0 if confidence is None else confidence)

    if verbose: print("\rMeasuring calibration observables...".ljust(100), end="")
    p_arr = np.full((*cal_samples.shape, cal_diagonal_observables_ops.shape[1]), np.nan)
    cal_observables = np.full((*cal_samples.shape, cal_diagonal_observables_ops.shape[1]),  np.nan)
    cal_observables_confidence = np.full((*cal_samples.shape, cal_diagonal_observables_ops.shape[1], 2), np.nan)
    for i in range(cal_shape[0]):
        last_obs_ind = len([obs for obs in cal_diagonal_observables_ops[i, :] if obs is not np.nan])
        cal_observables[i, :, :last_obs_ind], cal_observables_confidence[i, :, :last_obs_ind, :] = measure.many_measure_diagonal_observables(cal_samples[i, :], cal_diagonal_observables_ops[i, :last_obs_ind], confidence=0 if confidence is None else confidence)
        p_arr[i, :, :last_obs_ind] = odr.compute_depolarizing_probability(cal_observables[i, :, :last_obs_ind], odr_expected_observables[i, :last_obs_ind], 0, nqubits)

    # Renormalize job observables
    clean_observables, clean_observables_confidence = odr.renormalize_observable(raw_observables, p_arr, 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observables, observable_confidence=raw_observables_confidence)
    clean_observables, clean_observables_confidence = np.zeros_like(cal_observables), np.zeros_like(cal_observables_confidence)
    if verbose: print(f"\rLoaded {len(job_samples.flatten()) + len(cal_samples.flatten())} jobs".ljust(100))
    to_return = [[clean_observables, cal_observables, raw_observables]]
    if confidence is not None:
        to_return.append([clean_observables_confidence, cal_observables_confidence, raw_observables_confidence])
    if return_samples:
        to_return.append([job_samples, cal_samples])
    return to_return if len(to_return) > 1 else to_return[0]

def diag_obs_odr_byweight(job_samples, cal_samples, job_data, confidence=None,
                          return_samples=False, verbose=False):
    jobs_shape = job_samples.shape
    cal_shape = cal_samples.shape
    if jobs_shape[1] != cal_shape[1]:
        raise ValueError("job_samples and cal_samples must have the same number of columns")

    for samples in job_samples.flatten():
        try:
            nqubits = len(list(samples.keys())[0])
            break
        except IndexError:
            continue
    else:
        raise ValueError("Every job_samples entry is empty, unvalid batch")
    
    try:
        sim_observables_str = job_data["sim_observables"]
    except KeyError:
        raise WorkflowError("job_data must contain 'sim_observables' key")
    diagonal_observables_ops = [[measure.invert_pauli_order(s) for s in group] for group in sim_observables_str]
    if any(not measure.is_diagonal(op) for op in miscutils.iterate_nested_list(diagonal_observables_ops)):
        raise WorkflowError("The observables in 'sim_observables' are not all diagonal in the Z basis")
    if any(len(op) != nqubits for op in miscutils.iterate_nested_list(diagonal_observables_ops)):
        raise WorkflowError("The observables in 'sim_observables' do not all match the number of qubits in the samples")
    diagonal_observables_ops = miscutils.to_padded_array(diagonal_observables_ops, fill_value=np.nan, dtype=object)
    
    try:
        cal_observables_str = job_data["cal_observables"]
    except KeyError:
        raise WorkflowError("job_data must contain 'cal_observables' key")
    cal_diagonal_observables_ops = [measure.invert_pauli_order(s) for s in cal_observables_str]
    cal_weights = [measure.pauli_weight(op) for op in cal_diagonal_observables_ops]
    if any(not measure.is_diagonal(op) for op in cal_diagonal_observables_ops):
        raise WorkflowError("The observables in 'cal_observables' are not all diagonal in the Z basis")
    if any(len(op) != nqubits for op in cal_diagonal_observables_ops):
        raise WorkflowError("The observables in 'cal_observables' do not all match the number of qubits in the samples")

    try:
        odr_expected_observables = job_data["odr_expected_observables"]
    except KeyError:
        raise WorkflowError("job_data must contain 'odr_expected_observables' key")

    # Measure raw job and calibration observables
    if verbose: print("\rMeasuring job observables...".ljust(100), end="")
    raw_observables = np.full((*job_samples.shape, diagonal_observables_ops.shape[1]), np.nan)
    raw_observables_confidence = np.full((*job_samples.shape, diagonal_observables_ops.shape[1], 2), np.nan)
    cal_observables = np.full(cal_samples.shape, np.nan)
    cal_observables_confidence = np.full((*cal_samples.shape, 2), np.nan)
    p_arr = np.full(cal_samples.shape, np.nan)
    for i in range(jobs_shape[0]):
        last_obs_ind = len([obs for obs in diagonal_observables_ops[i, :] if obs is not np.nan])
        raw_observables[i, :, :last_obs_ind], raw_observables_confidence[i, :, :last_obs_ind, :] = measure.many_measure_diagonal_observables(job_samples[i, :], diagonal_observables_ops[i, :last_obs_ind], confidence=0 if confidence is None else confidence)
    
    if verbose: print("\rMeasuring calibration observables...".ljust(100), end="")
    for i in range(cal_shape[0]):
        obs, obs_conf = measure.many_measure_diagonal_observables(cal_samples[i, :], [cal_diagonal_observables_ops[i]], confidence=0 if confidence is None else confidence)
        cal_observables[i, :], cal_observables_confidence[i, :, :] = obs[:, 0], obs_conf[:, 0, :]
        p_arr[i, :] = odr.compute_depolarizing_probability(cal_observables[i, :], odr_expected_observables[i], 0, nqubits)
    
    # Renormalize job observables
    clean_observables = np.full_like(raw_observables, np.nan)
    clean_observables_confidence = np.full_like(raw_observables_confidence, np.nan)
    for i in range(jobs_shape[0]):
        for k in range(diagonal_observables_ops.shape[1]):
            if diagonal_observables_ops[i, k] is np.nan:
                continue
            this_obs_weight = measure.pauli_weight(diagonal_observables_ops[i, k])
            cal_weight_index = cal_weights.index(this_obs_weight)
            clean_observables[i, :, k], clean_observables_confidence[i, :, k, :] = odr.renormalize_observable(raw_observables[i, :, k], p_arr[cal_weight_index, :], 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observables[cal_weight_index], observable_confidence=raw_observables_confidence[i, :, k, :])

    if verbose: print(f"\rLoaded {len(job_samples.flatten()) + len(cal_samples.flatten())} jobs".ljust(100))
    to_return = [[clean_observables, cal_observables, raw_observables]]
    if confidence is not None:
        to_return.append([clean_observables_confidence, cal_observables_confidence, raw_observables_confidence])
    if return_samples:
        to_return.append([job_samples, cal_samples])
    return to_return if len(to_return) > 1 else to_return[0]

def layout_zne(job_samples, job_data, observables_genfuncs,
               confidence=None, return_samples=False, verbose=False):
    try:
        layouts_zne = np.array(job_data["zne_layouts"])
    except KeyError:
        raise WorkflowError("job_data must contain the 'layouts' key")
    try:
        noise_rates = np.array(job_data["noise_rates"])
    except KeyError:
        raise WorkflowError("job_data must contain the 'noise_rates' key")
    if not isinstance(observables_genfuncs, (tuple, list)):
        observables_genfuncs = [observables_genfuncs]
    raw_observables_shape = (*job_samples.shape, len(observables_genfuncs))
    job_samples = job_samples.flatten()
    nqubits = len(list(job_samples.flatten()[0].keys())[0])
    diagonal_observbles_ops, non_diagonal_observables_ops = measure.generate_observables(nqubits, observables_genfuncs)

    # Measure observables
    if verbose: print("\rMeasuring diagonal observables...".ljust(100), end="")
    raw_observables, raw_observables_confidence = measure.many_measure_diagonal_observables(job_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence)
    raw_observables = raw_observables.reshape(*raw_observables_shape)
    raw_observables_confidence = raw_observables_confidence.reshape(*raw_observables_shape, 2)

    # Average over the cycles of the same layout
    layout_mean_observables = np.zeros((layouts_zne.shape[0], *raw_observables_shape[1:]))
    layout_mean_observables_confdelta = np.zeros_like(layout_mean_observables)
    for i in range(layouts_zne.shape[0]):
        this_layout_cycles_obs = raw_observables[layouts_zne.shape[1]*i:layouts_zne.shape[1]*(i+1)]
        layout_mean_observables[i] = np.mean(this_layout_cycles_obs, axis=0)
        this_observables_confdelta = np.abs(raw_observables_confidence[layouts_zne.shape[1]*i:layouts_zne.shape[1]*(i+1), ..., 0] - raw_observables_confidence[layouts_zne.shape[1]*i:layouts_zne.shape[1]*(i+1), ..., 1]) / 2
        layout_mean_observables_confdelta[i] = np.sqrt(np.sum(this_observables_confdelta**2/len(this_observables_confdelta), axis=0))

    # ZNE extrapolation
    observables_for_reg = layout_mean_observables.T.reshape(-1, layouts_zne.shape[0])
    confdelta_for_reg = layout_mean_observables_confdelta.T.reshape(-1, layouts_zne.shape[0])
    noise_rates_for_reg = noise_rates.T.reshape(-1, layouts_zne.shape[0])
    noise_rates_for_reg = np.tile(noise_rates_for_reg, (observables_for_reg.shape[0]//noise_rates_for_reg.shape[0], 1))
    # noise_rates_for_reg = np.repeat(noise_rates_for_reg, observables_for_reg.shape[0]//noise_rates_for_reg.shape[0], axis=0)
    results = [zne.perform_linear_zne(nr, obs) for nr, obs in zip(noise_rates_for_reg, observables_for_reg)]
    zne_observables = np.array([res.intercept for res in results]).reshape(layout_mean_observables.shape[1::][::-1]).T
    zne_observables_condelta = np.array([res.std_err for res in results]).reshape(layout_mean_observables.shape[1::][::-1]).T
    if verbose: print(f"\rLoaded {len(job_samples.flatten())} jobs".ljust(100))
    to_return = [[zne_observables, layout_mean_observables, raw_observables]]
    if confidence is not None:
        to_return.append([zne_observables_condelta, layout_mean_observables_confdelta, raw_observables_confidence])
    if return_samples:
        to_return.append(job_samples)
    return to_return if len(to_return) > 1 else to_return[0]

def zne(job_samples, job_data, observables_genfuncs, 
        zne_func=znemod.perform_linear_zne,
        confidence=None, return_samples=False, verbose=False):
    try:
        noise_rates = np.array(job_data["noise_rates"])
    except KeyError:
        raise WorkflowError("job_data must contain the 'noise_rates' key")
    if not isinstance(observables_genfuncs, (tuple, list)):
        observables_genfuncs = [observables_genfuncs]
    raw_observables_shape = (*job_data["job_circuits_shape"], len(observables_genfuncs))
    job_samples = np.array(job_samples, dtype=object).flatten()
    nqubits = len(list(job_samples[0].keys())[0])
    diagonal_observbles_ops, non_diagonal_observables_ops = measure.generate_observables(nqubits, observables_genfuncs)
    if non_diagonal_observables_ops:
        raise ValueError("The provided observables are not diagonal in the Z basis")
    if len(noise_rates) != raw_observables_shape[0]:
        raise ValueError("The length of noise_rates must match the first dimension of job_samples")

    # Measure observables
    if verbose: print("\rMeasuring diagonal observables...".ljust(100), end="")
    raw_observables, raw_observables_confidence = measure.many_measure_diagonal_observables(job_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence)
    raw_observables = raw_observables.reshape(*raw_observables_shape)
    raw_observables_confidence = raw_observables_confidence.reshape(*raw_observables_shape, 2)

    # ZNE extrapolation
    observables_for_reg = raw_observables.T.reshape(-1, noise_rates.shape[0])
    zne_results = [zne_func(noise_rates, obs) for obs in observables_for_reg]
    zne_observables = np.array([res.intercept for res in zne_results]).reshape(raw_observables_shape[1::][::-1]).T
    zne_observables_condelta = np.array([res.std_err for res in zne_results]).reshape(raw_observables_shape[1::][::-1]).T
    if verbose: print(f"\rLoaded {len(job_samples.flatten())} jobs".ljust(100))
    to_return = [[zne_observables, raw_observables]]
    if confidence is not None:
        to_return.append([zne_observables_condelta, raw_observables_confidence])
    if return_samples:
        to_return.append(job_samples)
    return to_return if len(to_return) > 1 else to_return[0]

def postselection_zne(job_samples, job_data, postselection_genfunc, observables_genfuncs, 
                      flip_threshold=None, flipped_syndromes=None, minimum_shots=None,
                      zne_func=znemod.perform_linear_zne,
                      confidence=None,
                      return_samples=False, return_postselected_samples=False, 
                      verbose=False):
    try:
        noise_rates = np.array(job_data["noise_rates"])
    except KeyError:
        raise WorkflowError("job_data must contain the 'noise_rates' key")
    if not isinstance(observables_genfuncs, (tuple, list)):
        observables_genfuncs = [observables_genfuncs]
    raw_observables_shape = (*job_data["job_circuits_shape"][::-1], len(observables_genfuncs))
    # DEBUG
    # raw_observables_shape = (*job_data["job_circuits_shape"], len(observables_genfuncs))
    job_samples = np.array(job_samples, dtype=object).T.flatten()
    nqubits = len(list(job_samples[0].keys())[0])
    postselection_operators, diagonal_observbles_ops, non_diagonal_observables_ops, basis = ps.initialize_postselection(nqubits, postselection_genfunc, observables_genfuncs)
    if non_diagonal_observables_ops:
        raise ValueError("The provided observables are not diagonal in the Z basis")
    if len(noise_rates) != raw_observables_shape[0]:
        raise ValueError("The length of noise_rates must match the first dimension of job_samples")


    if verbose: print("\rGenerating postselected samples...".ljust(100), end="")
    postselected_samples = ps.get_recovered_samples(job_samples, postselection_operators, 
                                                    flip_threshold=flip_threshold, flipped_syndromes=flipped_syndromes, 
                                                    minimum_shots=minimum_shots)

    # Measure observables
    if verbose: print("\rMeasuring diagonal observables...".ljust(100), end="")
    raw_observables, raw_observables_confidence = measure.many_measure_diagonal_observables(job_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence)
    raw_observables = raw_observables.reshape(*raw_observables_shape)
    raw_observables_confidence = raw_observables_confidence.reshape(*raw_observables_shape, 2)

    postselected_observables, postselected_observables_confidence = measure.many_measure_diagonal_observables(postselected_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence)
    postselected_observables = postselected_observables.reshape(*raw_observables_shape)
    postselected_observables_confidence = postselected_observables_confidence.reshape(*raw_observables_shape, 2)

    # ZNE extrapolation
    # observables_for_reg = postselected_observables.reshape(-1, noise_rates.shape[0])
    # zne_results = [zne_func(noise_rates, obs) for obs in observables_for_reg]
    # zne_observables = np.array([res.intercept for res in zne_results]).reshape(raw_observables_shape[1::])
    # zne_observables_condelta = np.array([res.std_err for res in zne_results]).reshape(raw_observables_shape[1::])
    if verbose: print("\rPerforming ZNE extrapolation...".ljust(100), end="")
    zne_observables = np.full(raw_observables_shape[1::], np.nan)
    zne_observables_confidence = np.full((*raw_observables_shape[1::], 2), np.nan)
    for ti in range(raw_observables_shape[1]):
        for oind in range(raw_observables_shape[2]):
            observables_for_reg = postselected_observables[:, ti, oind]
            zne_results = zne_func(noise_rates, observables_for_reg)
            zne_observables[ti, oind] = zne_results.intercept
            zne_observables_confidence[ti, oind, 0] = zne_results.intercept - zne_results.std_err
            zne_observables_confidence[ti, oind, 1] = zne_results.intercept + zne_results.std_err
    
    if verbose: print(f"\rLoaded {len(job_samples.flatten())} jobs".ljust(100))
    to_return = [[zne_observables, postselected_observables, raw_observables]]
    if confidence is not None:
        to_return.append([zne_observables_confidence, postselected_observables_confidence, raw_observables_confidence])
    if return_samples:
        to_return.append([postselected_samples, job_samples])
    return to_return if len(to_return) > 1 else to_return[0]

def peazne(executor_results, job_data, observables_genfuncs,
           zne_func=znemod.perform_linear_zne,
           confidence=None,
           return_samples=False,
           verbose=False):
    if confidence is not None:
        raise NotImplementedError("Confidence intervals are not yet implemented for peazne")

    try:
        noise_factors = np.array(job_data["noise_factors"])
    except KeyError:
        raise WorkflowError("job_data must contain the 'noise_factors' key")

    if not isinstance(observables_genfuncs, (tuple, list)):
        observables_genfuncs = [observables_genfuncs]
    raw_observables_shape = (len(noise_factors), *job_data["job_circuits_shape"], len(observables_genfuncs))
    job_results = np.array(executor_results, dtype=object).flatten()
    succesful_jobs_mask = np.array([len(res.keys()) > 0 for res in job_results])
    nqubits = list(job_results[0].values())[0].shape[-1]
    diagonal_observbles_ops, non_diagonal_observables_ops = measure.generate_observables(nqubits, observables_genfuncs)
    if non_diagonal_observables_ops:
        raise ValueError("The provided observables are not diagonal in the Z basis")

    # Measure observables
    # TODO: Implement confidence
    # TODO: Properly address failed jobs (compute the expected size in job_db)
    if verbose: print("\rMeasuring diagonal observables...".ljust(100), end="")
    raw_measured_observables = measure.many_measure_observables_executor(job_results[succesful_jobs_mask], diagonal_observbles_ops)
    raw_measured_observables = np.swapaxes(raw_measured_observables, 0, 1)
    raw_observables = np.full(raw_observables_shape, np.nan)
    raw_observables[:, succesful_jobs_mask, :] = raw_measured_observables

    # ZNE extrapolation
    if verbose: print("\rPerforming ZNE extrapolation...".ljust(100), end="")
    zne_observables = np.full(raw_observables_shape[1::], np.nan)
    for inds in np.ndindex(raw_observables.shape[1::]):
        observables_for_reg = raw_observables[:, *inds]
        if np.isnan(observables_for_reg).any():
            zne_observables[inds] = np.nan
        else:
            zne_results = zne_func(noise_factors, observables_for_reg)
            zne_observables[inds] = zne_results.intercept
    
    if verbose: print(f"\rLoaded {len(job_results.flatten())} jobs".ljust(100))
    to_return = [[zne_observables, raw_observables]]
    if return_samples:
        to_return.append(job_results)
    return to_return if len(to_return) > 1 else to_return[0]

def sum_observables_terms(observables_expectation, observables_coupling, observables_confidence=None):
    observables_coupling = miscutils.to_padded_array(observables_coupling, fill_value=np.nan, dtype=float)
    if observables_expectation.shape[::2] != observables_coupling.shape:
        raise ValueError("observables_expectation and observables_coupling must have the same shape")
    observables_expectation_times_coupling = observables_expectation * observables_coupling[:, None, :]
    summed_observables = np.nansum(observables_expectation_times_coupling, axis=2)
    summed_observables = np.nansum(summed_observables, axis=0)
    if observables_confidence is not None:
        left_err = observables_expectation - observables_confidence[:, :, :, 0]
        right_err = observables_expectation + observables_confidence[:, :, :, 1]
        left_err_times_coupling = np.abs(left_err * observables_coupling[:, None, :])**2
        right_err_times_coupling = np.abs(right_err * observables_coupling[:, None, :])**2
        summed_left_err = np.nansum(left_err_times_coupling, axis=2)
        summed_left_err = np.sqrt(np.nansum(summed_left_err, axis=0))
        summed_right_err = np.nansum(right_err_times_coupling, axis=2)
        summed_right_err = np.sqrt(np.nansum(summed_right_err, axis=0))
        summed_observables_confidence = np.zeros((*summed_observables.shape, 2), dtype=complex)
        summed_observables_confidence[:, 0] = summed_observables - summed_left_err*np.exp(1j*np.angle(summed_observables))
        summed_observables_confidence[:, 1] = summed_observables + summed_right_err*np.exp(1j*np.angle(summed_observables))
        return summed_observables, summed_observables_confidence
    return summed_observables

def diag_obs_dec_sparse(job_samples, cal_samples, job_data, observables_genfuncs, confidence=None,
                        alpha=1, normatol=1e-8, max_iter=1000,
                        postselect_ops_genfunc=None, flip_threshold=None, flipped_syndromes=None, minimum_shots=None, recovery_function=ps.gauge_recovery,
                        return_dec_samples=False, return_samples=False,
                        unsafe=False, verbose=False):
    if (jshape := job_samples.shape) != cal_samples.shape:
        raise ValueError("job_samples and cal_samples must have the same shape")
    if len(jshape) == 2:
        # Combine samples dicts from different layouts
        if verbose: print("\rCombining samples dicts from different layouts...".ljust(100), end="")
        job_samples = samples.multilayout_combine_samples_dicts(job_samples)
        cal_samples = samples.multilayout_combine_samples_dicts(cal_samples)
    elif len(jshape) == 1:
        job_samples = np.array(job_samples, dtype=object)
        cal_samples = np.array(cal_samples, dtype=object)
    elif len(job_samples.shape) != 1:
        raise ValueError("job_samples and cal_samples must be at most a matrix of samples dicts")
    if isinstance(alpha, (int, float)):
        alpha = [alpha] * len(job_samples)
    alpha = jlconvert(jl.Vector[jl.Float64], alpha)
    nqubits = len(list(job_samples.flatten()[0].keys())[0])
    diagonal_observbles_ops, non_diagonal_observables_ops = measure.generate_observables(nqubits, observables_genfuncs)
    if len(non_diagonal_observables_ops) > 0:
        raise ValueError("The provided observables are not diagonal in the Z basis")
    if verbose: print("\rPerforming DEC mitigation...".ljust(100), end="")
    shots = sum(job_samples[0].values())
    jl.seval("include(\"./utils/DEC.jl\")")
    jl.seval("using .DEC")
    dec = jl.DEC
    jl_job_samples = jlconvert(jl.Vector[jl.Dict[jl.String, jl.Int]], job_samples)
    jl_cal_samples = jlconvert(jl.Vector[jl.Dict[jl.String, jl.Int]], cal_samples)
    quasi_dicts = dec.quasiprobability_dict(jl_job_samples, jl_cal_samples, alpha=alpha, normatol=normatol, max_iter=max_iter, unsafe=unsafe)
    prob_dicts = QuasiDistribution(quasi_dicts).nearest_probability_distribution().binary_probabilities_dicts(nqubits)
    prob_dicts = [jlconvert(d) for d in prob_dicts]
    if postselect_ops_genfunc is not None:
        postselection_ops, diag_observables_ops, nondiag_observables_ops, basis = ps.initialize_postselection(nqubits, postselect_ops_genfunc, observables_genfuncs)
        recovered_dec_samples = ps.get_recovered_samples(job_samples, postselection_ops,
                                                         flip_threshold=flip_threshold,
                                                         flipped_syndromes=flipped_syndromes,
                                                         minimum_shots=minimum_shots,
                                                         recovery_function=recovery_function)
        final_observables, final_observables_confidence = measure.many_measure_diagonal_observables(recovered_dec_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence, shots=shots)
        observables = [final_observables]
        if confidence is not None:
            observables_confidence = [final_observables_confidence]
        if return_dec_samples: dec_samples_dicts = [recovered_dec_samples, prob_dicts]
    else:
        observables = []
        if confidence is not None:
            observables_confidence = []
        if return_dec_samples: dec_samples_dicts = [prob_dicts]
    dec_observables, dec_observables_confidence = measure.many_measure_diagonal_observables(prob_dicts, diagonal_observbles_ops, confidence=0 if confidence is None else confidence, shots=shots)
    raw_observables, raw_observables_confidence = measure.many_measure_diagonal_observables(job_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence, shots=shots)
    observables += [dec_observables, raw_observables]
    if confidence is not None:
        observables_confidence += [dec_observables_confidence, raw_observables_confidence]
    if verbose: print(f"\rLoaded {len(job_samples.flatten()) + len(cal_samples.flatten())} jobs".ljust(100))
    to_return = [observables]
    if confidence is not None:
        to_return.append(observables_confidence)
    if return_dec_samples:
        to_return.append(dec_samples_dicts)
    if return_samples:
        to_return.append([job_samples, cal_samples])
    return to_return if len(to_return) > 1 else to_return[0]

def diag_obs_dec_dense(job_samples, cal_samples, job_data, observables_genfuncs, confidence=None,
                       cal_basis_state_ints=0,
                       postselect_ops_genfunc=None, flip_threshold=None, flipped_syndromes=None, minimum_shots=None, recovery_function=ps.gauge_recovery,
                       return_dec_samples=False, return_samples=False,
                       verbose=False):
    if (jshape := job_samples.shape) != cal_samples.shape:
        raise ValueError("job_samples and cal_samples must have the same shape")
    if len(jshape) == 2:
        # Combine samples dicts from different layouts
        if verbose: print("\rCombining samples dicts from different layouts...".ljust(100), end="")
        job_samples = samples.multilayout_combine_samples_dicts(job_samples)
        cal_samples = samples.multilayout_combine_samples_dicts(cal_samples)
    elif len(jshape) == 1:
        job_samples = np.array(job_samples, dtype=object)
        cal_samples = np.array(cal_samples, dtype=object)
    elif len(job_samples.shape) != 1:
        raise ValueError("job_samples and cal_samples must be at most a matrix of samples dicts")
    if isinstance(cal_basis_state_ints, int):
        cal_basis_state_ints = [cal_basis_state_ints] * cal_samples.shape[0]
    else:
        if len(cal_basis_state_ints) != cal_samples.shape[0]:
            raise ValueError("cal_basis_state_ints must be an int or a list of ints with length equal to the number of calibration samples dicts")
    nqubits = len(list(job_samples.flatten()[0].keys())[0])
    diagonal_observbles_ops, non_diagonal_observables_ops = measure.generate_observables(nqubits, observables_genfuncs)
    if len(non_diagonal_observables_ops) > 0:
        raise ValueError("The provided observables are not diagonal in the Z basis")
    if verbose: print("\rPerforming DEC mitigation...".ljust(100), end="")
    shots = sum(job_samples[0].values())
    quasi_dicts = [dec.quasiprobability_dict_dense(js, cs, cal_basis_state_ints[i]) for i, (js, cs) in enumerate(zip(job_samples, cal_samples))]
    prob_dicts = [QuasiDistribution(qd).nearest_probability_distribution().binary_probabilities(nqubits) for qd in quasi_dicts]
    if postselect_ops_genfunc is not None:
        postselection_ops, diag_observables_ops, nondiag_observables_ops, basis = ps.initialize_postselection(nqubits, postselect_ops_genfunc, observables_genfuncs)
        recovered_dec_samples = ps.get_recovered_samples(job_samples, postselection_ops,
                                                         flip_threshold=flip_threshold,
                                                         flipped_syndromes=flipped_syndromes,
                                                         minimum_shots=minimum_shots,
                                                         recovery_function=recovery_function)
        final_observables, final_observables_confidence = measure.many_measure_diagonal_observables(recovered_dec_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence, shots=shots)
        observables = [final_observables]
        if confidence is not None:
            observables_confidence = [final_observables_confidence]
        if return_dec_samples: dec_samples_dicts = [recovered_dec_samples, prob_dicts]
    else:
        observables = []
        if confidence is not None:
            observables_confidence = []
        if return_dec_samples: dec_samples_dicts = [prob_dicts]
    measure_shots = [sum(sd.values()) for sd in job_samples]
    dec_observables, dec_observables_confidence = measure.many_measure_diagonal_observables(prob_dicts, diagonal_observbles_ops, confidence=0 if confidence is None else confidence, shots=measure_shots)
    raw_observables, raw_observables_confidence = measure.many_measure_diagonal_observables(job_samples, diagonal_observbles_ops, confidence=0 if confidence is None else confidence, shots=measure_shots)
    observables += [dec_observables, raw_observables]
    if confidence is not None:
        observables_confidence += [dec_observables_confidence, raw_observables_confidence]
    if verbose: print(f"\rLoaded {len(job_samples.flatten()) + len(cal_samples.flatten())} jobs".ljust(100))
    to_return = [observables]
    if confidence is not None:
        to_return.append(observables_confidence)
    if return_dec_samples:
        to_return.append(dec_samples_dicts)
    if return_samples:
        to_return.append([job_samples, cal_samples])
    return to_return if len(to_return) > 1 else to_return[0]

def load_from_db_and_curate(job_db, batch_args, ibmq_service, curate_func, *,
                            job_index=0, jobs_result_folder=None,
                            curate_func_args=(), curate_func_kwargs={},
                            curated_data_filepath="",
                            return_data=False,
                            overwrite=False, verbose=False):

    if overwrite or not os.path.exists(curated_data_filepath):
        r = job_db.load_final_samples_from_local_or_ibmq(
            batch_args, ibmq_service, jobs_result_folder, job_index, return_data=True, verbose=verbose
        )
        if len(r) == 2:
            samples, job_data = r
            cal_samples = None
        elif len(r) == 3:
            samples, cal_samples, job_data = r
        curate_func_kwargs["verbose"] = verbose
        if cal_samples is not None:
            # job_samples, cal_samples = samples
            curated_data = curate_func(
                samples, cal_samples, job_data,
                *curate_func_args, **curate_func_kwargs,
            )
        else:
            curated_data = curate_func(
                samples, job_data,
                *curate_func_args, **curate_func_kwargs,
            )
        if curated_data_filepath is not None:
            with open(curated_data_filepath, "wb") as f:
                pickle.dump(curated_data, f)
            if verbose: print(f"Saved curated data.")
    else:
        with open(curated_data_filepath, "rb") as f:
            curated_data = pickle.load(f)
        if verbose: print(f"Loaded curated data.")

    if return_data:
        return curated_data, job_data
    else:
        return curated_data