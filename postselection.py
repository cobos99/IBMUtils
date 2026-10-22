from qiskit_addon_utils.noise_management.post_selection import PostSelector
from hardware.utils.samples import threadsafe_samples_transfer
from hardware.utils.bootstrap import bootstrap_sample_dict_mean
from multiprocessing.shared_memory import SharedMemory
from qiskit.quantum_info import Pauli, PauliList
import hardware.utils.measure as measure
from pymatching import Matching
import multiprocessing as mp
import numpy as np

def check_postselection_observable_commutation(observables, postselection_ops):
    postselection_ops = PauliList([Pauli(postselection_op) for postselection_op in postselection_ops])
    observables = PauliList([Pauli(observable) for observable in observables])
    nqubits = len(postselection_ops[0])
    for postselection_op in postselection_ops:
        this_nqubits = len(postselection_op)
        if this_nqubits != nqubits:
            raise ValueError("All operators must act on the same number of qubits")
    for observable in observables:
        this_nqubits = len(observable)
        if this_nqubits != nqubits:
            raise ValueError("All operators must act on the same number of qubits")
    return len(postselection_ops.commutes_with_all(observables)) == len(postselection_ops)

def initialize_postselection(nqubits, postselects_generating_func, observable_generating_funcs):
    if type(observable_generating_funcs) != list:
        observable_generating_funcs = [observable_generating_funcs]

    postselection_ops = PauliList(set(postselects_generating_func(nqubits)))
    post_selection_diagonal, basis = measure.diagonal_operators_check(postselection_ops)
    if not post_selection_diagonal:
        raise NotImplementedError("Only supports diagonal postselection operators in some basis")
    diagonal_observables, non_diagonal_observables = measure.generate_observables(nqubits, observable_generating_funcs)
    return postselection_ops, diagonal_observables, non_diagonal_observables, basis

def gauge_recovery(samples_dict, postselection_ops, basis="Z", flip_threshold=None, flipped_syndromes=None, minimum_shots=None):
    if not isinstance(samples_dict, dict):
        raise ValueError("samples_dict must be a dictionary")
    if len(samples_dict) == 0:
        return {}
    strings_arr = np.array([[int(c) for c in string] for string in samples_dict.keys()])
    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    if flipped_syndromes is not None:
        if len(flipped_syndromes) != postselection_mask.shape[0]:
            raise ValueError("flipped syndromes must be the same length as postselection_ops")
    decoder = Matching(postselection_mask)
    string_syndromes = (strings_arr @ postselection_mask.T) % 2
    if flipped_syndromes is not None:
        string_syndromes = (string_syndromes + flipped_syndromes) % 2
    predicted_flips = decoder.decode_batch(string_syndromes)
    # predicted_flips = decode_with_optimized_weights(string_syndromes, postselection_mask)
    npredicted_flips = predicted_flips.sum(axis=1)
    recovered_strings = (strings_arr + predicted_flips) % 2
    nflip_sort_mask = np.argsort(npredicted_flips)
    npredicted_flips = npredicted_flips[nflip_sort_mask]
    sampled_states = np.array(list(samples_dict.keys()))[nflip_sort_mask]
    recovered_strings = recovered_strings[nflip_sort_mask]
    recovered_samples_dict = {}
    fmt_str = "%i"*recovered_strings.shape[1]
    added_shots = 0
    for i, string in enumerate(sampled_states):
        recovered_string = fmt_str % tuple(recovered_strings[i])
        shots_to_add = samples_dict[string]
        if recovered_samples_dict.get(recovered_string, None) is not None:
            recovered_samples_dict[recovered_string] = recovered_samples_dict[recovered_string] + shots_to_add
        else:
            recovered_samples_dict[recovered_string] = samples_dict[string]
        added_shots += shots_to_add
        if (flip_threshold is not None) and (npredicted_flips[(i + 1) % len(npredicted_flips)] > flip_threshold):
            if minimum_shots is not None:
                if added_shots > minimum_shots:
                    break
            else:
                break
            
    return recovered_samples_dict

# def gauge_recovery_and_measure_executor(executor_result, circuit, backend, postselection_ops, diagonal_observables, trex_scale_factors=None, postselection_strategy="edge", basis="Z", flip_threshold=None, flipped_syndromes=None, minimum_shots=None, register_name="meas"):
#     if flipped_syndromes is not None:
#         if len(flipped_syndromes) != postselection_mask.shape[0]:
#             raise ValueError("flipped syndromes must be the same length as postselection_ops")
    
#     raw_register = executor_result[register_name]
#     correction_flips = executor_result[f"measurement_flips.{register_name}"]

#     # Non-markovian postselection
#     post_selector = PostSelector.from_circuit(circuit, coupling_map=backend.coupling_map)
#     non_markovian_mask = post_selector.compute_mask(executor_result, strategy=postselection_strategy)
#     # num_shots_kept = np.sum(non_markovian_mask, axis=-1)
    
#     randomization_corrected_samples = (raw_register ^ correction_flips).reshape((-1, raw_register.shape[-1]))
#     postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
#     decoder = Matching(postselection_mask)
#     string_syndromes = (randomization_corrected_samples @ postselection_mask.T) % 2
#     if flipped_syndromes is not None:
#         string_syndromes = (string_syndromes + flipped_syndromes) % 2
#     predicted_flips = decoder.decode_batch(string_syndromes)
#     npredicted_flips = predicted_flips.sum(axis=1).reshape(raw_register.shape[:-1])
#     postselection_recovered_samples = (predicted_flips + randomization_corrected_samples) % 2

#     observables_mask = np.array([[int(str(opel) == basis) for opel in measure.invert_pauli_order(observable)] for observable in diagonal_observables])
#     observables_phases = np.array([1j**obs.phase for obs in diagonal_observables])
#     if np.any(observables_phases.imag != 0):
#         raise ValueError("Diagonal observables must have +-1 phases")
#     observables_phases = observables_phases.real
#     strings_expectation_values = (postselection_recovered_samples @ observables_mask.T) % 2
#     strings_expectation_values = (1 - 2*strings_expectation_values) * non_markovian_mask.reshape(-1, 1) * observables_phases
#     strings_expectation_values = strings_expectation_values.reshape(*raw_register.shape[:-1], len(diagonal_observables))

#     unique_nflips = np.unique(npredicted_flips)
#     expectation_values = np.zeros((*raw_register.shape[1:-2], raw_register.shape[-1]))
#     num_shots_kept = np.zeros(raw_register.shape[1:-2])
#     for unflips in unique_nflips:
#         if (flip_threshold is not None) and (unflips > flip_threshold):
#             if (minimum_shots is not None) and (num_shots_kept > minimum_shots):
#                 break
#         unflips_mask = np.equal(npredicted_flips, unflips)
#         this_nflips_expectations = strings_expectation_values.copy()
#         this_nflips_expectations[~unflips_mask, :] = 0
#         num_shots_kept += np.sum(np.all(~np.equal(this_nflips_expectations, 0), axis=-1), axis=(0, -1))
#         expectation_values += np.sum(this_nflips_expectations, axis=(0, -2))
#     expectation_values = expectation_values / num_shots_kept[:, np.newaxis] #* trex_scale_factors
#     return expectation_values

def gauge_recovery_and_measure_executor(executor_result, postselection_ops, diagonal_observables, non_markovian_postselector=None, non_markovian_postselection_strategy="edge", basis="Z", flip_threshold=None, flipped_syndromes=None, minimum_shots=None, register_name="meas", confidence=None):
    # gauge_recovery followed by measuring the observables, for an executor item result with all its randomizations merged.
    # The postselection operators are applied as in gauge_recovery, and the observables as in measure_observables_executor (label
    # position q acting on qubit q). The strings are kept in increasing number of predicted flips as in gauge_recovery: all the
    # strings with at most flip_threshold flips, and then more until more than minimum_shots shots are kept (strings with the same
    # number of flips are taken in bitstring order, while gauge_recovery takes them in the order of the samples dict).
    # With confidence, it also returns bootstrap confidence intervals computed as in measure_observables
    if isinstance(diagonal_observables, Pauli):
        diagonal_observables = [diagonal_observables]
    if non_markovian_postselector is not None:
        non_markovian_mask = non_markovian_postselector.compute_mask(executor_result, strategy=non_markovian_postselection_strategy)
    else:
        non_markovian_mask = None
    strings, counts = measure.executor_unique_samples(executor_result, register_name, shots_mask=non_markovian_mask)
    # Bitstrings in samples dict order (qubit 0 last), so that the decoding is the same as in gauge_recovery. In this order, the
    # postselection operators and the observables are both applied as written by iterating over their Pauli objects
    strings = strings[:, ::-1]
    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    if flipped_syndromes is not None and len(flipped_syndromes) != postselection_mask.shape[0]:
        raise ValueError("flipped syndromes must be the same length as postselection_ops")
    observables_mask = np.array([[int(str(opel) == basis) for opel in observable] for observable in diagonal_observables])
    observables_phases = np.array([1j**obs.phase for obs in diagonal_observables])
    if np.any(observables_phases.imag != 0):
        raise ValueError("Diagonal observables must have +-1 phases")
    observables_phases = observables_phases.real
    if len(counts) == 0:
        expectation_values = np.full(len(diagonal_observables), np.nan)
        return (expectation_values, np.full((len(diagonal_observables), 2), np.nan)) if confidence is not None else expectation_values

    decoder = Matching(postselection_mask)
    string_syndromes = (strings @ postselection_mask.T) % 2
    if flipped_syndromes is not None:
        string_syndromes = (string_syndromes + flipped_syndromes) % 2
    predicted_flips = decoder.decode_batch(string_syndromes)
    recovered_strings = (strings + predicted_flips) % 2
    if flip_threshold is not None:
        npredicted_flips = predicted_flips.sum(axis=1)
        order = np.argsort(npredicted_flips, kind="stable")
        # As in gauge_recovery, stop after a string followed by one with more than flip_threshold flips, once more than
        # minimum_shots shots are kept
        stop = np.roll(npredicted_flips[order], -1) > flip_threshold
        if minimum_shots is not None:
            stop &= np.cumsum(counts[order]) > minimum_shots
        nkept = np.argmax(stop) + 1 if np.any(stop) else len(order)
        recovered_strings = recovered_strings[order[:nkept]]
        counts = counts[order[:nkept]]

    strings_expectation_values = (1 - 2*((recovered_strings @ observables_mask.T) % 2)) * observables_phases
    expectation_values = counts @ strings_expectation_values / np.sum(counts)
    if confidence is not None:
        # bootstrap_sample_dict_mean only uses the counts of the samples dict
        confidence_intervals = bootstrap_sample_dict_mean(dict(enumerate(counts)), strings_expectation_values, confidence)
        return expectation_values, confidence_intervals
    return expectation_values

def get_recovered_samples(samples_dicts, postselection_ops, flip_threshold=None, minimum_shots=None, flipped_syndromes=None, recovery_function=None):
    all_diagonal, basis = measure.diagonal_operators_check(postselection_ops)
    if recovery_function is None:
        recovery_function = gauge_recovery
    if not all_diagonal:
        raise ValueError("Only supports diagonal postselection operators in some basis")
    try:
        postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    except ValueError:
        raise ValueError("All postselection operators and samples must have the same number of qubits")
    samples_dicts = np.array(samples_dicts)
    samples_dicts_shape = samples_dicts.shape
    samples_dicts = samples_dicts.flatten()
    shmem = SharedMemory(create=True, size=samples_dicts.nbytes)
    shared_samples = np.ndarray(samples_dicts.size, dtype=object, buffer=shmem.buf)
    shared_samples[:] = samples_dicts
    indices = np.arange(len(samples_dicts))
    recover_args = [(recovery_function, shmem.name, samples_dicts.shape, i, (postselection_ops, basis, flip_threshold, flipped_syndromes, minimum_shots)) for i in indices]
    with mp.get_context("fork").Pool(processes=mp.cpu_count()//2) as pool:
        recovered_samples_dicts = pool.starmap(threadsafe_samples_transfer, recover_args)
    shmem.close()
    shmem.unlink()
    return np.array(recovered_samples_dicts).reshape(samples_dicts_shape)

def many_gauge_recovery_and_measure_executor(executor_results, postselection_ops, diagonal_observables, basis="Z", flip_threshold=None, flipped_syndromes=None, minimum_shots=None, register_name="meas", confidence=None):
    # gauge_recovery_and_measure_executor in parallel for an array (or list) of executor item results, with NaN for missing results
    # (None). Returns the expectation values with shape (*executor_results.shape, nobservables), and with confidence also their
    # confidence intervals with shape (*executor_results.shape, nobservables, 2)
    all_diagonal, _ = measure.diagonal_operators_check(postselection_ops)
    all_observables_diagonal, _ = measure.diagonal_operators_check(diagonal_observables)
    if not all_diagonal or not all_observables_diagonal:
        raise ValueError("Only supports diagonal postselection operators and observables in some basis")
    if not isinstance(executor_results, np.ndarray):
        # Item results are mappings, which np.array would try to unpack
        results_list = list(executor_results)
        executor_results = np.empty(len(results_list), dtype=object)
        for i, result in enumerate(results_list):
            executor_results[i] = result
    og_results_shape = executor_results.shape
    flat_results = executor_results.flatten()
    available = [i for i, result in enumerate(flat_results) if result is not None]
    results = []
    if available:
        shared_results = np.empty(len(available), dtype=object)
        for k, i in enumerate(available):
            shared_results[k] = dict(flat_results[i])
        shmem = SharedMemory(create=True, size=shared_results.nbytes)
        shared_samples = np.ndarray(shared_results.shape, dtype=object, buffer=shmem.buf)
        shared_samples[:] = shared_results
        fargs = (postselection_ops, diagonal_observables, None, "edge", basis, flip_threshold, flipped_syndromes, minimum_shots, register_name, confidence)
        measure_args = [(gauge_recovery_and_measure_executor, shmem.name, shared_results.shape, k, fargs) for k in range(len(available))]
        with mp.get_context("fork").Pool(processes=max(mp.cpu_count()//2, 1)) as pool:
            results = pool.starmap(threadsafe_samples_transfer, measure_args)
        shmem.close()
        shmem.unlink()
    nobservables = len(diagonal_observables)
    expectation_values = np.full((len(flat_results), nobservables), np.nan)
    confidence_intervals = np.full((len(flat_results), nobservables, 2), np.nan)
    for k, i in enumerate(available):
        if confidence is not None:
            expectation_values[i], confidence_intervals[i] = results[k]
        else:
            expectation_values[i] = results[k]
    expectation_values = expectation_values.reshape((*og_results_shape, nobservables))
    if confidence is not None:
        return expectation_values, confidence_intervals.reshape((*og_results_shape, nobservables, 2))
    return expectation_values
