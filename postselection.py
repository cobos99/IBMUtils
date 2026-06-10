from qiskit_addon_utils.noise_management.post_selection import PostSelector
from hardware.utils.samples import threadsafe_samples_transfer
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

def gauge_recovery_and_measure_executor(executor_result, circuit, backend, postselection_ops, diagonal_observables, trex_scale_factors=None, postselection_strategy="edge", basis="Z", flip_threshold=None, flipped_syndromes=None, minimum_shots=None, register_name="meas", verbose=True, label=None, noise_factor_labels=None):

    if trex_scale_factors is None:
        trex_scale_factors = np.ones(len(diagonal_observables))

    raw_register = executor_result[register_name]
    correction_flips = executor_result[f"measurement_flips.{register_name}"]

    # Non-markovian postselection
    post_selector = PostSelector.from_circuit(circuit, coupling_map=backend.coupling_map)
    non_markovian_mask = post_selector.compute_mask(executor_result, strategy=postselection_strategy)

    if verbose:
        measure.print_postselection_retention(
            non_markovian_mask, label=label, noise_factor_labels=noise_factor_labels,
            header_suffix="non-Markovian post-selection retention",
        )

    randomization_corrected_samples = (raw_register ^ correction_flips).reshape((-1, raw_register.shape[-1]))
    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])

    if flipped_syndromes is not None:
        if len(flipped_syndromes) != postselection_mask.shape[0]:
            raise ValueError("flipped syndromes must be the same length as postselection_ops")

    decoder = Matching(postselection_mask)
    string_syndromes = (randomization_corrected_samples @ postselection_mask.T) % 2
    if flipped_syndromes is not None:
        string_syndromes = (string_syndromes + flipped_syndromes) % 2
    predicted_flips = decoder.decode_batch(string_syndromes)
    npredicted_flips = predicted_flips.sum(axis=1).reshape(raw_register.shape[:-1])
    postselection_recovered_samples = (predicted_flips + randomization_corrected_samples) % 2

    observables_mask = np.array([[int(str(opel) == basis) for opel in measure.invert_pauli_order(observable)] for observable in diagonal_observables])
    observables_phases = np.array([1j**obs.phase for obs in diagonal_observables])
    if np.any(observables_phases.imag != 0):
        raise ValueError("Diagonal observables must have +-1 phases")
    observables_phases = observables_phases.real
    strings_expectation_values = (postselection_recovered_samples @ observables_mask.T) % 2
    strings_expectation_values = (1 - 2*strings_expectation_values) * non_markovian_mask.reshape(-1, 1) * observables_phases
    strings_expectation_values = strings_expectation_values.reshape(*raw_register.shape[:-1], len(diagonal_observables))

    # Filter by gauge-recovery flip count and accumulate per noise factor.
    # raw_register.shape = (R, F, S, Q); we keep F as the noise-factor axis -> output (F, nobs).
    unique_nflips = np.unique(npredicted_flips)
    expectation_values = np.zeros((*raw_register.shape[1:-2], len(diagonal_observables)))
    gauge_num_shots_kept = np.zeros(raw_register.shape[1:-2])
    # Track which shots survived the gauge-flip cutoff (combined with NM mask) for verbose reporting
    gauge_kept_mask = np.zeros(raw_register.shape[:-1], dtype=bool)
    for unflips in unique_nflips:
        if (flip_threshold is not None) and (unflips > flip_threshold):
            if (minimum_shots is not None) and (gauge_num_shots_kept.min() > minimum_shots):
                break
            elif minimum_shots is None:
                break
        unflips_mask = np.equal(npredicted_flips, unflips)
        # A shot contributes if it survived non-Markovian post-selection AND has this flip count.
        kept_this_nflips = non_markovian_mask & unflips_mask
        gauge_kept_mask |= kept_this_nflips
        gauge_num_shots_kept += np.sum(kept_this_nflips, axis=(0, -1))
        contributing = strings_expectation_values * unflips_mask[..., np.newaxis]
        expectation_values += np.sum(contributing, axis=(0, -2))

    if verbose:
        # GSC correction rate: of NM-surviving shots, fraction where MWPM flipped at least one bit
        corrected_mask = (npredicted_flips > 0) & non_markovian_mask
        measure.print_postselection_retention(
            corrected_mask, label=label, noise_factor_labels=noise_factor_labels,
            header_suffix="GSC correction rate (NM-survivors needing >=1 flip)",
        )
        # Final yield: shots kept after non-Markovian PS *and* gauge-flip cutoff
        measure.print_postselection_retention(
            gauge_kept_mask, label=label, noise_factor_labels=noise_factor_labels,
            header_suffix="final retention (NM + GSC flip cutoff)",
        )

    # NaN-safe divide for noise factors with zero kept shots
    safe_kept = np.where(gauge_num_shots_kept > 0, gauge_num_shots_kept, 1)
    expectation_values = expectation_values / safe_kept[:, np.newaxis] * trex_scale_factors
    expectation_values = np.where(gauge_num_shots_kept[:, np.newaxis] > 0, expectation_values, np.nan)

    return expectation_values


def gauge_recovery_executor(executor_result, postselection_ops,  flip_threshold=None, minimum_shots=None, flipped_syndromes=None, recovery_function=None, basis="Z", register_name="meas"):
    
    if flipped_syndromes is not None:
        if len(flipped_syndromes) != postselection_mask.shape[0]:
            raise ValueError("flipped syndromes must be the same length as postselection_ops")
    
    raw_register = executor_result[register_name]
    correction_flips = executor_result[f"measurement_flips.{register_name}"]
    print('Raw register')
    
    randomization_corrected_samples = (raw_register ^ correction_flips).reshape((-1, raw_register.shape[-1]))
    postselection_mask = np.array([[int(str(opel) == basis) for opel in operator] for operator in postselection_ops])
    decoder = Matching(postselection_mask)
    string_syndromes = (randomization_corrected_samples @ postselection_mask.T) % 2
    if flipped_syndromes is not None:
        string_syndromes = (string_syndromes + flipped_syndromes) % 2
    predicted_flips = decoder.decode_batch(string_syndromes)
    npredicted_flips = predicted_flips.sum(axis=1).reshape(raw_register.shape[:-1])
    postselection_recovered_samples = (predicted_flips + randomization_corrected_samples) % 2

    return postselection_recovered_samples


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