from hardware.utils.stateprep import GHZBlockChainStatePrep
from hardware.utils.workflows import WorkflowError
from qiskit.quantum_info import Pauli, PauliList
import hardware.utils.grouper as grouper
import hardware.utils.measure as measure
import hardware.utils.misc as miscutils
import hardware.utils.odr as odr
import numpy as np

def times_matrix(krylov_dt, krylov_dim):
    times_row_col = np.linspace(0, krylov_dt*(krylov_dim-1), krylov_dim)
    times_matrix = (times_row_col[:, None] - times_row_col[None, :]).T
    return times_matrix

def observables_to_matrix(observables, constant=1, sign_lower_triangular=1):
    observables = np.array(observables)
    if len(observables.shape) != 1:
        raise ValueError("observables must be a 1D array-like")
    matrix = np.zeros((len(observables), len(observables)), dtype=complex)
    np.fill_diagonal(matrix, constant * observables[0])
    matrix[0, :] = constant * observables
    for i in range(1, len(observables)):
        matrix[i, :i] = np.conj(matrix[0, 1:i+1])[::-1]
        matrix[i, i+1:] = (matrix[0, 1:len(observables)-i])
    return matrix

def fix_T_first_grouping(groupings, T_op):
    readout_circuits = grouper.generate_readout_circuits(groupings, len(T_op))
    observables_per_circuit = []
    for i, group in enumerate(groupings):
        if len(group["operators"]) > 0:
            observables_per_circuit.append(group["operators"])
        else:
            del readout_circuits[i]
    T_group_index = observables_per_circuit.index([str(T_op)])
    if T_group_index != 0:
        readout_circuits[0], readout_circuits[T_group_index] = readout_circuits[T_group_index], readout_circuits[0]
        observables_per_circuit[0], observables_per_circuit[T_group_index] = observables_per_circuit[T_group_index], observables_per_circuit[0]
    return readout_circuits, observables_per_circuit

def find_operator_in_groupings(groupings, operator):
    groupings = miscutils.to_padded_array(groupings, dtype=Pauli, extra_iterable_types=(PauliList,))
    for i, group in enumerate(groupings):
        for j, op in enumerate(group):
            if op is not np.nan:
                if measure.pauli_equal_uptophase(op, operator):
                    return i, j
    raise ValueError("Operator not found in groupings")

class XYChainStabilizerState(GHZBlockChainStatePrep):
    def __init__(self, operator, block_number, roll=0):
        nqubits = len(operator)
        super().__init__(nqubits, block_number, roll=roll)
        for i, opel in enumerate(operator):
            if str(opel) == 'Z':
                self.h(i)
            if str(opel) == 'Y':
                self.s(i)

def workflow(job_samples, cal_samples, job_data, confidence=None,
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
        sim_dobservables_T = job_data['sim_dobservables_T']
        sim_dobservables_sb = job_data['sim_dobservables_sb']
        sim_dobservables_int = job_data['sim_dobservables_int']
    except KeyError as e:
        raise KeyError(f"Missing required job_data key: {e}")
    sim_dobservables_T = [[Pauli(op) for op in group] for group in job_data['sim_dobservables_T']]
    sim_dobservables_sb = [[Pauli(op) for op in group] for group in job_data['sim_dobservables_sb']]
    sim_dobservables_int = [[Pauli(op) for op in group] for group in job_data['sim_dobservables_int']]
    sim_dobservables = sim_dobservables_T + sim_dobservables_sb + sim_dobservables_int
    if any(not measure.is_diagonal(op) for op in miscutils.iterate_nested_list(sim_dobservables)):
        raise WorkflowError("The observables in 'sim_dobservables' are not all diagonal in the Z basis")
    if any(len(op) != nqubits for op in miscutils.iterate_nested_list(sim_dobservables)):
        raise WorkflowError("The observables in 'sim_dobservables' do not all match the number of qubits in the samples")
    if len(sim_dobservables) != jobs_shape[0]:
        raise WorkflowError("'sim_dobservables' length does not match the rows of job samples")
    sim_dobservables_T = miscutils.to_padded_array(sim_dobservables_T, fill_value=np.nan, dtype=object)
    sim_dobservables_sb = miscutils.to_padded_array(sim_dobservables_sb, fill_value=np.nan, dtype=object)
    sim_dobservables_int = miscutils.to_padded_array(sim_dobservables_int, fill_value=np.nan, dtype=object)
    sim_dobservables = miscutils.to_padded_array(sim_dobservables, fill_value=np.nan, dtype=object)

    try:
        cal_dobservables_T = job_data['cal_dobservables_T']
        cal_dobservables_sb = job_data['cal_dobservables_sb']
        cal_dobservables_int = job_data['cal_dobservables_int']
    except KeyError as e:
        raise KeyError(f"Missing required job_data key: {e}")
    cal_dobservables_T = [[Pauli(op) for op in group] for group in job_data['cal_dobservables_T']]
    cal_observables_T_weights = [[measure.pauli_weight(op) for op in group] for group in cal_dobservables_T]
    cal_dobservables_sb = [[Pauli(op) for op in group] for group in job_data['cal_dobservables_sb']]
    cal_observables_sb_weights = [[measure.pauli_weight(op) for op in group] for group in cal_dobservables_sb]
    cal_dobservables_int = [[Pauli(op) for op in group] for group in job_data['cal_dobservables_int']]
    cal_observables_int_weights = [[measure.pauli_weight(op) for op in group] for group in cal_dobservables_int]
    cal_dobservables = cal_dobservables_T + cal_dobservables_sb + cal_dobservables_int
    if any(not measure.is_diagonal(op) for op in miscutils.iterate_nested_list(cal_dobservables)):
        raise WorkflowError("The observables in 'cal_dobservables' are not all diagonal in the Z basis")
    if any(len(op) != nqubits for op in miscutils.iterate_nested_list(cal_dobservables)):
        raise WorkflowError("The observables in 'cal_dobservables' do not all match the number of qubits in the samples")
    if len(cal_dobservables) != cal_shape[0]:
        raise WorkflowError("'cal_dobservables' length does not match the rows of job samples")
    cal_dobservables_T = miscutils.to_padded_array(cal_dobservables_T, fill_value=np.nan, dtype=object)
    cal_dobservables_sb = miscutils.to_padded_array(cal_dobservables_sb, fill_value=np.nan, dtype=object)
    cal_dobservables_int = miscutils.to_padded_array(cal_dobservables_int, fill_value=np.nan, dtype=object)
    cal_dobservables = miscutils.to_padded_array(cal_dobservables, fill_value=np.nan, dtype=object)

    try:
        odr_expected_observables_T = job_data["odr_expected_observables_T"]
        odr_expected_observables_sb = job_data["odr_expected_observables_sb"]
        odr_expected_observables_int = job_data["odr_expected_observables_int"]
    except KeyError:
        raise KeyError(f"Missing required job_data key: {e}")
    odr_expected_observables = odr_expected_observables_T + odr_expected_observables_sb + odr_expected_observables_int
    odr_expected_observables = miscutils.to_padded_array(odr_expected_observables, fill_value=np.nan, dtype=float)
    if odr_expected_observables.shape != cal_dobservables.shape:
        raise WorkflowError("'odr_expected_observables' shape does not match 'cal_dobservables' shape")

    job_samples_T = job_samples[:1, :]
    job_samples_sb = job_samples[1:1+sim_dobservables_sb.shape[0], :]
    job_samples_int = job_samples[1+sim_dobservables_sb.shape[0]:, :]
    cal_samples_T = cal_samples[:1, :]
    cal_samples_sb = cal_samples[1:1+cal_dobservables_sb.shape[0], :]
    cal_samples_int = cal_samples[1+cal_dobservables_sb.shape[0]:, :]

    # Measure raw job observables
    if verbose: print("\rMeasuring job observables...".ljust(100), end="")
    raw_job_obsexp_T, raw_job_obsexp_conf_T = measure.many_measure_diagonal_observables(job_samples_T, sim_dobservables_T[0], confidence=0 if confidence is None else confidence)
    raw_job_obsexp_sb = np.full((*job_samples_sb.shape, sim_dobservables_sb.shape[1]), np.nan)
    raw_job_obsexp_conf_sb = np.full((*job_samples_sb.shape, sim_dobservables_sb.shape[1], 2), np.nan)
    for i in range(job_samples_sb.shape[0]):
        last_obs_ind = len([obs for obs in sim_dobservables_sb[i, :] if obs is not np.nan])
        raw_job_obsexp_sb[i, :, :last_obs_ind], raw_job_obsexp_conf_sb[i, :, :last_obs_ind, :] = measure.many_measure_diagonal_observables(job_samples_sb[i, :], sim_dobservables_sb[i, :last_obs_ind], confidence=0 if confidence is None else confidence)
    raw_job_obsexp_int = np.full((*job_samples_int.shape, sim_dobservables_int.shape[1]), np.nan)
    raw_job_obsexp_conf_int = np.full((*job_samples_int.shape, sim_dobservables_int.shape[1], 2), np.nan)
    for i in range(job_samples_int.shape[0]):
        last_obs_ind = len([obs for obs in sim_dobservables_int[i, :] if obs is not np.nan])
        raw_job_obsexp_int[i, :, :last_obs_ind], raw_job_obsexp_conf_int[i, :, :last_obs_ind, :] = measure.many_measure_diagonal_observables(job_samples_int[i, :], sim_dobservables_int[i, :last_obs_ind], confidence=0 if confidence is None else confidence)
    
    # Measure calibration observables
    if verbose: print("\rMeasuring job observables...".ljust(100), end="")
    cal_obsexp_T, cal_obsexp_conf_T = measure.many_measure_diagonal_observables(cal_samples_T, cal_dobservables_T[0], confidence=0 if confidence is None else confidence)
    p_arr_T = odr.compute_depolarizing_probability(cal_obsexp_T, odr_expected_observables_T, 0, nqubits)
    cal_obsexp_sb = np.full((*cal_samples_sb.shape, cal_dobservables_sb.shape[1]), np.nan)
    cal_obsexp_conf_sb = np.full((*cal_samples_sb.shape, cal_dobservables_sb.shape[1], 2), np.nan)
    p_arr_sb = np.full_like(cal_obsexp_sb, np.nan)
    for i in range(cal_samples_sb.shape[0]):
        last_obs_ind = len([obs for obs in cal_dobservables_sb[i, :] if obs is not np.nan])
        cal_obsexp_sb[i, :, :last_obs_ind], cal_obsexp_conf_sb[i, :, :last_obs_ind, :] = measure.many_measure_diagonal_observables(cal_samples_sb[i, :], cal_dobservables_sb[i, :last_obs_ind], confidence=0 if confidence is None else confidence)
        p_arr_sb[i, :] = odr.compute_depolarizing_probability(cal_obsexp_sb[i, :, :last_obs_ind], odr_expected_observables_sb[i][:last_obs_ind], 0, nqubits)

    cal_obsexp_int = np.full((*cal_samples_int.shape, cal_dobservables_int.shape[1]), np.nan)
    cal_obsexp_conf_int = np.full((*cal_samples_int.shape, cal_dobservables_int.shape[1], 2), np.nan)
    p_arr_int = np.full_like(cal_obsexp_int, np.nan)
    for i in range(cal_samples_int.shape[0]):
        last_obs_ind = len([obs for obs in cal_dobservables_int[i, :] if obs is not np.nan])
        cal_obsexp_int[i, :, :last_obs_ind], cal_obsexp_conf_int[i, :, :last_obs_ind, :] = measure.many_measure_diagonal_observables(cal_samples_int[i, :], cal_dobservables_int[i, :last_obs_ind], confidence=0 if confidence is None else confidence)
        p_arr_int[i, :] = odr.compute_depolarizing_probability(cal_obsexp_int[i, :, :last_obs_ind], odr_expected_observables_int[i][:last_obs_ind], 0, nqubits)
    
    # Renormalize calibration observables
    clean_obsexp_T = np.full_like(raw_job_obsexp_T, np.nan)
    clean_obsexp_conf_T = np.full_like(raw_job_obsexp_conf_T, np.nan)
    for i in range(sim_dobservables_T.shape[0]):
        for j in range(sim_dobservables_T.shape[1]):
            if sim_dobservables_T[i, j] is not np.nan:
                this_obs_weight = measure.pauli_weight(sim_dobservables_T[i, j])
                cal_weight_index = cal_observables_T_weights[i].index(this_obs_weight)
                clean_obsexp_T[i, :, j], clean_obsexp_conf_T[i, :, j, :] = odr.renormalize_observable(raw_job_obsexp_T[i, :, j], p_arr_T[i, :, cal_weight_index], 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observables_T[i][cal_weight_index], observable_confidence=raw_job_obsexp_conf_T[i, :, j, :])

    clean_obsexp_sb = np.full_like(raw_job_obsexp_sb, np.nan)
    clean_obsexp_conf_sb = np.full_like(raw_job_obsexp_conf_sb, np.nan)
    for i in range(sim_dobservables_sb.shape[0]):
        for j in range(sim_dobservables_sb.shape[1]):
            if sim_dobservables_sb[i, j] is not np.nan:
                this_obs_weight = measure.pauli_weight(sim_dobservables_sb[i, j])
                if i == len(sim_dobservables_sb)-1:
                    cal_row_ind = -1
                    cal_weight_index = cal_observables_sb_weights[cal_row_ind].index(this_obs_weight)
                else:
                    cal_row_ind = 0
                    cal_weight_index = cal_observables_sb_weights[cal_row_ind].index(this_obs_weight)
                clean_obsexp_sb[i, :, j], clean_obsexp_conf_sb[i, :, j, :] = odr.renormalize_observable(raw_job_obsexp_sb[i, :, j], p_arr_sb[cal_row_ind, :, cal_weight_index], 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observables_sb[cal_row_ind][cal_weight_index], observable_confidence=raw_job_obsexp_conf_sb[i, :, j, :])

    clean_obsexp_int = np.full_like(raw_job_obsexp_int, np.nan)
    clean_obsexp_conf_int = np.full_like(raw_job_obsexp_conf_int, np.nan)
    for i in range(sim_dobservables_int.shape[0]):
        for j in range(sim_dobservables_int.shape[1]):
            if sim_dobservables_int[i, j] is not np.nan:
                this_obs_weight = measure.pauli_weight(sim_dobservables_int[i, j])
                if i == len(sim_dobservables_int)-1:
                    cal_row_ind = -1
                    cal_weight_index = cal_observables_int_weights[cal_row_ind].index(this_obs_weight)
                else:
                    cal_row_ind = 0
                    cal_weight_index = cal_observables_int_weights[cal_row_ind].index(this_obs_weight)
                clean_obsexp_int[i, :, j], clean_obsexp_conf_int[i, :, j, :] = odr.renormalize_observable(raw_job_obsexp_int[i, :, j], p_arr_int[cal_row_ind, :, cal_weight_index], 0, nqubits, observable_range=None, expected_observable_arr=odr_expected_observables_int[cal_row_ind][cal_weight_index], observable_confidence=raw_job_obsexp_conf_int[i, :, j, :])

    # DEBUG
    # to_return = [[clean_obsexp_T], [cal_obsexp_T], [raw_job_obsexp_T]]
    # return to_return

    if verbose: print(f"\rLoaded {len(job_samples.flatten()) + len(cal_samples.flatten())} jobs".ljust(100))
    to_return = [[[clean_obsexp_T, clean_obsexp_sb, clean_obsexp_int],
                 [cal_obsexp_T, cal_obsexp_sb, cal_obsexp_int],
                 [raw_job_obsexp_T, raw_job_obsexp_sb, raw_job_obsexp_int]]]
    if confidence is not None:
        to_return.append([[clean_obsexp_conf_T, clean_obsexp_conf_sb, clean_obsexp_conf_int],
                          [cal_obsexp_conf_T, cal_obsexp_conf_sb, cal_obsexp_conf_int],
                          [raw_job_obsexp_conf_T, raw_job_obsexp_conf_sb, raw_job_obsexp_conf_int]])
    if return_samples:
        to_return.append([job_samples, cal_samples])
    return to_return if len(to_return) > 1 else to_return[0]