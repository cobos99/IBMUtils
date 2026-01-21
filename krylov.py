from hardware.utils.stateprep import GHZBlockChainStatePrep
from qiskit.quantum_info import Pauli, PauliList
import hardware.utils.grouper as grouper
import hardware.utils.measure as measure
import hardware.utils.misc as miscutils
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
    def __init__(self, operator, block_number):
        nqubits = len(operator)
        super().__init__(nqubits, block_number)
        for i, opel in enumerate(operator):
            if str(opel) == 'Z':
                self.h(i)
            if str(opel) == 'Y':
                self.s(i)