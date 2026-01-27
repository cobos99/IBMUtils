from multiprocessing.shared_memory import SharedMemory
from utils.bootstrap import bootstrap_sample_dict_mean
from utils.samples import threadsafe_samples_transfer
from hardware.utils.misc import iterate_nested_list
from qiskit.quantum_info import Pauli, PauliList
from qiskit.exceptions import QiskitError
from collections import Counter
import multiprocessing as mp
import numpy as np

def pauli_to_str(pauli):
    if type(pauli) == Pauli:
        return "".join([str(opel) for opel in invert_pauli_order(pauli)])
    elif type(pauli) == str:
        return pauli
    else:
        raise ValueError("pauli argument is not a Pauli object")

def pauli_weight(pauli):
    counts = Counter(str(pauli))
    return len(pauli) - counts['I']

def pauli_equal_uptophase(pauli1, pauli2):
    try:
        pauli1 = Pauli(pauli1)
        pauli2 = Pauli(pauli2)
    except QiskitError:
        raise ValueError("Both arguments must be convertible to Pauli objects")
    return bool((pauli1.x == pauli2.x).all() and (pauli1.z == pauli2.z).all())

def invert_pauli_order(pauli):
    if not isinstance(pauli, Pauli):
        pauli = invert_pauli_order(Pauli(pauli))
    return Pauli((pauli.z[::-1], pauli.x[::-1], pauli.phase))
    
# def get_commuting_groups_and_change_of_basis(operators):
#     try:
#         operators = PauliList(operators)
#     except QiskitError:
#         raise ValueError("operators must be convertible to PauliList")
#     commuting_groups = operators.group_commuting()
#     change_of_bases = []
#     for group in commuting_groups:
#         pass

def organize_observables_by_weight(obs_list):
    weight_obs = {}
    for obs in iterate_nested_list(obs_list):
        weight = pauli_weight(obs)
        if weight not in weight_obs:
            weight_obs[weight] = []
        weight_obs[weight].append(obs)
    return weight_obs

def pick_one_observable_per_weight(obs_list):
    weight_obs = organize_observables_by_weight(obs_list)
    picked_observables = []
    for weight in sorted(weight_obs.keys()):
        rand_index = np.random.choice(len(weight_obs[weight]))
        picked_observables.append(weight_obs[weight][rand_index])
    picked_observables = sorted(picked_observables, key=lambda op: pauli_weight(op))
    return picked_observables

def is_diagonal(pauli, basis="Z"):
    if not ((type(pauli) != Pauli) or (type(pauli) != str)):
        raise ValueError("This function only work for Pauli objects or strings")
    pauli_str = pauli_to_str(pauli)
    for c in pauli_str:
        if c not in ["I", basis]:
            return False
    return True

def diagonal_operators_check(operators):
    different_paulis = set("".join([pauli_to_str(op) for op in operators]))
    if len(remaining_ops := (different_paulis - {"I"})) > 1:
        return False
    basis = remaining_ops.pop()
    all_diagonal = all([is_diagonal(op, basis=basis) for op in operators])
    return all_diagonal, basis

def operators_affecting_qb_range(first, last, operators, return_operator_range=False):
    operators = PauliList(operators)
    affecting_operators = []
    first_affected_qb = last
    last_affected_qb = first
    for op in operators:
        op_str = pauli_to_str(op)
        sub_op_string = op_str[first:last+1]
        present_paulis = set(sub_op_string)
        if len(present_paulis - {"I"}) > 0:
            affecting_operators.append(op)
            if return_operator_range:
                non_identity_indices = [i for i, c in enumerate(op_str) if c != "I"]
                this_first_affected_qb = min(non_identity_indices)
                this_last_affected_qb = max(non_identity_indices)
                if this_first_affected_qb < first_affected_qb:
                    first_affected_qb = this_first_affected_qb
                if this_last_affected_qb > last_affected_qb:
                    last_affected_qb = this_last_affected_qb
    if return_operator_range:
        return PauliList(affecting_operators), (first_affected_qb, last_affected_qb)
    else:
        return PauliList(affecting_operators)
    
def count_non_diagonal_qubits(commuting_pauli_set):
    max_non_diagonal_qubits = 0
    for pauli in commuting_pauli_set:
        ndiagonal = pauli_to_str(pauli).count("I") + pauli_to_str(pauli).count("Z")
        this_non_diagonal_qubits = len(pauli) - ndiagonal
        if this_non_diagonal_qubits > max_non_diagonal_qubits:
            max_non_diagonal_qubits = this_non_diagonal_qubits
    return max_non_diagonal_qubits

def binary_pauli_representation(commuting_pauli_set):
    Sz = commuting_pauli_set.z[:, ::-1].T
    Sx = commuting_pauli_set.x[:, ::-1].T
    return np.concatenate((Sz, Sx), axis=0).astype(int)

def generate_observables(nqubits, observable_generating_funcs, diagonal_basis="Z"):
    if not isinstance(observable_generating_funcs, (tuple, list, np.ndarray)):
        observable_generating_funcs = [observable_generating_funcs]
    logical_observables = PauliList([Pauli(observable_func(nqubits)) for observable_func in observable_generating_funcs if observable_func is not np.nan])
    diagonal_observables = []
    non_diagonal_observables = []
    for observable in logical_observables:
        if is_diagonal(observable, basis=diagonal_basis):
            diagonal_observables.append(observable)
        else:
            non_diagonal_observables.append(observable)
    return diagonal_observables, non_diagonal_observables

def measure_observables(samples_dict, diagonal_observables, basis="Z", confidence=None, shots=None):
    if len(samples_dict) == 0:
        if confidence is not None:
            return np.full(len(diagonal_observables), np.nan), np.full((len(diagonal_observables), 2), np.nan)
        else: 
            return np.full(len(diagonal_observables), np.nan)
    strings_expectation_values = np.zeros((len(samples_dict), len(diagonal_observables)))
    observables_mask = np.array([[int(str(opel) == basis) for opel in observable] for observable in diagonal_observables])
    observables_phases = np.array([1j**obs.phase for obs in diagonal_observables])
    if np.any(observables_phases.imag != 0):
        raise ValueError("Diagonal observables must have +-1 phases")
    observables_phases = observables_phases.real
    for i, state_string in enumerate(samples_dict.keys()):
        state_string_rep = np.repeat(np.array([[int(c) for c in state_string]]), [len(diagonal_observables)], axis=0)
        observables_expectation_value = (-1)**(np.sum(state_string_rep*observables_mask, axis=1) % 2) * observables_phases
        strings_expectation_values[i] = observables_expectation_value
    weights = np.array(list(samples_dict.values()))/np.sum(list(samples_dict.values()))
    strings_expectation_times_weights = strings_expectation_values * weights.reshape((len(samples_dict), 1))
    expectation_values = np.sum(strings_expectation_times_weights, axis=0)
    if confidence is not None:
        if list(samples_dict.values())[0] < 1:
            if shots is None:
                raise ValueError("To boostrap a probability dictionary, shots must be provided")
            bootstrap_samples = {k:int(np.floor(v*shots)) for k, v in samples_dict.items()}
        else:
            bootstrap_samples = samples_dict
        confidence_intervals = bootstrap_sample_dict_mean(bootstrap_samples, strings_expectation_values.real, confidence)
        return expectation_values, confidence_intervals
    else:
        return expectation_values

def many_measure_diagonal_observables(samples_dicts, diagonal_observables, confidence=None, shots=None):
    nqubits = len(diagonal_observables[0])
    samples_dicts = np.array(samples_dicts)
    for observable in diagonal_observables:
        if len(observable) != nqubits:
            raise ValueError("All observables must act on the same number of qubits")
    for state_string in samples_dicts.flatten()[0].keys():
        if len(state_string) != nqubits:
            raise ValueError("The state strings and all observables must contain the same number of qubits")
    all_diagonal, basis = diagonal_operators_check(diagonal_observables)
    if not all_diagonal:
        raise ValueError("The provided observables are not diagonal in the same basis")
    og_samples_dicts_shape = samples_dicts.shape
    samples_dicts = samples_dicts.flatten()
    indices = np.arange(len(samples_dicts))
    if (confidence is not None) and (shots is not None):
        shots = np.array(shots)
        if shots.shape != og_samples_dicts_shape:
            raise ValueError("If provided, shots must have the same shape as samples_dicts")
        shots = shots.flatten()
    shmem = SharedMemory(create=True, size=samples_dicts.nbytes)
    shared_samples = np.ndarray(samples_dicts.shape, dtype=object, buffer=shmem.buf)
    shared_samples[:] = samples_dicts
    indices = np.arange(len(samples_dicts))
    measure_args = [(measure_observables, shmem.name, samples_dicts.shape, i, (diagonal_observables, basis, confidence, shots[i] if shots is not None else None)) for i in indices]
    with mp.get_context("fork").Pool(processes=mp.cpu_count()//2) as pool:
        result = pool.starmap(threadsafe_samples_transfer, measure_args)
    shmem.close()
    shmem.unlink()
    # measure_args = [(samples_dicts[i], diagonal_observables, basis, confidence, shots[i] if shots is not None else None) for i in indices]
    # with mp.get_context("fork").mp.Pool(processes=int(mp.cpu_count()/2)) as pool:
    #     result = pool.starmap(measure_observables, measure_args)
    # result = []
    # for args in measure_args:
    #     result.append(measure_observables(*args))
    if confidence is not None:
        expectation_values, confidence_intervals = np.array([result[i][0] for i in indices]), np.array([result[i][1] for i in indices])
        expectation_values = expectation_values.reshape((*og_samples_dicts_shape, len(diagonal_observables)))
        confidence_intervals = confidence_intervals.reshape((*og_samples_dicts_shape, len(diagonal_observables), 2))
        return expectation_values, confidence_intervals
    else:
        expectation_values = np.array(result).reshape((*og_samples_dicts_shape, len(diagonal_observables)))
        return expectation_values
    
# -----------------------------------------
#           BASIS CHANGE CIRCUITS
# -----------------------------------------

import hardware.utils.circs as circutils
from qiskit import QuantumCircuit
from collections import Counter
import sympy as sym

# AUXILIARY FUNCTIONS

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

def count_non_diagonal_qubits(commuting_pauli_set):
    max_non_diagonal_qubits = 0
    for pauli in commuting_pauli_set:
        ndiagonal = pauli_to_str(pauli).count("I") + pauli_to_str(pauli).count("Z")
        this_non_diagonal_qubits = len(pauli) - ndiagonal
        if this_non_diagonal_qubits > max_non_diagonal_qubits:
            max_non_diagonal_qubits = this_non_diagonal_qubits
    return max_non_diagonal_qubits

def operators_affecting_qb_range(first, last, operators, return_operator_range=False):
    operators = PauliList(operators)
    affecting_operators = []
    first_affected_qb = last
    last_affected_qb = first
    for op in operators:
        op_str = pauli_to_str(op)
        sub_op_string = op_str[first:last+1]
        present_paulis = set(sub_op_string)
        if len(present_paulis - {"I"}) > 0:
            affecting_operators.append(op)
            if return_operator_range:
                non_identity_indices = [i for i, c in enumerate(op_str) if c != "I"]
                this_first_affected_qb = min(non_identity_indices)
                this_last_affected_qb = max(non_identity_indices)
                if this_first_affected_qb < first_affected_qb:
                    first_affected_qb = this_first_affected_qb
                if this_last_affected_qb > last_affected_qb:
                    last_affected_qb = this_last_affected_qb
    if return_operator_range:
        return PauliList(affecting_operators), (first_affected_qb, last_affected_qb)
    else:
        return PauliList(affecting_operators)

def diagonalization_susceptible_qb_range(observables, postselection_ops, return_operators=False):
    if type(observables) != list:
        observables = [observables]
    operator_ranges = np.zeros((len(observables), 2), dtype=int)
    if return_operators: operators_affecting_obs_range = {}
    for i, observable in enumerate(observables):
        observable_str = pauli_to_str(Pauli(observable))
        first_obs_qb = len(observable_str)
        last_obs_qb = 0
        for j, c in enumerate(observable_str):
            if c in ["X", "Y"]:
                if j < first_obs_qb:
                    first_obs_qb = j
                elif j > last_obs_qb:
                    last_obs_qb = j
        operators_affecting_obs_range, this_operator_range = operators_affecting_qb_range(first_obs_qb, last_obs_qb, observable + postselection_ops, return_operator_range=True)
        operator_ranges[i, :] = this_operator_range
        if return_operators: operators_affecting_obs_range += set(operators_affecting_obs_range)
    operator_ranges = (np.min(operator_ranges[:, 0]), np.max(operator_ranges[:, 1]))
    if return_operators:
        return operator_ranges, PauliList(operators_affecting_obs_range)
    else:
        return operator_ranges
    
def is_diagonal(pauli, basis="Z"):
    if not ((type(pauli) != Pauli) or (type(pauli) != str)):
        raise ValueError("This function only work for Pauli objects or strings")
    pauli_str = pauli_to_str(pauli)
    for c in pauli_str:
        if c not in ["I", basis]:
            return False
    return True

def binary_pauli_representation(commuting_pauli_set):
    Sz = commuting_pauli_set.z[:, ::-1].T
    Sx = commuting_pauli_set.x[:, ::-1].T
    return np.concatenate((Sz, Sx), axis=0).astype(int)

def mod2_gaussian_elimination(homogeneous_system_matrix):
    solution = homogeneous_system_matrix.copy().astype(int)
    nrows, ncols = homogeneous_system_matrix.shape
    h, k = 0, 0
    while (h < nrows) and (k < ncols):
        i_max = np.argmax(np.abs(solution[h:nrows, k])) + h
        if solution[i_max, k] == 0:
            k += 1
        else:
            solution[[h, i_max]] = solution[[i_max, h]] # Swap rows
            for i in range(h+1, nrows):
                f = solution[i, k]/solution[h, k]
                solution[i, k] = 0
                for j in range(k+1, ncols):
                    solution[i, j] = solution[i, j] - solution[h, j]*f
            h += 1
            k += 1
    return solution % 2

class Permutation:
    def __init__(self, perm):
        self.perm = np.array(perm)

    def __mul__(self, other):
        try:
            if len(self.perm) != len(other.perm):
                ValueError("Permutations must be the same size")
        except AttributeError:
            ValueError(f"{other} is not a permutation")

        composed_perm = self.perm[other.perm[np.arange(len(self.perm))]]
        return Permutation(composed_perm)

    @classmethod
    def transposition(cls, n, i, j):
        if (not (0 <= i < n)) or (not (0 <= j < n)):
            raise ValueError(f"i, j must be in range [0, n)")
        
        perm = np.arange(n)
        perm[[i, j]] = perm[[j, i]]
        return cls(perm)
    
    @classmethod
    def identity(cls, n):
        return cls(np.arange(n))
    
    def __call__(self, i):
        if not (0 <= i < len(self.perm)):
            raise ValueError("i must be in range [0, n)")
        return self.perm[i]

    def __getitem__(self, i):
        if not (0 <= i < len(self.perm)):
            raise ValueError("i must be in range [0, n)")
        return self.perm[i]
    
    def __repr__(self):
        return f"Transposition({self.perm})"

# INSTRUCTION ABSTRACTIONS

class H:
    def __init__(self, *qb):
        self.qubits = list(set(list(qb)))
        self._largestqb = max(self.qubits)

    def combine(self, other):
        if type(other) != H:
            raise ValueError("Only H can be combined")
        self.qubits = (self.qubits - other.qubits) | (other.qubits - self.qubits)
        self._largestqb = max(self.qubits) if len(self.qubits) > 0 else None

    def __repr__(self):
        indices = ", ".join([str(qb) for qb in self.qubits])
        return f"H({indices})"
    
    def binary_representation_conjugation(self, S):
        S = S.copy()
        S_qubits = S.shape[0] // 2
        if S_qubits < self._largestqb:
            raise ValueError(f"This gate acts on qubits not present in S")
        to_exchange = np.array(list(self.qubits)) + S_qubits
        S[[to_exchange, self.qubits]] = S[[self.qubits, to_exchange]]
        return S
    
    def phases_computation(self, phases, S, return_Sp=False):
        phases = phases.copy()
        nqubits = S.shape[0] // 2
        for qubit in self.qubits:
            z_row = S[qubit, :]
            x_row = S[nqubits+qubit, :]
            phases = (phases + z_row*x_row) % 2
        if return_Sp:
            return phases, self.binary_representation_conjugation(S)
        else:
            return phases

class P:
    def __init__(self, *qb):
        self.qubits = list(set(list(qb)))
        self._largestqb = max(self.qubits)
    
    def __repr__(self):
        indices = ", ".join([str(qb) for qb in self.qubits])
        return f"P({indices})"
    
    def binary_representation_conjugation(self, S):
        S = S.copy()
        S_qubits = S.shape[0] // 2
        if S_qubits < self._largestqb:
            raise ValueError(f"This gate acts on qubits not present in S")
        affected_rows = np.array(self.qubits)
        S[affected_rows] = (S[affected_rows] + S[affected_rows + S_qubits]) % 2
        return S
    
    def phases_computation(self, phases, S, return_Sp=False):
        phases = phases.copy()
        nqubits = S.shape[0] // 2
        for qubit in self.qubits:
            z_row = S[qubit, :]
            x_row = S[nqubits+qubit, :]
            phases = (phases + z_row*x_row) % 2
        if return_Sp:
            return phases, self.binary_representation_conjugation(S)
        else:
            return phases

class CZ:
    def __init__(self, ctrlqbs, tgqbs):
        try:
            clen = len(ctrlqbs)
            try:
                if len(tgqbs) != clen:
                    raise ValueError("Ctrlqbs and tgqbs must have the same length")
            except TypeError:
                raise ValueError("Ctrlqbs and tgqbs must have the same length")
        except TypeError:
                try:
                    len(tgqbs)
                    raise ValueError("Ctrlqbs and tgqbs must have the same length")
                except TypeError:
                    ctrlqbs = [ctrlqbs]
                    tgqbs = [tgqbs]

        for qb in ctrlqbs:
            if qb in tgqbs:
                raise ValueError("Ctrl and target qubits sets must not intersect")
        
        ctrl_counter = Counter(ctrlqbs)
        trgt_counter = Counter(tgqbs)
        for nctrl in ctrl_counter.values():
            if nctrl > 1:
                raise ValueError("Ther must not be repeated qubit indices in ctrlqbs")
        for ntrgt in trgt_counter.values():
            if ntrgt > 1:
                raise ValueError("Ther must not be repeated qubit indices in tgqbs")

        self._largestqb = max(max(ctrlqbs), max(tgqbs))
        self.qubits = [ctrlqbs, tgqbs]
    
    @property
    def ctrl(self):
        return self.qubits[0]

    @property
    def target(self):
        return self.qubits[1]
    
    @property
    def pairs(self):
        for ctrl, target in zip(self.ctrl, self.target):
            yield ctrl, target

    def binary_representation_conjugation(self, S):
        S = S.copy()
        S_qubits = S.shape[0] // 2
        if S_qubits < self._largestqb:
            raise ValueError(f"This gate acts on qubits not present in S")
        ctrls = np.array(self.ctrl)
        trgt = np.array(self.target)
        S[ctrls, :] = (S[ctrls, :] + S[trgt + S_qubits, :]) % 2
        S[trgt, :] = (S[trgt, :] + S[ctrls + S_qubits, :]) % 2
        return S
    
    def phases_computation(self, phases, S, return_Sp=False):
        phases = phases.copy()
        nqubits = S.shape[0] // 2
        for ctrlqb, targetqb in self.pairs:
            z_ctrl_row = S[ctrlqb, :]
            z_trgt_row = S[targetqb, :]
            x_ctrl_row = S[nqubits + ctrlqb, :]
            x_trgt_row = S[nqubits + targetqb, :]
            phases = (phases + x_ctrl_row*x_trgt_row*((z_ctrl_row + z_trgt_row) % 2)) % 2
        if return_Sp:
            return phases, self.binary_representation_conjugation(S)
        else:
            return phases

    def __repr__(self):
        ctrl_indices = ", ".join([str(ctrlind) for ctrlind in self.ctrl])
        target_indices = ", ".join([str(trgtind) for trgtind in self.target])
        indices = f"({ctrl_indices}), ({target_indices})"
        return f"CZ[{indices}]"

class InstructionList:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.instructions = []

    def add(self, *instructions):
        try:
            for instruction in instructions:
                if instruction._largestqb >= self.nqubits:
                    raise ValueError("All instructions must fit in the InstructionList qubit set 0 <= n < N")
                if len(self.instructions) > 0:
                    if type(instruction) is type(self.instructions[-1]):
                        if type(instruction) is H:
                            self.instructions[-1].combine(instruction)
                            if len(self.instructions[-1].qubits) == 0:
                                del self.instructions[-1]
                        if type(instruction) is CZ:
                            last_pairs = list(self.instructions[-1].pairs)
                            pairs_to_add = list(instruction.pairs)
                            last_all_qb = set(self.instructions[-1].ctrl + self.instructions[-1].target)
                            for i, pair in enumerate(instruction.pairs):
                                if pair in last_pairs:
                                    last_pairs.remove(pair)
                                    pairs_to_add.remove(pair)
                                else: 
                                    if ({*pair} == ({*pair} - last_all_qb)):
                                        last_pairs.append(pair)
                                        pairs_to_add.remove(pair)
                            self.instructions[-1] = CZ([pair[0] for pair in last_pairs], [pair[1] for pair in last_pairs])
                            if len(pairs_to_add) > 0:
                                self.instructions.append(CZ([pair[0] for pair in pairs_to_add], [pair[1] for pair in pairs_to_add]))
                        if type(instruction) is P:
                            self.instructions.append(instruction)
                        else:
                            ValueError("Instructions must be a Clifford Object [H, P, CZ]")
                    else:
                        if type(instruction) in [H, P, CZ]:
                            self.instructions.append(instruction)
                        else:
                            ValueError("Instructions must be a Clifford Object [H, P, CZ]")
                else:
                    if type(instruction) in [H, P, CZ]:
                        self.instructions.append(instruction)
                    else:
                        ValueError("Instructions must be a Clifford Object [H, P, CZ]")
        except TypeError:
            if type(instruction) in [H, P, CZ]:
                self.instructions.append(instruction)
            else:
                ValueError("Instructions must be a Clifford Object [H, P, CZ]")
    
    def remove(self, index):
        del self.instructions[index]

    def __repr__(self):
        gates = ", ".join(str(instruction) for instruction in self.instructions)
        return f"InstructionList<{self.nqubits}>[{gates}]"
    
    def __getitem__(self, i):
        return self.instructions[i]

    def binary_representation_conjugation(self, S):
        for instruction in self.instructions:
            S = instruction.binary_representation_conjugation(S)
        return S
    
    def phases_computation(self, S, return_Sp=False):
        phases = np.zeros(S.shape[1], dtype=int)
        S = S.copy()
        for instruction in self.instructions:
            phases, S = instruction.phases_computation(phases, S, return_Sp=True)
        if return_Sp:
            return phases, S
        else:
            return phases
    
    def to_qiskit_circuit(self, barriers_before_cz=True):
        qc = QuantumCircuit(self.nqubits)
        last_instruction_type = None
        for instruction in self.instructions:
            # if last_instruction_type != CZ and type(instruction) == CZ:
            #     qc.barrier()
            if type(instruction) == H:
                qc.h(instruction.qubits)
                last_instruction_type = H
            elif type(instruction) == P:
                qc.s(instruction.qubits)
                last_instruction_type = P
            else:
                qc.cz(instruction.ctrl, instruction.target)
                last_instruction_type = CZ
        return circutils.simplify_logical_circuits(qc)
    
# UNITARY SYNTHESIS

# References: Quantum 5, 385 (2021)
#             PRA 70, 052328 (2004) Lemma 3

def get_S_Rinv(pauli_representation):
    gauss_pauli_representation = mod2_gaussian_elimination(pauli_representation)
    nrows, ncols = gauss_pauli_representation.shape
    pivot_cols = []
    for j in range(ncols):
        if np.sum(gauss_pauli_representation[j:, j].astype(int)) == 1:
            pivot_cols.append(j)
    S = pauli_representation[:, pivot_cols]
    R0_inv = gauss_pauli_representation[pivot_cols]
    return S, R0_inv

def first_reduction_step(S):
    # PRA 70, 052328 (2004) Lemma 3 (Adapted to the other's article notation)
    nrows = S.shape[0]//2
    ncols = S.shape[1]
    S_gauss = mod2_gaussian_elimination(S.T).T
    S_rank = np.linalg.matrix_rank(S_gauss)
    Sx_rank = np.linalg.matrix_rank(S_gauss[nrows:])
    C_diag = np.diagonal(S_gauss[:S_rank-Sx_rank, :S_rank-Sx_rank])
    # zero_diag_elements = np.where(np.equal(C_diag, 0))[0]
    row_swaps = Permutation.identity(nrows)
    for i in range(len(C_diag)):
        if C_diag[i] == 0:
            zero_diag_col = S[:, i]
            first_one_row_ind = np.where(np.equal(zero_diag_col, 1))[0][0]
            row_swaps = row_swaps * Permutation.transposition(nrows, i, first_one_row_ind)
            S_gauss[[i, first_one_row_ind]] = S[[first_one_row_ind, i]]
    hadamard_qubits = []
    for i, row in enumerate(S_gauss[nrows:, :]):
        if row[::-1][min(i, ncols - 1)] != 1:
            hadamard_qubits.append(row_swaps(i))
    instructions = InstructionList(nrows)
    instructions.add(H(*hadamard_qubits))
    S1 = instructions.binary_representation_conjugation(S)
    return S1, instructions

# def first_reduction_step(S):
#     nrows = S.shape[0]//2
#     ncols = S.shape[1]
#     S_gauss = mod2_gaussian_elimination(S.T).T
#     Sx_rank = np.linalg.matrix_rank(S_gauss[nrows:])

def second_reduction_step(S1, instructions=None):
    nrows = S1.shape[0]//2
    if instructions is None:
        instructions = InstructionList(nrows)
    R1_inv = S1[nrows:nrows+np.linalg.matrix_rank(S1), :] % 2
    R1 = (np.linalg.inv(R1_inv).astype(int))
    S2 = (S1 @ R1) % 2
    return S2, R1_inv, instructions

def third_reduction_step(S2, instructions=None):
    nrows = S2.shape[0]//2
    initial_ncols = S2.shape[1]
    ncols = initial_ncols
    if instructions is None:
        instructions = InstructionList(nrows)
    symbols = [sym.symbols(f"b{i}") for i in range(2*nrows)]
    reversed_blocks_symbols = symbols[nrows:] + symbols[:nrows]
    S2_rank = np.linalg.matrix_rank(S2)
    S3 = S2.copy()
    for n in range(S2_rank+1, nrows+1):
        expressions = []
        for colind in range(ncols):
            col = S3[:, colind]
            nonzero_indices = np.where(col)[0]
            if len(nonzero_indices) > 1:
                this_expression = reversed_blocks_symbols[nonzero_indices[0]] ^ reversed_blocks_symbols[nonzero_indices[1]]
                for nonzero_index in nonzero_indices[2::]:
                    this_expression = this_expression ^ reversed_blocks_symbols[nonzero_index]
            else:
                this_expression = reversed_blocks_symbols[nonzero_indices[0]]
            expressions.append(~this_expression)
        last_expression = symbols[0] & symbols[nrows]
        for i in range(1, nrows):
            last_expression = (last_expression) ^ (symbols[i] & symbols[nrows + i])
        expressions.append(~last_expression)
        final_clause = expressions[0] & expressions[1]
        for expression in expressions[2::]:
            final_clause = final_clause & expression
        clause_valid_solutions = sym.logic.inference.satisfiable(final_clause, all_models=True, algorithm="z3") # Z3 SAT algorithm prioritizes low weight solutions
        for valid_solution in clause_valid_solutions:
            solution_col = np.array([valid_solution[var] for var in symbols], dtype=bool)
            if (solution_col[nrows:].sum() == 0):
                continue
            S3_candidate = np.concatenate((S3, solution_col.reshape(2*nrows, 1)), axis=1)
            if np.linalg.matrix_rank(S3_candidate[nrows:]) == n:
                S3 = S3_candidate.copy()
                ncols = S3.shape[1]
                break
    R2_inv = np.concatenate((np.identity(initial_ncols, dtype=int), np.zeros((ncols - initial_ncols, initial_ncols), dtype=int)), axis=0)
    return S3, R2_inv, instructions

# def third_reduction_step(S2, instructions=None):
#     nrows = S2.shape[0]//2
#     initial_ncols = S2.shape[1]
#     if instructions is None:
#         instructions = InstructionList(nrows)
#     S2_null_space = (S2.T.null_space()).T
#     ncols_to_add = nrows - np.linalg.matrix_rank(S2)
#     solutions_weight = np.sum(np.array(S2_null_space, dtype=int), axis=0)
#     sorted_weight_indices = np.argsort(solutions_weight)
#     cols_to_add = np.zeros((S2.shape[0], ncols_to_add), dtype=int)
#     S3 = S2.copy()
#     added_cols = 0
#     for ind in sorted_weight_indices:
#         if added_cols >= ncols_to_add:
#             break
#         S3_candidate = np.concatenate((S3, S2_null_space[:, ind].reshape(2*nrows, 1)), axis=1)
#         if np.linalg.matrix_rank(S3_candidate[nrows:]) == S3_candidate.shape[1]:
#             S3 = S3_candidate.copy()
#             added_cols += 1
#     # S3 = np.concatenate((S2, S2_null_space[:, cols_to_add]), axis=1)
#     R2_inv = np.concatenate((np.identity(initial_ncols, dtype=int), np.zeros((ncols_to_add, initial_ncols), dtype=int)), axis=0)
#     return S3, R2_inv, instructions

def fourth_reduction_step(S3, instructions=None):
    nrows = S3.shape[0]//2
    if instructions is None:
        instructions = InstructionList(nrows)
    R3_inv = S3[nrows:, :]
    R3 = np.linalg.inv(R3_inv)
    S4prev = ((S3 @ R3) % 2)
    Ediag = np.diagonal(S4prev)
    non_zero_indices = np.where(Ediag)[0]
    if len(non_zero_indices) > 0:
        Q = P(*non_zero_indices)
        instructions.add(Q)
        S4 = Q.binary_representation_conjugation(S4prev)
        return S4, R3_inv, instructions
    else:
        return S4prev, R3_inv, instructions

def fifth_reduction_step(S4, instructions=None):
    nrows = S4.shape[0]//2
    if instructions is None:
        instructions = InstructionList(nrows)
    S5 = S4.copy()
    upper_half = S5[:nrows, :]
    lower_half = S5[nrows:, :]
    # Reduce upper half
    non_zeros_upper = np.where(upper_half)
    cz_pairs = np.concatenate((non_zeros_upper[0].reshape(len(non_zeros_upper[0]), 1), non_zeros_upper[1].reshape(len(non_zeros_upper[1]), 1)), axis=1)
    cz_pairs = np.unique(np.sort(cz_pairs, axis=1), axis=0)
    cz_pairs_distance = np.abs(cz_pairs[:, 1] - cz_pairs[:, 0])
    mask = np.argsort(cz_pairs_distance)
    cz_pairs = cz_pairs[mask, :]
    Q = InstructionList(nrows)
    for pair in cz_pairs:
        this_Q = CZ(pair[0], pair[1])
        Q.add(this_Q)
    S5 = Q.binary_representation_conjugation(S5)
    instructions.add(*Q)
    # Add Hadamards to flip blocks
    last_hadamards = H(*np.arange(nrows))
    S5 = last_hadamards.binary_representation_conjugation(S5)
    instructions.add(last_hadamards)
    return S5, instructions

def reduce_binary_string_representation(pauli_representation):
    S, R0_inv = get_S_Rinv(pauli_representation)
    S1, instructions = first_reduction_step(S)
    S2, R1_inv, instructions = second_reduction_step(S1, instructions)
    S3, R2_inv, instructions = third_reduction_step(S2, instructions)
    S4, R3_inv, instructions = fourth_reduction_step(S3, S2, instructions)
    S5, instructions = fifth_reduction_step(S4, instructions)
    R_inv = (R3_inv @ R2_inv @ R1_inv @ R0_inv) % 2
    phases = instructions.phases_computation(pauli_representation)
    return [S, S1, S2, S3, S4, S5], instructions, R_inv, phases

def paulis_diagonalization_circuit(commuting_pauli_list):
    # Quantum 5, 385 (2021)
    all_commute = len(commuting_pauli_list.commutes_with_all(commuting_pauli_list)) == len(commuting_pauli_list)
    if not all_commute:
        raise ValueError("The provided PauliList contains non-commuting operators")
    nqubits = len(commuting_pauli_list[0])
    non_diagonal_qubits_num = count_non_diagonal_qubits(commuting_pauli_list)
    if non_diagonal_qubits_num == 0:
        return QuantumCircuit(nqubits)
    Sp = binary_pauli_representation(commuting_pauli_list)
    redS, instructions, R_inv, phases = reduce_binary_string_representation(Sp)
    return instructions.to_qiskit_circuit(), R_inv, phases