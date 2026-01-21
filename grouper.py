# ADAPTED AND EXTENDED FROM:
# https://github.com/Mc-Zen/htgrouper/blob/master/data/ht_grouper_helpers.py

import numpy as np
import qiskit
from qiskit.providers import Job
from qiskit.quantum_info import Pauli
from qiskit.result import Result
import json
from typing import Dict, List, Optional, Sequence, Union
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation
from qiskit.circuit.library import HGate


h_gate_canceller = PassManager([InverseCancellation([HGate()])])


def reverse_paulis(hamiltonian: Dict[str, float]) -> Dict[str, float]:
    """Reverse the qubit order of Paulis in a dictionary where Paulis are the keys. """
    return {key[::-1]: value for key, value in hamiltonian.items()}


def write_hamiltonian_to_json(filename: str, hamiltonian: Dict[str, float]):
    """
    Write a hamiltonian in form of a dictionary with Pauli strings
    as keys and real-valued coefficients as values to a JSON file. 

    Pauli strings should read as 
       `"XYZ"` -> X on qubit 0, Y on qubit 1, Z on qubit 2
    """
    with open(filename, "w") as file:
        json.dump(hamiltonian, file, indent=2)


def read_hamiltonian_from_json(filename: str) -> Dict[str, float]:
    """
    Read a hamiltonian from a JSON file containing Pauli strings
    as keys and real-valued coefficients as values. 

    Pauli strings are read as 
       `"XYZ"` -> X on qubit 0, Y on qubit 1, Z on qubit 2

    Returns a dictionary of the same format. 
    """
    with open(filename) as file:
        return json.load(file)


def read_grouping_from_json(filename: str) -> List[dict]:
    """
    Read a Pauli grouping from a JSON file of the following format:
    ```
    {
      "grouping": [
        {
          "operators": ["III","IIZ","ZZZ"],
          "edges": [],
          "cliffords": ["H","H","H"]
        },
        {
          "operators": ["XYZ","YZI","IIX"],
          "edges": [[0,1],[1,2]],
          "cliffords": ["S", "HS", "I"]
        }
      ]
    }
    ```

    A grouping is a list of groups, each containing a list of Pauli 
    operators, a list of edges (stored as 2-element lists) representing
    the graphs (or CZ gates) and a list of Clifford gates that need to be 
    applied to each qubit in the readout circuit. 

    Valid Clifford gates are "I", "H", "S", "SH", "HS" and "HSH". 

    Pauli strings are read as 
       `"XYZ"` -> X on qubit 0, Y on qubit 1, Z on qubit 2


    Parameters
    ----------
    filename : str
        Input file path

    Returns
    -------
    List[dict]
        Grouping as a list of dictionaries similar to the form shown above. 
    """
    with open(filename) as file:
        result = json.load(file)
        return result["grouping"]


def generate_readout_circuits(grouping: List[dict], nqubits: int) -> List[QuantumCircuit]:
    """
    Generate readout circuits from a Pauli grouping specified in the format
    as described in the documentation of :func:``read_grouping()``
    """
    circuits = []
    clifford_map = {"H": QuantumCircuit.h, "S": QuantumCircuit.s, "I": lambda *params: None}
    for group in grouping:
        num_qubits = nqubits
        edges = group["edges"]
        cliffords = group["cliffords"]
        circuit = QuantumCircuit(num_qubits)
        for qubit, clifford in enumerate(cliffords):
            for gate in clifford[::-1]:
                clifford_map[gate](circuit, qubit)

        for q1, q2 in edges:
            circuit.cz(q1, q2)
        circuit.h(range(num_qubits))

        circuits.append(h_gate_canceller.run(circuit))
    return circuits


Bitstring = np.int64


class BinaryResult:
    """
    Class for storing a single quantum circuit result with result
    signature and count. 

    The result signature is encoded in a binary string in little-endian
    (least-significant bit represents the zeroth quantum bit result). 

    """

    __slots__ = ("bitstring", "count")

    def __init__(self, bitstring: Union[Bitstring, int], count: int):
        self.bitstring = Bitstring(bitstring)
        self.count = count

    def __eq__(self, other) -> bool:
        return self.bitstring == other.bitstring and self.count == other.count

    def __str__(self, num_qubits=0) -> str:
        return f"{self.bitstring:0>{num_qubits}b}: {self.count}"

    def __repr__(self) -> str:
        return f"BinaryResult({self.bitstring}, {self.count})"


class CircuitResult:
    """
    Class encapsuling results from one quantum circuit, similar to 
    :class:`qiskit.result.ExperimentResultData` but the keys are not stored
    using hexadecimal strings but binary bitstrings which allows for
    some performance improvements when evaluating readout results. . 

    """

    __slots__ = ("results", "num_qubits")

    def __init__(self, counts: Dict[str, int], qubits: Optional[Sequence[int]] = None):
        """Create a CircuitResult from a dictionary as returned by `qiskit.result.get_counts()`.
        The keys are little-endian bitstrings, i.e. the zeroth register is at the rightmost 
        position in the string.

        E.g., ``{"001": 5, "010": 4}`` to denote a result where the outcome with only the zeroth 
        qubit being 1 occured 5 times and the outcome of the first qubit begin 1 occured 5 times. 

        Optionally, the results on only a selection of qubits can be extracted (marginalization).  
        Example: 

        >>> CircuitResult({"101": 9, "001": 4}, [0, 2])

        will result in storing ``{"11": 9, "01": 4}``. 

        Parameters
        ----------
        counts : Dict[str, int]
            Outcome/count dictionary
        qubits : Optional[Sequence[int]], optional
            If specified, only the given qubits are extracted. 
        """
        self.results: List[BinaryResult] = []

        if qubits is None:
            if len(counts) != 0:
                self.num_qubits = len(next(iter(counts)).replace(" ", ""))
            for key, value in counts.items():
                self.results.append(BinaryResult(Bitstring(int(key.replace(" ", ""), 2)), value))

        else:
            self.num_qubits = len(qubits)
            for key, value in counts.items():
                key = key.replace(" ", "")  # might contain spaces to separate registers
                key = "".join(key[index] for index in qubits)
                self.results.append(BinaryResult(Bitstring(int(key, 2)), value))

    def __str__(self) -> str:
        return "CircuitResult(" + ", ".join(result.__str__(self.num_qubits) for result in self.results) + ")"
    
# Other selfmade utilities

from hardware.utils.measure import pauli_weight
from collections import Counter
import matplotlib.pyplot as plt

def generate_config_string(ham_json, outfile, connectivity,
                           num_graphs=1000, max_edge_count=1000, sort_graphs_by_edgecount=True,
                           extract_computational_basis=True, intermediate_file_frequency=0,
                           numthreads=1):
    confstring = f"""filename = {ham_json}                      # where to read the hamiltonian data from 
outfilename = {outfile}                                     # where to output the grouping
connectivity = {connectivity}                               # which connectivity file to use
numGraphs = {num_graphs}                                    # Hyperparameter: Maximum number of random subgraphs
maxEdgeCount = {max_edge_count}                             # Hyperparameter: Maximum number of edges for subgraphs
sortGraphsByEdgeCount = {str(sort_graphs_by_edgecount).lower()}          # Sort possible subgraphs by edge count so graphs with lower edge count are preferred
extractComputationalBasis = {str(extract_computational_basis).lower()}   # Pre-eliminate Paulis in the computational basis like IIZ, ZIZ, ZZZ, ...
intermediateFileFrequency = {intermediate_file_frequency}   # If != 0, an output file is written in the form <filename>_savingpoint_X.<extension> for every nth group
numThreads = {numthreads}                                   # option for multithreading"""
    return confstring

def plot_weight_histogram(diagonal_observables_per_circuit):
    weights = []
    for pl in diagonal_observables_per_circuit:
        for p in pl:
            weights.append(pauli_weight(p))
    wcount = Counter(weights)
    plt.bar(wcount.keys(), wcount.values(), color=plt.get_cmap("Set2")(0), width=0.8)
    plt.xticks(list(wcount.keys()), list(wcount.keys()))
    plt.xlabel("Pauli weight")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pauli weights")
    plt.show()

def get_unique_weights_observables(diagonal_observables_per_circuit):
    unique_obs_weights = []
    unique_obs_indices = []
    unique_dobs_per_circuit_sb = []
    for i, pl in enumerate(diagonal_observables_per_circuit):
        for j, p in enumerate(pl):
            weight = pauli_weight(p)
            if weight not in unique_obs_weights:
                unique_dobs_per_circuit_sb.append(p)
                unique_obs_indices.append((i, j))
                unique_obs_weights.append(weight)
    return unique_obs_weights, unique_obs_indices, unique_dobs_per_circuit_sb