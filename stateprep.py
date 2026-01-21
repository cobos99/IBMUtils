from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_gate

class BasisStatePrep(QuantumCircuit):
    def __init__(self, chain_length, int_label):
        super().__init__(chain_length)
        int_bstring = f"{int_label:0{chain_length}b}"
        for i, bit in enumerate(int_bstring):
            if bit == '1':
                self.x(i)

class GHZBlockChainStatePrep(QuantumCircuit):
    def __init__(self, nqubits, block_number):
        super().__init__(nqubits)
        if nqubits < 2:
            raise ValueError("nqubits must be at least 2 for GHZ state preparation.")
        if nqubits % block_number != 0:
            raise ValueError("nqubits must be divisible by the number of blocks for GHZ state preparation.")
        block_size = nqubits // block_number
        if block_size < 2:
            raise ValueError(f"Block size = {block_size} must be at least 2 for GHZ state preparation.")
        for i in range(block_number):
            start = i * block_size
            end = start + block_size
            center = start + block_size // 2
            self.h(center)
            for j in reversed(range(start, center)):
                self.cx(j+1, j)
            for j in range(center, end-1):
                self.cx(j, j+1)