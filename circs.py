from qiskit.transpiler.passes import Optimize1qGates, SetLayout, ApplyLayout, FullAncillaAllocation, Optimize1qGatesDecomposition, Optimize1qGatesSimpleCommutation, RemoveBarriers
from qiskit.converters import circuit_to_instruction, dag_to_circuit, circuit_to_dag
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, StagedPassManager, Layout
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions
from hardware.utils.misc import iterate_nested_list
from qiskit.circuit.library import XGate
import hardware.utils.measure as measure
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit, qpy
import samplomatic
import numpy as np
import os

def count_non_idle_qubits(circ):
    if type(circ) is not DAGCircuit:
        dag = circuit_to_dag(circ)
    idle_wires = list(dag.idle_wires())
    return dag.num_qubits() - len(idle_wires)

def remove_idle_qwires(circ, relabeling=None):
    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())
    num_non_idle = dag.num_qubits() - len(idle_wires)
    dag.remove_qubits(*idle_wires)

    layout = circ.layout.final_index_layout() if circ.layout is not None else list(range(num_non_idle))
    qubits_to_apply = list(np.argsort(layout))
    circ_without_idle = dag_to_circuit(dag)
    circ_instruction = circuit_to_instruction(circ_without_idle)
    qc_args = (num_non_idle, num_non_idle) if circ.num_clbits > 0 else (num_non_idle,)
    to_return = QuantumCircuit(*qc_args)
    to_return = to_return.compose(circ_instruction, qubits=qubits_to_apply).decompose()
    if relabeling is not None:
        before_relabeling = to_return.copy()
        qc_args = (before_relabeling.num_qubits, before_relabeling.num_qubits) if before_relabeling.num_clbits > 0 else (before_relabeling.num_qubits,)
        to_return = QuantumCircuit(*qc_args)
        measurement_operations = []
        for instruction_data in before_relabeling.data:
            if instruction_data.operation.name == "measure":
                measurement_operations.append(instruction_data)
            elif instruction_data.operation.name == "barrier":
                to_return.barrier()
            else:
                to_return.append(instruction_data.operation, qargs=[relabeling[qubit._index] for qubit in instruction_data.qubits])
        # Place the measurement operations at the end
        for measurement_instruction in measurement_operations:
            new_indices = [relabeling[qubit._index] for qubit in measurement_instruction.qubits]
            to_return.append(measurement_instruction.operation, qargs=new_indices, cargs=new_indices)
        
    return to_return

def remove_final_measurements(circuit):
    circuit = circuit.copy()
    for op in circuit.data[::-1]:
        if op.operation.name == "measure":
            circuit.data.remove(op)
        elif op.operation.name == "barrier":
            circuit.data.remove(op)
        else:
            break
    return circuit

def remove_barriers(circuit):
    circuit = circuit.copy()
    if circuit.layout is not None:
        layout = sorted(circuit.layout.final_index_layout())
    else:
        layout = None
    circuit = RemoveBarriers()(circuit)
    circuit = remove_idle_qwires(circuit)
    pm = generate_preset_pass_manager(backend=None, optimization_level=0, initial_layout=layout)
    circuit = pm.run(circuit)
    return circuit

def remove_barriers_transpiled(circuit, backend, layout=None):
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend, initial_layout=layout)
    return pm.run(RemoveBarriers()(circuit))

def check_and_measure_active_qubits(circuit):
    if not circuit.cregs:
        circuit = circuit.copy()
        circuit.measure_active()
    return circuit

def compute_uncompute(original_circuit, barrier=False):
    to_return = original_circuit.copy()
    if barrier: to_return.barrier()
    return to_return.compose(original_circuit.inverse())

def simplify_logical_circuits(circuits, optimization_level=3, backend=None):
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    return pm.run(circuits)

def join_transpiled_circuits(*circuits, layout_circ=0, backend=None):
    layout = circuits[layout_circ].layout.final_index_layout()
    joined_circuit = QuantumCircuit(circuits[layout_circ].num_qubits, circuits[layout_circ].num_clbits)
    for circuit in circuits:
        joined_circuit = joined_circuit.compose(circuit)
    # Reorder measurements
    reordered_circ = QuantumCircuit(joined_circuit.num_qubits, joined_circuit.num_clbits)
    measurements = []
    for instruction_data in joined_circuit.data:
        if instruction_data.operation.name == "measure":
            measurements.append(instruction_data)
        else:
            reordered_circ.append(instruction_data.operation, qargs=[qubit._index for qubit in instruction_data.qubits])
    for measurement_instruction in measurements:
        indices = [qubit._index for qubit in measurement_instruction.qubits]
        reordered_circ.append(measurement_instruction.operation, qargs=indices, cargs=indices)
    if backend is not None:
        joined_circuit = remove_idle_qwires(reordered_circ, relabeling=np.arange(len(layout))[np.argsort(layout)])
        layout_dict = {layout[i]:reordered_circ.qubits[i] for i in range(joined_circuit.num_qubits)}
        layout_pm = PassManager([SetLayout(layout=Layout(layout_dict)), FullAncillaAllocation(coupling_map=backend.target), ApplyLayout()])
        sqcancel_pm = PassManager([Optimize1qGates(target=backend.target), Optimize1qGatesDecomposition(target=backend.target), Optimize1qGatesSimpleCommutation(target=backend.target)])
        sqopt_pm = StagedPassManager(stages=["optimization", "layout"], layout=layout_pm, optimization=sqcancel_pm)
        reordered_circ = sqopt_pm.run(reordered_circ)
        return reordered_circ
    else:
        return reordered_circ
    
def append_basis_change_circuit(circuit, basis, backend=None):
    nqubits = count_non_idle_qubits(circuit)
    change_of_basis = QuantumCircuit(nqubits)
    if basis == "Y":
        change_of_basis.sdg(np.arange(nqubits))
        change_of_basis.h(np.arange(nqubits))
    if basis == "X":
        change_of_basis.h(np.arange(nqubits))
    if (backend is not None) and (circuit.layout is not None):
        pm = generate_preset_pass_manager(backend=backend, initial_layout=circuit.layout.final_index_layout(), optimization_level=0)
        change_of_basis = pm.run(change_of_basis)
    return join_transpiled_circuits(circuit, change_of_basis, backend=backend)

def depth2qb(circ):
    if isinstance(circ, (list, tuple, np.ndarray)):
        return np.array([depth2qb(c) for c in circ])
    else:
        return circ.depth(lambda x: len(x.qubits)>1 and x.name!="barrier")

def load_circs(filepath):
    with open(filepath, "rb") as f:
        circs = qpy.load(f)
    return circs

def save_circs(filepath, circs):
    with open(filepath, "wb") as f:
        qpy.dump(circs, f)

def generate_and_save_circs(filepath, circs_function, *circs_func_args):
    if os.path.exists(filepath):
        circs = load_circs(filepath)
    else:
        circs = circs_function(*circs_func_args)
        save_circs(filepath, circs)
    return circs

def save_many_layout_circs(filepath, circuit_matrix):
    nlayouts = len(circuit_matrix)
    flatten_circuits = []
    for circ_arr in circuit_matrix:
        flatten_circuits += circ_arr
    flatten_circuits[0].metadata = {"nlayouts": nlayouts}
    save_circs(filepath, flatten_circuits)

def load_many_layout_circs(filepath):
    circuits = load_circs(filepath)
    nlayouts = circuits[0].metadata["nlayouts"]
    circuits_matrix = []
    len_each_layout_circs = len(circuits) // nlayouts
    for i in range(nlayouts):
        circuits_matrix.append(circuits[i*len_each_layout_circs:(i+1)*len_each_layout_circs])
    return circuits_matrix

def numpy_array_of_circuits(circuits):
    if isinstance(circuits, np.ndarray):
        return circuits
    dimensions = 0
    last_element = circuits
    dimensions_length = []
    if isinstance(last_element, QuantumCircuit):
        arr = np.array([], dtype=object)
        arr.append(last_element)
        return arr
    while not isinstance(last_element, QuantumCircuit):
        try:
            dimensions_length.append(len(last_element))
            last_element = last_element[0]
            dimensions += 1
        except TypeError:
            raise TypeError("circuits is not a list of circuits")
    arr = np.empty(dimensions_length, dtype=object)
    dimensions_length = np.array(dimensions_length)
    for i, qc in enumerate(iterate_nested_list(circuits)):
        indices = np.unravel_index(i, dimensions_length)
        arr[indices] = qc
    return arr

def append_pauli_string_basischange(logical_circuit, pauli):
    pauli_str = measure.pauli_to_str(pauli)
    nqubits = len(pauli_str)
    if logical_circuit.num_qubits != nqubits:
        raise ValueError("The provided circuit and pauli do not act on the same number of qubits")
    change_of_basis = QuantumCircuit(nqubits)
    for i, opel in enumerate(pauli_str):
        match opel:
            case 'X':
                change_of_basis.h(i)
            case 'Y':
                change_of_basis.sdg(i)
                change_of_basis.h(i)
            case _:
                pass
    final_circuit = logical_circuit.copy()
    final_circuit.barrier()
    final_circuit = final_circuit.compose(change_of_basis).decompose()
    final_circuit.measure_all()
    return final_circuit

def get_backend_best_qubit_chain(backend, nqubits):
    qlists = backend.properties().general_qlists
    for qlist in qlists:
        if qlist["name"] == f"lf_{nqubits:d}":
            return qlist["qubits"]

def transpile_circuits_with_consistent_layout(logical_circuits, backend, target=None, layout=None, optimization_level=3):
    logical_circuits = numpy_array_of_circuits(logical_circuits)
    ogshape = logical_circuits.shape
    logical_circuits = logical_circuits.flatten().tolist()
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, target=target, initial_layout=layout)
    if layout is None:
        transpiled_circuits = pm.run(logical_circuits[:2])
        if transpiled_circuits[1].layout is not None:
            first_layout = transpiled_circuits[1].layout.final_index_layout()
        else:
            first_layout = None
        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, target=target, initial_layout=first_layout)
        transpiled_circuits += pm.run(logical_circuits[2:])
    else:
        transpiled_circuits = pm.run(logical_circuits)
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits).reshape(ogshape)
    return transpiled_circuits

def transpile_circuit_to_many_layouts(logical_circuit, backend, initial_layouts, optimization_level=3):
    if not isinstance(logical_circuit, QuantumCircuit):
        raise ValueError("logical_circuit must be a QuantumCircuit")
    pms = [generate_preset_pass_manager(backend=backend, initial_layout=layout, optimization_level=optimization_level) for layout in initial_layouts]
    transpiled_circuits = np.full(len(initial_layouts), None, dtype=object)
    for i, pm in enumerate(pms):
        transpiled_circuits[i] = pm.run(logical_circuit)
    return transpiled_circuits

def box_circuits(circ_arr, **box_kwargs):
    bpm = generate_boxing_pass_manager(**box_kwargs)
    ogshape = circ_arr.shape
    circ_arr = circ_arr.flatten().tolist()
    boxed_circuits = bpm.run(circ_arr)
    boxed_circuits = numpy_array_of_circuits(boxed_circuits).reshape(ogshape)
    return boxed_circuits

def get_unique_boxes(circ_arr, undress_boxes=True, normalize_annotations=None):
    if isinstance(circ_arr, QuantumCircuit):
        circ_arr = [circ_arr]
    elif isinstance(circ_arr, (list, tuple)):
        if not all(isinstance(c, QuantumCircuit) for c in circ_arr):
            raise ValueError("All elements of circ_arr must be QuantumCircuits")
    else:
        raise ValueError("circ_arr must be a QuantumCircuit or a list/tuple of QuantumCircuits")
    unique_instructions = []
    for circ in circ_arr:
        unique_instructions.append(find_unique_box_instructions(circ, undress_boxes=undress_boxes, normalize_annotations=normalize_annotations))
    return unique_instructions

def samplomatic_build(circuits):
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    circuits = numpy_array_of_circuits(circuits)
    ogshape = circuits.shape
    templates = []
    samplexes = []
    for circ in circuits.flatten():
        this_template, this_samples = samplomatic.build(circ)
        templates.append(this_template)
        samplexes.append(this_samples)
    templates = numpy_array_of_circuits(templates).reshape(ogshape).squeeze().tolist()
    samplexes = np.array(samplexes, dtype=object).reshape(ogshape).squeeze().tolist()
    return templates, samplexes

def translate_to_backend_gateset(backend, circuits, optimization_level=3):
    pm = generate_preset_pass_manager(basis_gates=backend.configuration().basis_gates, optimization_level=optimization_level)
    return pm.run(circuits)

def sqrtX_to_X(qc, backend=None):
    qc = qc.copy()
    if qc.layout is not None:
        layout = sorted(qc.layout.final_index_layout())
    else:
        layout = None
    dag = circuit_to_dag(qc)
    for node in dag.op_nodes():
        if node.op.name == "sx":
            dag.substitute_node(node, XGate())
    qc = dag_to_circuit(dag)
    qc = remove_idle_qwires(qc)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=layout)
    qc = pm.run(qc)
    return qc

def sqrtX_to_X_except_barrier_interval(qc, barrier_interval, backend=None):
    qc = qc.copy()
    nbarriers = qc.count_ops().get("barrier", 0)
    barrier_interval = np.array(barrier_interval)
    negative_mask = barrier_interval < 0
    barrier_interval[negative_mask] = nbarriers + barrier_interval[negative_mask]
    dag = circuit_to_dag(qc)
    if qc.layout is not None:
        layout = sorted(qc.layout.final_index_layout())
    else:
        layout = None
    found_barriers = 0
    for node in dag.topological_op_nodes():
        if node.name == "barrier":
            found_barriers += 1
        if not barrier_interval[0] <= found_barriers <= barrier_interval[1]-1:
            if node.name == "sx":
                dag.substitute_node(node, XGate())
    qc = dag_to_circuit(dag)
    qc = remove_idle_qwires(qc)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=layout)
    qc = pm.run(qc)
    return qc

def append_with_barrier(c1, c2):
    if c1.num_qubits != c2.num_qubits:
        raise ValueError("Circuits must have the same number of qubits to be appended")
    c1 = c1.copy()
    c1.barrier()
    return c1.compose(c2)

def measure_all(circuit, add_barrier=True):
    circuit = circuit.copy()
    if circuit.num_clbits < circuit.num_qubits:
        new_creg = circuit._create_creg(circuit.num_qubits, "meas")
        circuit.add_register(new_creg)
    if add_barrier:
        circuit.barrier()
    circuit.measure(circuit.qubits, circuit.clbits[0:circuit.num_qubits])
    return circuit

def multiply_last_simdepth_cal_circuits(sim_circuits, cal_circuits):
    sim_circuits = numpy_array_of_circuits(sim_circuits)
    cal_circuits = numpy_array_of_circuits(cal_circuits)
    if sim_circuits.shape != cal_circuits.shape:
        raise ValueError("sim_circuits and cal_circuits must have the same length.")
    if sim_circuits.ndim != 1:
        raise ValueError("sim_circuits and cal_circuits must be 1-dimensional arrays.")
    sim_depths = np.array([depth2qb(c) for c in sim_circuits])
    cal_depths = np.array([depth2qb(c) for c in cal_circuits])
    if not np.all(np.sort(sim_depths) == sim_depths):
        raise ValueError("sim_circuits must be ordered by increasing depth.")
    if not np.all(np.sort(cal_depths) == cal_depths):
        raise ValueError("cal_circuits must be ordered by increasing depth.")
    first_max_depth_ind = np.argmax(cal_depths >= sim_depths[-1])
    nrepetitions = len(cal_circuits) - first_max_depth_ind
    if nrepetitions > 0:
        cal_circuits[first_max_depth_ind:] = numpy_array_of_circuits([cal_circuits[first_max_depth_ind]]*nrepetitions)
    return cal_circuits

def choose_layout_from_longest_circuit(backend, circuits):
    circuits = numpy_array_of_circuits(circuits).flatten()
    depths = np.array([depth2qb(c) for c in circuits.flatten()])
    longest_circuit_ind = np.argmax(depths)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled_longest_circuit = pm.run(circuits[longest_circuit_ind])
    return transpiled_longest_circuit.layout.final_index_layout()