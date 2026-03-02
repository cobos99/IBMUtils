from qiskit_experiments.library import T1, T2Hahn, LocalReadoutError, StandardRB
from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit.transpiler import InstructionProperties
from qiskit_ibm_runtime import SamplerV2
from collections import defaultdict
import numpy as np
import rustworkx
import copy

def ibm_metrics_experiments(backend):
    qubits = list(range(backend.num_qubits))
    
    # Get layered couplig map
    coupling_graph = backend.coupling_map.graph.to_undirected(multigraph=False)

    # Get layered coupling map
    edge_coloring = rustworkx.graph_bipartite_edge_color(coupling_graph)
    layered_coupling_map = defaultdict(list)
    for edge_idx, color in edge_coloring.items():
        layered_coupling_map[color].append(
            coupling_graph.get_edge_endpoints_by_index(edge_idx)
        )
    layered_coupling_map = [
        sorted(layered_coupling_map[i])
        for i in sorted(layered_coupling_map.keys())
    ]

    # Get unidirectional coupling map
    one_dir_coupling_map = coupling_graph.edge_list()
 
    # Get unidirectional coupling map
    one_dir_coupling_map = coupling_graph.edge_list()
    # Create T1 experiments on all qubit in parallel
    t1_exp = ParallelExperiment(
        [
            T1(
                physical_qubits=[qubit],
                delays=[1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6],
            )
            for qubit in qubits
        ],
        backend,
        analysis=None,
    )
    
    # Create T2-Hahn experiments on all qubit in parallel
    t2_exp = ParallelExperiment(
        [
            T2Hahn(
                physical_qubits=[qubit],
                delays=[1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6],
            )
            for qubit in qubits
        ],
        backend,
        analysis=None,
    )
    
    # Create readout experiments on all qubit in parallel
    readout_exp = LocalReadoutError(qubits)
    
    # Create single-qubit RB experiments on all qubit in parallel
    singleq_rb_exp = ParallelExperiment(
        [
            StandardRB(
                physical_qubits=[qubit], lengths=[10, 100, 500], num_samples=10
            )
            for qubit in qubits
        ],
        backend,
        analysis=None,
    )
    
    # Create two-qubit RB experiments on the three layers of disjoint edges of the heavy-hex
    twoq_rb_exp_batched = BatchExperiment(
        [
            ParallelExperiment(
                [
                    StandardRB(
                        physical_qubits=pair,
                        lengths=[10, 50, 100],
                        num_samples=10,
                    )
                    for pair in layer
                ],
                backend,
                analysis=None,
            )
            for layer in layered_coupling_map
        ],
        backend,
        flatten_results=True,
        analysis=None,
    )

    return [t1_exp, t2_exp, readout_exp, singleq_rb_exp, twoq_rb_exp_batched]

def update_target(backend, session):
    benchmark_batches = ibm_metrics_experiments(backend)
    benchmark_batches_exp = BatchExperiment(benchmark_batches, backend)
    coupling_graph = backend.coupling_map.graph.to_undirected(multigraph=False)
    one_dir_coupling_map = coupling_graph.edge_list()
    instruction_2q_name = "cz"

    # Run benchmarks for circuits
    layout_sampler = SamplerV2(mode=session)

    # Run characterization experiments
    batches_exp_data = benchmark_batches_exp.run(
        sampler=layout_sampler, **{"shots": 1e3, "dynamic": False}
    ).block_for_results()

    EPG_sx_result_list = batches_exp_data.analysis_results("EPG_sx")
    EPG_sx_result_q_indices = [
        result.device_components.index for result in EPG_sx_result_list
    ]
    EPG_x_result_list = batches_exp_data.analysis_results("EPG_x")
    EPG_x_result_q_indices = [
        result.device_components.index for result in EPG_x_result_list
    ]
    T1_result_list = batches_exp_data.analysis_results("T1")
    T1_result_q_indices = [
        result.device_components.index for result in T1_result_list
    ]

    T2_result_list = batches_exp_data.analysis_results("T2")
    T2_result_q_indices = [
        result.device_components.index for result in T2_result_list
    ]

    Readout_result_list = batches_exp_data.analysis_results(
        "Local Readout Mitigator"
    )

    EPG_2q_result_list = batches_exp_data.analysis_results(
        f"EPG_{instruction_2q_name}"
    )

    # Update target properties
    target = copy.deepcopy(backend.target)
    for i in range(target.num_qubits - 1):
        qarg = (i,)

        if qarg in EPG_sx_result_q_indices:
            target.update_instruction_properties(
                instruction="sx",
                qargs=qarg,
                properties=InstructionProperties(
                    error=EPG_sx_result_list[i].value.nominal_value
                ),
            )
        if qarg in EPG_x_result_q_indices:
            target.update_instruction_properties(
                instruction="x",
                qargs=qarg,
                properties=InstructionProperties(
                    error=EPG_x_result_list[i].value.nominal_value
                ),
            )

        err_mat = Readout_result_list.value.assignment_matrix(i)
        readout_assignment_error = (
            err_mat[0, 1] + err_mat[1, 0]
        ) / 2  # average readout error
        target.update_instruction_properties(
            instruction="measure",
            qargs=qarg,
            properties=InstructionProperties(error=readout_assignment_error),
        )

        if qarg in T1_result_q_indices:
            target.qubit_properties[i].t1 = T1_result_list[
                i
            ].value.nominal_value
        if qarg in T2_result_q_indices:
            target.qubit_properties[i].t2 = T2_result_list[
                i
            ].value.nominal_value

    for pair_idx, pair in enumerate(one_dir_coupling_map):
        qarg = tuple(pair)
        try:
            target.update_instruction_properties(
                instruction=instruction_2q_name,
                qargs=qarg,
                properties=InstructionProperties(
                    error=EPG_2q_result_list[pair_idx].value.nominal_value
                ),
            )
        except:
            target.update_instruction_properties(
                instruction=instruction_2q_name,
                qargs=qarg[::-1],
                properties=InstructionProperties(
                    error=EPG_2q_result_list[pair_idx].value.nominal_value
                ),
            )

    return target