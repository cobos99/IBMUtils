from qiskit_experiments.library import T1, T2Hahn, LocalReadoutError, StandardRB
from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from collections import defaultdict
import rustworkx

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