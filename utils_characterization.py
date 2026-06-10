from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple, Union
import copy
import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from pandas import DataFrame

from qiskit import QuantumCircuit
from qiskit.result import marginal_counts as mcts

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.models import BackendProperties

from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_experiments.library import StandardRB
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity

# ---------------------- Helper functions (module-level) ---------------------- #
def _build_readout_circuits(num_qubits: int) -> List[QuantumCircuit]:
    """Create spam0/spam1 circuits over all qubits."""
    spam0 = QuantumCircuit(num_qubits, num_qubits)
    spam0.measure_all()
    spam1 = QuantumCircuit(num_qubits, num_qubits)
    spam1.x(range(num_qubits))
    spam1.measure_all()
    return [spam0, spam1]


def _select_twoq_gate(backend: IBMBackend) -> str:
    """Prefer ecr -> cz -> cx."""
    basis = backend.configuration().basis_gates
    if "ecr" in basis:
        return "ecr"
    if "cz" in basis:
        return "cz"
    return "cx"


def _oneq_basis_excluding(twoq_gate: str, backend: IBMBackend) -> List[str]:
    """One-qubit basis gates excluding the chosen two-qubit gate; skip rx/rzz."""
    oneq = []
    for g in backend.configuration().basis_gates:
        if g.casefold() in {"rx", "rzz"}:
            continue
        if g.casefold() != twoq_gate.casefold():
            oneq.append(g)
    return oneq


def _build_oneq_rb_experiments(
    backend: IBMBackend,
    lengths: np.ndarray,
    num_samples: int,
    seed: int,
    samples_m: int,
) -> BatchExperiment:
    """Parallel + Batch StandardRB across independent qubit sets."""
    G = backend.coupling_map.graph.to_undirected(multigraph=False)
    sqrb_batches = _get_iso_qubit_list(G)

    sqrb_exp_list: List[ParallelExperiment] = []
    for batch in sqrb_batches:
        rb1q_exps = []
        for q in batch:
            rb1q_exps.append(
                StandardRB(
                    physical_qubits=[int(q)],
                    lengths=lengths,
                    backend=backend,
                    seed=seed,
                    num_samples=num_samples,
                )
            )
        sqrb_exp_list.append(ParallelExperiment(rb1q_exps, backend=backend, flatten_results=True))

    sqrb_exp = BatchExperiment(sqrb_exp_list, backend=backend, flatten_results=True)
    sqrb_exp.set_experiment_options(separate_jobs=True)
    sqrb_exp.experiment_options.max_circuits = samples_m * num_samples
    return sqrb_exp


def _build_layered_rb_experiments(
    backend: IBMBackend,
    lengths: List[int],
    num_samples: int,
    seed: int,
    max_circuits: int,
) -> Tuple[LayerFidelity, LayerFidelity, List[List[Tuple[int, int]]]]:
    """Layered RB (LayerFidelity) for horizontal & vertical grids; returns (lf_h, lf_v, layers)."""
    twoq_gate = _select_twoq_gate(backend)
    oneq_gates = _oneq_basis_excluding(twoq_gate, backend)

    grid_chains = _get_grids(backend)
    coupling_map = backend.coupling_map
    edges = list(backend.target[twoq_gate].keys())

    layers: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
    grid_chain_flt = [[], []]

    for i in range(2):  # 0=H, 1=V
        all_pairs = _path_to_edges(grid_chains[i], coupling_map)
        for j, pair_lst in enumerate(all_pairs):
            grid_chain_flt[i] += grid_chains[i][j]
            sub_pairs = [tuple(p) if tuple(p) in edges else tuple(p)[::-1] for p in pair_lst]
            layers[2 * i] += sub_pairs[0::2]
            layers[2 * i + 1] += sub_pairs[1::2]

    h_qubits = grid_chain_flt[0]
    v_qubits = grid_chain_flt[1]

    lf_h = LayerFidelity(
        physical_qubits=h_qubits,
        two_qubit_layers=[layers[0], layers[1]],
        lengths=lengths,
        backend=backend,
        num_samples=num_samples,
        seed=seed,
        two_qubit_gate=twoq_gate,
        one_qubit_basis_gates=oneq_gates,
    )
    lf_v = LayerFidelity(
        physical_qubits=v_qubits,
        two_qubit_layers=[layers[2], layers[3]],
        lengths=lengths,
        backend=backend,
        num_samples=num_samples,
        seed=seed,
        two_qubit_gate=twoq_gate,
        one_qubit_basis_gates=oneq_gates,
    )

    lf_h.experiment_options.max_circuits = max_circuits
    lf_v.experiment_options.max_circuits = max_circuits

    return lf_h, lf_v, layers


def _get_readout_errors(num_qubits: int, readout_result) -> Dict[int, float]:
    """Per-qubit readout error from spam0/spam1 marginal counts."""
    ro_error: Dict[int, float] = {}
    cts_spam0 = readout_result[0].data.meas.get_counts()
    cts_spam1 = readout_result[1].data.meas.get_counts()
    spam_shots = readout_result[0].data.meas.num_shots
    for q in range(num_qubits):
        try:
            ro_error[q] = 1 - ((mcts(cts_spam0, [q])["0"] + mcts(cts_spam1, [q])["1"]) / 2) / spam_shots
        except KeyError:
            ro_error[q] = 1 - (mcts(cts_spam0, [q])["0"] / 2) / spam_shots
    return ro_error


def _get_oneq_errors(job_results_df: DataFrame) -> Dict[Union[int, Tuple[int, int]], float]:
    """Map qiskit-experiments analysis DataFrame to {component: value} (1q EPG)."""
    values_map: Dict[Union[int, Tuple[int, int]], float] = {}
    for _, row in job_results_df.iterrows():
        if len(row.components) > 1:
            q1 = row.components[0].index
            q2 = row.components[1].index
            key: Union[int, Tuple[int, int]] = (q1, q2)
        else:
            key = row.components[0].index
        values_map[key] = row.value.nominal_value
    return values_map


def _get_twoq_errors(backend: IBMBackend, lf_result_h_df: DataFrame, lf_result_v_df: DataFrame, layers: List[List[Tuple[int, int]]]) -> Dict[str, float]:
    """Combine LF H/V DataFrames into a pair->error map using project utilities."""
    twoq_gate = _select_twoq_gate(backend)
    lf_err_dict = _make_error_dict(backend, twoq_gate)

    updated_err_dicts = []
    for i, lf_df in enumerate([lf_result_h_df, lf_result_v_df]):
        for j in range(2):
            updated_err_dicts.append(_df_to_error_dict(lf_df, layers[2 * i + j]))

    lf_err_dict = _update_error_dict(lf_err_dict, updated_err_dicts)
    return lf_err_dict


def _update_readout_props(props: Dict, error_map: Dict[int, float]) -> Dict:
    """Patch per-qubit readout_error values."""
    qubits = list(error_map.keys())
    for i in range(len(qubits)):
        for param in props["qubits"][i]:
            if param["name"] == "readout_error": # Check none here
                param["value"] = error_map[i]
                break
    return props


def _update_1q_props(props: Dict, error_map: Dict[Union[int, Tuple[int, int]], float]) -> Dict:
    """Patch 1q gate_error for sx/x from EPG maps (unchanged logic)."""
    for ix, gate in enumerate(props["gates"]):
        if len(gate["qubits"]) > 1:
            x = re.search(r"^([a-zA-Z]+)([0-9]+_[0-9]+)", gate["name"])
            gate_type, component = x[1], (int(x[2].split("_")[0]), int(x[2].split("_")[1]))
        else:
            x = re.search(r"^([a-zA-Z]+)([0-9]+)", gate["name"])
            gate_type, component = x[1], int(x[2])

        if gate_type in ("sx", "x"):
            for iy, parameter in enumerate(gate["parameters"]):
                if parameter["name"] == "gate_error" and component in error_map:
                    gate_error = error_map[component]
                    props["gates"][ix]["parameters"][iy]["value"] = gate_error
                    break
    return props


def _update_lf_props(props: Dict, error_map: Dict[str, float]) -> Dict:
    """Patch 2q gate_error for {cz,cx,ecr} from LF pair->error map (unchanged logic)."""
    for ix, gate in enumerate(props["gates"]):
        if gate["gate"] in ["cz", "cx", "ecr"]:
            q0, q1 = gate["qubits"]
            pair_key = f"{q0}_{q1}"
            pair_key_rev = f"{q1}_{q0}"
            if pair_key in error_map or pair_key_rev in error_map:
                gate_error = error_map.get(pair_key, error_map.get(pair_key_rev))
                for iy, param in enumerate(gate["parameters"]):
                    if param["name"] == "gate_error":
                        props["gates"][ix]["parameters"][iy]["value"] = float(gate_error)
                        break
    return props


def _path_to_edges(paths: List[List[int]], coupling_map=None):
    """Convert a list of paths into a list of edges that are in the
    coupling_map if defined

    If already edges (length 2 path) then convert into the edge that's in the
    coupling map

    Args:
        paths: list of qubit chains

    Returns:
        list of qubit paths in terms of the edges to traverse.
    """

    new_path = []
    for path_i in paths:
        if len(path_i) > 2:
            new_path.append([])

        for i in range(len(path_i) - 1):
            tmp_set = path_i[i : (i + 2)]
            if coupling_map is not None:
                if tuple(tmp_set) not in coupling_map and tmp_set not in coupling_map:
                    tmp_set.reverse()
                    if (
                        tuple(tmp_set) not in coupling_map
                        and tmp_set not in coupling_map
                    ):
                        raise ValueError("Path not found in coupling map")

            if len(path_i) > 2:
                new_path[-1].append(tmp_set)
            else:
                new_path.append(tmp_set)

    return new_path


def _get_iso_qubit_list(G):
    """Return a set of lists of isolated (separated by at least one idle qubit)
    qubits using graph coloring

    Args:
        G: system graph (assume G.to_undirected(multigraph=False) has been run)

    Returns:
        list of qubit lists
    """

    qlists = {}
    node_dict = rx.graph_greedy_color(G)
    for i in node_dict:
        if node_dict[i] in qlists:
            qlists[node_dict[i]].append(i)
        else:
            qlists[node_dict[i]] = [i]

    qlists = list(qlists.values())
    for i in range(len(qlists)):
        qlists[i] = list(np.sort(qlists[i]))

    return qlists


def _make_error_dict(backend: IBMBackend, two_q_gate: str, keep_perm: bool = False):
    """Convert a backend into a dictionary of errors for each edge

    Args:
    - backend: ibm backend
    - two_q_gate: the two qubit gate type
    - keep_perm: include the errors if both directions of the gate
    appear in the coupling map

    Return:
    - error dictionary
    """

    err_dict = {}
    props = backend.properties()
    coup_map = backend.coupling_map

    for i in coup_map:
        if not keep_perm:
            if "%d_%d" % (i[1], i[0]) in err_dict:
                continue

        err_dict["%d_%d" % (i[0], i[1])] = props.gate_error(two_q_gate, i)

    return err_dict

def _df_to_error_dict(lf_df: DataFrame, gate_list: list):
    """Take in the dataframe from the layer fidelity experiment
    and convert it to a dictionary of gate errors

    Args:
    - lf_df: Dataframe from layer fidelity experiment
    - gate_list: List of gates in the layer fidelity experiment

    Return:
    - error dictionary
    """

    pfdf = lf_df[lf_df.name == "ProcessFidelity"]
    err_rb_dict = {}

    for gate in gate_list:
        gate_pf = pfdf[pfdf.qubits == tuple(gate)].iloc[0].value.nominal_value

        err_rb_dict["%d_%d" % (gate[0], gate[1])] = 4 / 5 * (1 - gate_pf)

    return err_rb_dict

def _update_error_dict(
    error_dict: dict,
    new_error_dicts: dict,
    update_perm: bool = True,
):
    """Update errors in error_dict with the errors from the
    new_error_dict

    Args:
    - error_dict: original error dictionary (assume complete)
    - new_error_dicts: list of new error dictionaries
    - update_perm: treat Q0_Q1 the same as Q1_Q0

    Return:
    - updated error dictionary
    """

    for gate in error_dict:
        updated_errs = []
        for i in new_error_dicts:
            if gate in i:
                updated_errs.append(i[gate])

            if update_perm:
                rgate = "%s_%s" % (gate.split("_")[1], gate.split("_")[0])
                if rgate in i:
                    updated_errs.append(i[rgate])

        if len(updated_errs) > 0:
            error_dict[gate] = float(np.mean(updated_errs))

    return error_dict

def _get_grids(
    backend: IBMBackend,
):
    """Return the grid chains for the different backends

    Args:
    - backend: ibm backend

    Return:
    - list of list of chains: list of list of chains for the grids to run. Each
    sub list is a list of chains that can be run in one experiment, each list in that sublist
    is a continuous path
    """

    grid_chains = []

    if backend.num_qubits == 127:
        horz_chain = [[13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 52, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 90, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 112, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113]]
        vert_chain = [[113, 114, 109, 96, 95, 94, 90, 75, 76, 77, 71, 58, 57, 56, 52, 37, 38, 39, 33, 20, 19, 18, 14, 0, 1, 2, 3, 4, 15, 22, 23, 24, 34, 43, 42, 41, 53, 60, 61, 62, 72, 81, 80, 79, 91, 98, 99, 100, 110, 118, 119, 120, 121, 122, 111, 104, 103, 102, 92, 83, 84, 85, 73, 66, 65, 64, 54, 45, 46, 47, 35, 28, 27, 26, 16, 8, 9, 10, 11, 12, 17, 30, 31, 32, 36, 51, 50, 49, 55, 68, 69, 70, 74, 89, 88, 87, 93, 106, 107, 108, 112, 126]]
    elif backend.num_qubits == 133:
        horz_chain = [[14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 91, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 113, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 129]]
        vert_chain = [[129, 114, 115, 116, 110, 97, 96, 95, 91, 76, 77, 78, 72, 59, 58, 57, 53, 38, 39, 40, 34, 21, 20, 19, 15, 0, 1, 2, 3, 4, 16, 23, 24, 25, 35, 44, 43, 42, 54, 61, 62, 63, 73, 82, 81, 80, 92, 99, 100, 101, 111, 120, 119, 118, 130],
                      [131, 122, 123, 124, 112, 105, 104, 103, 93, 84, 85, 86, 74, 67, 66, 65, 55, 46, 47, 48, 36, 29, 28, 27, 17, 8, 9, 10, 11, 12, 18, 31, 32, 33, 37, 52, 51, 50, 56, 69, 70, 71, 75, 90, 89, 88, 94, 107, 108, 109, 113, 128, 127, 126, 132]]
    elif backend.num_qubits == 156:
        horz_chain = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20],
                      [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60],
                      [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 99, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100],
                      [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 139, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140]]
        vert_chain = [[0, 1, 2, 3, 16, 23, 22, 21, 36, 41, 42, 43, 56, 63, 62, 61, 76, 81, 82, 83, 96, 103, 102, 101, 116, 121, 122, 123, 136, 143, 142, 141, 140],
                      [5, 6, 7, 17, 27, 26, 25, 37, 45, 46, 47, 57, 67, 66, 65, 77, 85, 86, 87, 97, 107, 106, 105, 117, 125, 126, 127, 137, 147, 146, 145],
                      [9, 10, 11, 18, 31, 30, 29, 38, 49, 50, 51, 58, 71, 70, 69, 78, 89, 90, 91, 98, 111, 110, 109, 118, 129, 130, 131, 138, 151, 150, 149],
                      [13, 14, 15, 19, 35, 34, 33, 39, 53, 54, 55, 59, 75, 74, 73, 79, 93, 94, 95, 99, 115, 114, 113, 119, 133, 134, 135, 139, 155, 154, 153]]
    elif backend.num_qubits == 120:
        horz_chain = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59], [60, 61, 62, 63, 64, 65, 66, 67, 68, 69], [70, 71, 72, 73, 74, 75, 76, 77, 78, 79], [80, 81, 82, 83, 84, 85, 86, 87, 88, 89], [90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [100, 101, 102, 103, 104, 105, 106, 107, 108, 109], [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]]
        vert_chain = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111], [2, 12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112], [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113], [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114], [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115], [6, 16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116], [7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117], [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118], [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119]]
    else:
        raise ValueError("Current backend with %d qubits is missing a grid configuration" % backend.num_qubits)

    grid_chains = [horz_chain, vert_chain]

    return grid_chains



# --------------------------------- Main Function -------------------------------- #
def characterize_backend(
    backend: IBMBackend,
    experiments: Optional[Iterable[str]] = None,
    shots: Optional[Dict[str, int]] = None,
    verbose: bool = True,
):

    """
    Run selected characterization experiments on the given IBM Quantum backend and update its properties.

    Preserved behavior (unchanged from your code):
      - Experiments:
          * "readout": spam0/spam1 over all qubits
          * "rb_1q": StandardRB per qubit, batched via Parallel+BatchExperiment
          * "rb_2q": LayerFidelity on horizontal & vertical grid layers
      - Shots defaults: readout=10000, rb_1q=250, rb_2q=250
      - RB(1q): lengths=[1,50,100,500,1000,3000], num_samples=6, seed=42,
                separate_jobs=True, max_circuits=3 * num_samples
      - LF(2q): lengths=[1,10,20,30,40,60,80,100,150,200,400], num_samples=12,
                seed=60, max_circuits=144
      - Two-qubit gate preference: ecr -> cz -> cx
      - Updates 'readout_error', 1q 'sx'/'x' gate_error (EPG), and 2q gate_error from LF
      - Properties are patched on the provided backend (private attribute _properties)

    Args:
        backend: IBM Quantum backend compatible with qiskit-ibm-runtime SamplerV2.
        experiments: Optional subset of {"readout", "rb_1q", "rb_2q"}; defaults to all if None.
        shots: Optional dict overriding shots, keys in {"readout","rb_1q","rb_2q"}.

    Returns:
        The same backend instance with its BackendProperties updated in-place.
    """

    # ---- Local configuration (no globals) ----
    allowed_experiments = {"readout", "rb_1q", "rb_2q"}

    rb1q_lengths = np.array([1, 50, 100, 500, 1000, 3000])
    rb1q_num_samples = 6
    rb1q_seed = 42
    rb1q_samples_m = 3  # => max_circuits = 18

    lf_lengths = [1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400]
    lf_num_samples = 12
    lf_seed = 60
    lf_max_circuits = 144

    default_shots = {"readout": 10_000, "rb_1q": 250, "rb_2q": 250}
    shots = {**default_shots, **(shots or {})}

    chosen = set(experiments) if experiments is not None else allowed_experiments
    unknown = chosen - allowed_experiments
    if unknown:
        raise ValueError(f"Unsupported experiments: {sorted(unknown)}. Allowed: {allowed_experiments}")

    # Lightweight printer
    def _emit(msg: str):
        if verbose:
            print(msg, flush=True)

    # ---- Build experiments as requested ----
    sampler = Sampler(mode=backend)
    sampler.options.environment.job_tags = [f"callibration"]

    readout_circuits: Optional[List[QuantumCircuit]] = None
    if "readout" in chosen:
        readout_circuits = _build_readout_circuits(backend.num_qubits)

    sqrb_exp: Optional[BatchExperiment] = None
    if "rb_1q" in chosen:
        sqrb_exp = _build_oneq_rb_experiments(
            backend,
            lengths=rb1q_lengths,
            num_samples=rb1q_num_samples,
            seed=rb1q_seed,
            samples_m=rb1q_samples_m,
        )

    lf_h = lf_v = None
    layers: Optional[List[List[Tuple[int, int]]]] = None
    if "rb_2q" in chosen:
        lf_h, lf_v, layers = _build_layered_rb_experiments(
            backend,
            lengths=lf_lengths,
            num_samples=lf_num_samples,
            seed=lf_seed,
            max_circuits=lf_max_circuits,
        )

    # ---- Run and collect results ----
    if readout_circuits:
        sampler.options.default_shots = int(shots["readout"])
        job_readout = sampler.run(readout_circuits)
        # Verbose message for readout submission
        try:
            _emit(f"Readout job submitted: job_id = {job_readout.job_id()}")
        except Exception:
            _emit("Readout job submitted: job_id = None")

    if sqrb_exp:
        sampler.options.default_shots = int(shots["rb_1q"])
        job_sqrb = sqrb_exp.run(sampler=sampler)
        # Verbose message for 1q RB submission
        job_ids = getattr(job_sqrb, "job_ids", None)
        _emit(f"Single-qubit RB submitted: job ids = {job_ids if job_ids is not None else 'None'}")


    if lf_h and lf_v:
        sampler.options.default_shots = int(shots["rb_2q"])
        job_lf_v = lf_v.run(sampler=sampler)
        job_lf_h = lf_h.run(sampler=sampler)
        # Verbose message for layered 2q RB submission
        job_ids_v = getattr(job_lf_v, "job_ids", None)
        job_ids_h = getattr(job_lf_h, "job_ids", None)
        # Flatten/merge to one list-like representation 
        merged_ids = []
        if job_ids_v is not None:
            merged_ids.extend(job_ids_v if isinstance(job_ids_v, list) else [job_ids_v])
        if job_ids_h is not None:
            merged_ids.extend(job_ids_h if isinstance(job_ids_h, list) else [job_ids_h])
        _emit(f"Layered two-qubit RB submitted: job_ids = {merged_ids if merged_ids else 'None'}")

    # ---- Build error maps ----
    ro_error_map: Optional[Dict[int, float]] = None
    if readout_circuits:
        readout_result = job_readout.result()
        ro_error_map = _get_readout_errors(backend.num_qubits, readout_result)

    oneq_err_x = oneq_err_sx = None
    if sqrb_exp:
        sqrb_result_x = job_sqrb.analysis_results("EPG_x", dataframe=True)
        oneq_err_x = _get_oneq_errors(sqrb_result_x)
        sqrb_result_sx = job_sqrb.analysis_results("EPG_sx", dataframe=True)
        oneq_err_sx = _get_oneq_errors(sqrb_result_sx)

    lf_err_map: Optional[Dict[str, float]] = None
    if lf_h and lf_v:
        lf_result_v = job_lf_v.analysis_results("ProcessFidelity", dataframe=True)
        lf_result_h = job_lf_h.analysis_results("ProcessFidelity", dataframe=True)
        lf_err_map = _get_twoq_errors(backend, lf_result_h, lf_result_v, layers)

    # ---- Patch properties in place (unchanged behavior) ----
    backend = copy.deepcopy(backend)
    props_dict = backend.properties().to_dict()
    updated_sections = []

    if ro_error_map is not None:
        props_dict = _update_readout_props(props_dict, ro_error_map)
        updated_sections.append("readout")
    if oneq_err_x is not None or oneq_err_sx is not None:
        if oneq_err_x is not None:
            props_dict = _update_1q_props(props_dict, oneq_err_x)
        if oneq_err_sx is not None:
            props_dict = _update_1q_props(props_dict, oneq_err_sx)
        updated_sections.append("single-qubit errors")
    if lf_err_map is not None:
        props_dict = _update_lf_props(props_dict, lf_err_map)
        updated_sections.append("two-qubit errors")

    if updated_sections:
        _emit(f"Updated backend properties: {', '.join(updated_sections)}")

    props = BackendProperties.from_dict(props_dict)
    backend._properties = props  # intentional: preserved behavior

    return backend




# ----------------------------- Plotting Functions --------------------------------- #

# ----------------------------- Extractors (unchanged) ----------------------------- #
def _extract_readout_errors(props_dict: Dict):
    errors = {}
    for q, q_params in enumerate(props_dict["qubits"]):
        for p in q_params:
            if p["name"] == "readout_error":
                errors[q] = p["value"]
                break
    return errors

def _extract_oneq_gate_errors(props_dict: Dict):
    """
    Returns {qubit: error} for sx/x gates.
    If both exist, sx is preferred; otherwise x is used.
    """
    oneq_errors = {}
    for gate in props_dict["gates"]:
        if len(gate["qubits"]) == 1:
            q = gate["qubits"][0]
            gname = gate["gate"]
            if gname not in ("sx", "x"):
                continue
            for p in gate["parameters"]:
                if p["name"] == "gate_error":
                    prev = oneq_errors.get(q)
                    if prev is None or gname == "sx":
                        oneq_errors[q] = p["value"]
                    break
    return oneq_errors

def _extract_twoq_gate_errors(props_dict: Dict):
    """
    Returns {(q0,q1): error} for 2q gates (cx/cz/ecr).
    """
    errors = {}
    for gate in props_dict["gates"]:
        if len(gate["qubits"]) == 2 and gate["gate"] in ("cx", "cz", "ecr"):
            q0, q1 = gate["qubits"]
            for p in gate["parameters"]:
                if p["name"] == "gate_error":
                    errors[(q0, q1)] = p["value"]
                    break
    return errors

# ----------------------------- Sorting (unchanged) ----------------------------- #
def _sorted_union_keys(old_map: Dict, new_map: Dict, sort_by="new"):
    """
    Returns keys sorted by:
      - "new": sort by new_map value
      - "old": sort by old_map value
      - "max": sort by max(old, new) per key
    NaNs are sent to the end.
    """
    keys = sorted(set(old_map.keys()) | set(new_map.keys()))
    def _val(k, which):
        if which == "new":
            return new_map.get(k, np.nan)
        elif which == "old":
            return old_map.get(k, np.nan)
        elif which == "max":
            v_old = old_map.get(k, np.nan)
            v_new = new_map.get(k, np.nan)
            return np.nanmax([v_old, v_new])
        else:
            raise ValueError("sort_by must be 'new', 'old', or 'max'")
    return sorted(keys, key=lambda k: (np.isnan(_val(k, sort_by)), _val(k, sort_by)))

# ----------------------------- Style helpers (unchanged) ----------------------------- #
def _set_every_xtick_with_vertical_guides(ax, labels, *, rotation=90, fontsize=7, weight="regular"):
    """
    Keep ALL ticks, rotate labels (default 90°), shrink font,
    and draw a faint vertical gridline at EVERY tick position.
    """
    n = len(labels)
    idxs = np.arange(n)
    ax.set_xticks(idxs)
    ax.set_xticklabels(labels)  # all labels

    for t in ax.get_xticklabels():
        t.set_rotation(rotation)
        t.set_fontsize(fontsize)
        t.set_fontweight(weight)
        if rotation == 90:
            t.set_horizontalalignment("center")
            t.set_verticalalignment("top")
        elif rotation == 45:
            t.set_horizontalalignment("right")
            t.set_verticalalignment("top")
        else:
            t.set_horizontalalignment("center")

    ax.tick_params(axis="x", which="both", width=0.6, length=3)

    for i in idxs:
        ax.axvline(i, color="k", alpha=0.08, linewidth=1.0, zorder=0)

    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.28 if rotation == 90 else 0.20)

def _plot_lines(ax, x_idx, y_old, y_new, *, lw=1.1, ms=3.0):
    ax.plot(x_idx, y_old, color="tab:red",  marker="o", linewidth=lw, markersize=ms, label="Error (reported)")
    ax.plot(x_idx, y_new, color="tab:blue", marker="o", linewidth=lw, markersize=ms, label="Error (measured)")

# ----------------------------- NEW: sort label mapper ----------------------------- #
def _sort_label(sort_by: str) -> str:
    """
    Convert internal sort key to human-friendly label.
      'new' -> 'measured'
      'old' -> 'reported'
      'max' -> 'max(measured, reported)'
    """
    mapping = {
        "new": "measured",
        "old": "reported",
        "max": "max(measured, reported)",
    }
    try:
        return mapping[sort_by]
    except KeyError:
        raise ValueError("sort_by must be 'new', 'old', or 'max'")

# ----------------------------- Main plotter (titles updated) ----------------------------- #
def plot_characterization_comparison(
    old_props: Dict,
    new_props: Dict,
    sort_by="new",              # "new" | "old" | "max"
    title_prefix=None,          # optional prefix (e.g., backend name)
):
    """
    Create three comparison plots with ALL tick labels kept and styled:
      1) Readout error (linear y)
      2) Single-qubit RB EPG (sx/x) (log y)
      3) Layered 2Q RB EPG (log y)

    Titles now say: 'sorted by measured' (or 'reported' / 'max(measured, reported)').
    """
    prefix = (title_prefix + " – ") if title_prefix else ""
    sort_friendly = _sort_label(sort_by)

    # ---------------- FORMAT -----------------

    # Figure sizes
    readout_figsize=(18, 5.5)
    oneq_figsize=(18, 5.5)
    twoq_figsize=(22, 6.5)
    # Label styling
    readout_label_fontsize=7
    oneq_label_fontsize=7
    twoq_label_fontsize=7
    readout_rotation=90
    oneq_rotation=90
    twoq_rotation=90
    # Lines
    line_width=1.1
    marker_size=3.0

    # ---------------- READOUT ----------------
    old_ro = _extract_readout_errors(old_props)
    new_ro = _extract_readout_errors(new_props)
    keys_ro = _sorted_union_keys(old_ro, new_ro, sort_by=sort_by)
    if keys_ro:
        y_old = [old_ro.get(k, np.nan) for k in keys_ro]
        y_new = [new_ro.get(k, np.nan) for k in keys_ro]
        x_idx = np.arange(len(keys_ro))

        fig, ax = plt.subplots(figsize=readout_figsize)
        ax.set_title(f"{prefix}Readout Error — measured vs reported (sorted by {sort_friendly})")
        _plot_lines(ax, x_idx, y_old, y_new, lw=line_width, ms=marker_size)
        ax.set_xlabel("Qubit")
        ax.set_ylabel("Readout error")
        _set_every_xtick_with_vertical_guides(
            ax, [str(k) for k in keys_ro],
            rotation=readout_rotation, fontsize=readout_label_fontsize
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()

    # ---------------- 1Q RB (sx/x) ----------------
    old_1q = _extract_oneq_gate_errors(old_props)
    new_1q = _extract_oneq_gate_errors(new_props)
    keys_1q = _sorted_union_keys(old_1q, new_1q, sort_by=sort_by)
    if keys_1q:
        y_old = [old_1q.get(k, np.nan) for k in keys_1q]
        y_new = [new_1q.get(k, np.nan) for k in keys_1q]
        x_idx = np.arange(len(keys_1q))

        fig, ax = plt.subplots(figsize=oneq_figsize)
        ax.set_title(f"{prefix}1Q RB EPG (sx/x) — measured vs reported (sorted by {sort_friendly})")
        _plot_lines(ax, x_idx, y_old, y_new, lw=line_width, ms=marker_size)
        ax.set_xlabel("Qubit")
        ax.set_ylabel("EPG (1Q)")
        ax.set_yscale("log")
        _set_every_xtick_with_vertical_guides(
            ax, [str(k) for k in keys_1q],
            rotation=oneq_rotation, fontsize=oneq_label_fontsize
        )
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()

    # ---------------- 2Q RB (layered) ----------------
    old_2q = _extract_twoq_gate_errors(old_props)
    new_2q = _extract_twoq_gate_errors(new_props)
    keys_2q = _sorted_union_keys(old_2q, new_2q, sort_by=sort_by)
    if keys_2q:
        y_old = [old_2q.get(k, np.nan) for k in keys_2q]
        y_new = [new_2q.get(k, np.nan) for k in keys_2q]
        labels = [f"{i}-{j}" for (i, j) in keys_2q]
        x_idx = np.arange(len(keys_2q))

        fig, ax = plt.subplots(figsize=twoq_figsize)
        ax.set_title(f"{prefix}Layered 2Q RB EPG — measured vs reported (sorted by {sort_friendly})")
        _plot_lines(ax, x_idx, y_old, y_new, lw=line_width, ms=marker_size)
        ax.set_xlabel("2Q gate")
        ax.set_ylabel("EPG (2Q)")
        ax.set_yscale("log")
        _set_every_xtick_with_vertical_guides(
            ax, labels,
            rotation=twoq_rotation, fontsize=twoq_label_fontsize
        )
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()