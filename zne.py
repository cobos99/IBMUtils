from multiprocessing.shared_memory import SharedMemory
from rustworkx.visualization import graphviz_draw
import hardware.utils.circs as circutils
import hardware.utils.misc as miscutils
from IPython.display import display
from scipy.optimize import curve_fit
from scipy.stats import linregress
from dataclasses import dataclass
import matplotlib.pyplot as plt
import multiprocessing as mp
import rustworkx as rx
import numpy as np

def couplings_to_undirected_graph(couplings):
    edges = couplings.edges()
    edges_inds = np.sort(np.array(couplings.edge_list()), axis=1)
    un_edgeinds, inds = np.unique(edges_inds, axis=0, return_index=True)
    un_edges = [edges[i] for i in inds]
    graph = rx.PyGraph()
    graph.add_nodes_from(couplings.node_indices())
    graph.add_edges_from((*einds, e) for einds, e in zip(un_edgeinds, un_edges))
    return graph

def find_minimum_cycle_length(backend):
    backend_graph = couplings_to_undirected_graph(backend.coupling_map.graph)
    for i in range(2, backend.num_qubits+1):
        cycle_graph = rx.generators.cycle_graph(i)
        vf2_it = rx.vf2_mapping(backend_graph, cycle_graph, subgraph=True, id_order=False, call_limit=100000)
        try:
            next(vf2_it)
            return i
        except StopIteration:
            continue
    raise ValueError("No cycles found in the backend coupling graph.")

def find_graph_layouts(backend, graph, call_limit=500000):
    hardware_graph = couplings_to_undirected_graph(backend.coupling_map.graph)
    vf2_it = rx.vf2_mapping(hardware_graph, graph, subgraph=True, id_order=False, call_limit=call_limit)
    layouts = []
    for mapping in vf2_it:
        layout = {chain_node_ind: hardware_node_ind for hardware_node_ind, chain_node_ind in mapping.items()}
        layouts.append(layout)
    return layouts

def find_graph_layout_indices(backend, graph, call_limit=500000):
    hardware_graph = couplings_to_undirected_graph(backend.coupling_map.graph)
    vf2_it = rx.vf2_mapping(hardware_graph, graph, subgraph=True, id_order=False, call_limit=call_limit)
    indices = []
    for mapping in vf2_it:
        layout_logical = np.array(list(mapping.values()))
        layout_physical = np.array(list(mapping.keys()))
        mask = np.argsort(layout_logical)
        layout_indices = layout_physical[mask].tolist()
        indices.append(layout_indices)
    return indices

def get_cycles_containing_chain(backend, chain_length):
    minimum_cycle_length = find_minimum_cycle_length(backend)
    required_cycle_length = (chain_length // minimum_cycle_length + 1)*minimum_cycle_length
    cycle_graph = rx.generators.cycle_graph(required_cycle_length)
    cycle_indices = find_graph_layout_indices(backend, cycle_graph)
    _, unique_rolls = np.unique(np.sort(cycle_indices, axis=1), axis=0, return_index=True)
    cycle_indices = [cycle_indices[i] for i in unique_rolls]
    return cycle_indices

def instruction_mean_error_rate(backend_properties, circ_instruction, layout_indices):
    instruction_name = circ_instruction.operation.name
    if instruction_name == "barrier":
        return 0.0
    log_qubit_indices = []
    for q in circ_instruction.qubits:
        log_qubit_indices.append(q._index)
    log_qubit_indices = np.array(log_qubit_indices, dtype=int).squeeze().tolist()
    if np.max(log_qubit_indices) > len(layout_indices) - 1:
        raise ValueError("Qubit index in the circuit exceeds the length of the layout indices.")
    
    npermutations = len(layout_indices)
    gate_errors_sum = 0.0
    for i in range(npermutations):
        cyclic_permuted_phys_qubits = np.roll(layout_indices, i)[log_qubit_indices]
        gate_errors_sum += backend_properties.gate_error(instruction_name, cyclic_permuted_phys_qubits.tolist())
    mean_gate_error = gate_errors_sum / npermutations
    return mean_gate_error

def get_cycle_error_rate(backend_properties, circuit, cycle_indices):
    nqubits = circuit.num_qubits
    if nqubits > len(cycle_indices):
        raise ValueError("Number of qubits in the circuit exceeds the length of the cycle indices.")
    
    instructions_error_rate = 0.0
    for circ_instruction in circuit.data:
        if circ_instruction.operation.name == "barrier":
            continue
        log_qubit_indices = []
        for q in circ_instruction.qubits:
            log_qubit_indices.append(q._index)
        log_qubit_indices = np.array(log_qubit_indices, dtype=int).squeeze().tolist()
        if np.max(log_qubit_indices) > len(cycle_indices) - 1:
            raise ValueError("Qubit index in the circuit exceeds the length of the cycle indices.")
        inst_phys_qubits = cycle_indices[log_qubit_indices]
        instructions_error_rate += backend_properties.gate_error(circ_instruction.operation.name, inst_phys_qubits.tolist())
    return instructions_error_rate

def get_layout_mean_error_rate(backend_properties, circuit, layout_indices):
    nqubits = circuit.num_qubits
    if nqubits > len(layout_indices):
        raise ValueError("Number of qubits in the circuit exceeds the length of the cycle indices.")
    
    instructions_layout_mean_errors = 0.0
    for circ_instruction in circuit.data:
        instructions_layout_mean_errors += instruction_mean_error_rate(backend_properties, circ_instruction, layout_indices)

    return instructions_layout_mean_errors

def get_layout_mean_std_error_rate(backend_properties, circuit, layout_indices):
    nqubits = circuit.num_qubits
    if nqubits > len(layout_indices):
        raise ValueError("Number of qubits in the circuit exceeds the length of the cycle indices.")
    layout_indices = np.array(layout_indices)
    if layout_indices.ndim > 1:
        raise ValueError("layout_indices must be a 1D array-like object.")
    cycles = np.repeat(np.array(layout_indices)[None], len(layout_indices), axis=0)
    cycles = miscutils.roll_rows(cycles, list(range(len(layout_indices))))[:, :nqubits]
    cycles_error_rates = [get_cycle_error_rate(backend_properties, circuit, cycle) for cycle in cycles]
    return np.mean(cycles_error_rates), np.std(cycles_error_rates)

def create_backend_props_shared_memory(backend):
    backend_properties = backend.properties()
    bp_arr = np.array([backend_properties], dtype=object)
    shmem = SharedMemory(create=True, size=bp_arr.nbytes)
    bp_sharr = np.ndarray((1,), dtype=bp_arr.dtype, buffer=shmem.buf)
    bp_sharr[:] = bp_arr[:]
    return shmem

def threadsafe_backend_props_transfer(thread_func, shared_mem_name, *tfargs):
    shmem = SharedMemory(name=shared_mem_name)
    bp_sharr = np.ndarray((1,), dtype=object, buffer=shmem.buf)
    backend_properties = bp_sharr[0]
    return thread_func(backend_properties, *tfargs)

def get_many_layouts_mean_error_rate(backend, circuit, layout_indices_list):
    nqubits = circuit.num_qubits
    for layout_indices in layout_indices_list:
        if nqubits > len(layout_indices):
            raise ValueError("Number of qubits in the circuit exceeds the length of the cycle indices.")
    
    # backend_properties = backend.properties()
    # results = [get_layout_mean_error_rate(backend_properties, circuit, cycle_indices) for cycle_indices in layout_indices_list]

    shmem = create_backend_props_shared_memory(backend)
    with mp.get_context("fork").Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(threadsafe_backend_props_transfer, [(get_layout_mean_error_rate, shmem.name, circuit, cycle_indices) for cycle_indices in layout_indices_list])
    shmem.close()
    shmem.unlink()
    
    return results

def get_many_layouts_mean_std_error_rate(backend, circuit, layout_indices_list):
    nqubits = circuit.num_qubits
    for cycle_indices in layout_indices_list:
        if nqubits > len(cycle_indices):
            raise ValueError("Number of qubits in the circuit exceeds the length of the cycle indices.")
    
    # backend_properties = backend.properties()
    # results = [get_layout_mean_std_error_rate(backend_properties, circuit, cycle_indices) for cycle_indices in layout_indices_list]

    shmem = create_backend_props_shared_memory(backend)
    with mp.get_context("fork").Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(threadsafe_backend_props_transfer, [(get_layout_mean_std_error_rate, shmem.name, circuit, cycle_indices) for cycle_indices in layout_indices_list])
    means = np.array([res[0] for res in results])
    stds = np.array([res[1] for res in results])
    sort_indices = np.argsort(means)
    means = means[sort_indices]
    stds = stds[sort_indices]
    shmem.close()
    shmem.unlink()
    
    return means, stds

def get_many_circuits_mean_error_rate(backend, circuits, layout_indices):
    nqubits_list = [circuit.num_qubits for circuit in circuits]
    max_nqubits = max(nqubits_list)
    if max_nqubits > len(layout_indices):
        raise ValueError("Number of qubits in the circuit exceeds the length of the cycle indices.")
    
    # backend_properties = backend.properties()
    # results = [get_layout_mean_error_rate(backend_properties, circuit, layout_indices) for circuit in circuits]

    shmem = create_backend_props_shared_memory(backend)
    with mp.get_context("fork").Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(threadsafe_backend_props_transfer, [(get_layout_mean_error_rate, shmem.name, circuit, layout_indices) for circuit in circuits])
    shmem.close()
    shmem.unlink()
    
    return results

def choose_layouts_for_zne(backend, circuits, nlayouts, noise_factor=1.5, error_rate_func=get_many_layouts_mean_error_rate, return_cycles=False, return_error_rates=False):
    nqubits_list = [circuit.num_qubits for circuit in circuits]
    max_nqubits = max(nqubits_list)
    circuit_depths = np.array([circutils.depth2qb(c) for c in circuits])
    # min_circuit_depth_index = np.argmin(circuit_depths)
    # mean_circuit_depth_index = np.argmin(np.abs(circuit_depths - np.mean(circuit_depths)))
    max_circuit_depth_index = np.argmax(circuit_depths)
    cycles = get_cycles_containing_chain(backend, max_nqubits)
    # cycles_error_rates = np.array(error_rate_func(backend, circuits[min_circuit_depth_index], cycles))
    # cycles_error_rates = np.array(error_rate_func(backend, circuits[mean_circuit_depth_index], cycles))
    cycles_error_rates = np.array(error_rate_func(backend, circuits[max_circuit_depth_index], cycles))
    cycles_error_rates, unique_indices = np.unique(cycles_error_rates, return_index=True)
    cycles = [cycles[i] for i in unique_indices]
    print(cycles)
    print(cycles_error_rates)
    layouts = [cycles[meind := np.argmin(cycles_error_rates)]]
    cycles.pop(meind)
    cycles_error_rates = np.delete(cycles_error_rates, meind)
    for i in range(1, nlayouts):
        # next_error_rate = min_error_rate + i*(noise_factor - 1)*min_error_rate
        # if next_error_rate > max(cycles_error_rates):
        #     raise ValueError("Not enough layouts found within the noise factor limit.")
        # next_cycle_index = np.argmin(np.abs(next_error_rate - cycles_error_rates))
        # next_cycle_index = i
        next_cycle_index = 0
        layouts.append(cycles[next_cycle_index])
        cycles.pop(next_cycle_index)
        cycles_error_rates = np.delete(cycles_error_rates, next_cycle_index)
    if not return_cycles and not return_error_rates:
        return layouts
    else:
        to_return = [layouts]
        if return_cycles:
            to_return.append(layouts)
        if return_error_rates:
            to_return.append(cycles_error_rates)
        return tuple(to_return)
    
# def choose_layout_by_maxdepth_errorrate(backend, circuits, nlayouts, noise_rate_step=1, error_rate_func=get_many_layouts_mean_error_rate):
#     nqubits_list = [circuit.num_qubits for circuit in circuits]
#     max_nqubits = max(nqubits_list)
#     circuit_depths = np.array([circutils.depth2qb(c) for c in circuits])
#     max_circuit_depth_index = np.argmax(circuit_depths)
#     cycles = get_cycles_containing_chain(backend, max_nqubits)
#     cycles_error_rates = np.array(error_rate_func(backend, circuits[max_circuit_depth_index], cycles))
#     cycles_error_rates, unique_indices = np.unique(cycles_error_rates, return_index=True)
#     layouts = [cycles[i] for i in unique_indices[::noise_rate_step][:nlayouts]]
#     return layouts

# def choose_layout_by_mindepth_errorrate(backend, circuits, nlayouts, noise_rate_step=1, error_rate_func=get_many_layouts_mean_error_rate):
#     nqubits_list = [circuit.num_qubits for circuit in circuits]
#     max_nqubits = max(nqubits_list)
#     circuit_depths = np.array([circutils.depth2qb(c) for c in circuits])
#     min_circuit_depth_index = np.argmin(circuit_depths)
#     cycles = get_cycles_containing_chain(backend, max_nqubits)
#     cycles_error_rates = np.array(error_rate_func(backend, circuits[min_circuit_depth_index], cycles))
#     cycles_error_rates, unique_indices = np.unique(cycles_error_rates, return_index=True)
#     layouts = [cycles[i] for i in unique_indices[::noise_rate_step][:nlayouts]]
#     return layouts

def choose_layout_by_mindepth_errorrate(backend, circuits, nlayouts, min_noise_factor=1, error_rate_func=get_many_layouts_mean_error_rate):
    if not isinstance(circuits, (list, tuple, np.ndarray)):
        circuits = [circuits]
    nqubits_list = [circuit.num_qubits for circuit in circuits]
    max_nqubits = max(nqubits_list)
    circuit_depths = np.array([circutils.depth2qb(c) for c in circuits])
    min_circuit_depth_index = np.argmin(circuit_depths)
    layouts = np.array(get_cycles_containing_chain(backend, max_nqubits))
    layouts_error_rates = np.array(error_rate_func(backend, circuits[min_circuit_depth_index], layouts))
    selected_layouts_indices = [np.argmin(layouts_error_rates)]
    for i in range(1, nlayouts):
        prev_index = selected_layouts_indices[-1]
        prev_error_rate = layouts_error_rates[prev_index]
        candidate_indices = np.where(layouts_error_rates/prev_error_rate >= min_noise_factor)[0]
        if len(candidate_indices) == 0:
            raise ValueError("Not enough layouts found within the noise rate step limit.")
        selected_layouts_indices.append(candidate_indices[0])
    selected_layouts = layouts[selected_layouts_indices]
    if len(circuits) > 1:
        return np.repeat(selected_layouts[:, np.newaxis, :], len(circuits), axis=1)
    else:
        return selected_layouts

def get_all_cmappings_from_layouts(layouts, nqubits):
    layouts = np.array(layouts)
    nlayouts = layouts.shape[0]
    cycles_per_layout = layouts.shape[-1]

    if nqubits > cycles_per_layout:
        raise ValueError("Number of qubits exceeds the length of the layout cycles.")

    ncycles = layouts.shape[0]*cycles_per_layout
    nmiddle_indices = int(np.prod(layouts.shape[1:-1]))
    if nmiddle_indices == 1:
        layouts = layouts[:, np.newaxis, :]
    cycles_inds = np.zeros((ncycles, *layouts.shape[1:]), dtype=int)
    middle_indices = np.unravel_index(np.arange(nmiddle_indices), layouts.shape[1:-1])
    for i in range(nmiddle_indices):
        midi = tuple(mid_ind[i] for mid_ind in middle_indices)
        this_cycle_inds = cycles_inds[:, midi]
        for j in range(nlayouts):
            this_layout_cycles = np.ones((cycles_per_layout, cycles_per_layout), dtype=int)*layouts[j, midi]
            this_layout_cycles = miscutils.roll_rows(this_layout_cycles, np.arange(cycles_per_layout))
            this_cycle_inds[j*cycles_per_layout:(j+1)*cycles_per_layout, :] = this_layout_cycles[:, np.newaxis, :]
        cycles_inds[:, midi] = this_cycle_inds
    cmappings = cycles_inds[..., :nqubits]
    if nmiddle_indices == 1:
        cmappings = cmappings[:, 0, :]
    return cmappings

def transpile_circuits(circuit_arr, backend, cmappings, optimization_level=3):
    circuit_arr = circutils.numpy_array_of_circuits(circuit_arr)
    cmappings = np.array(cmappings)
    nqubits = np.array([c.num_qubits for c in miscutils.iterate_nested_list(circuit_arr)])
    mnqubits = np.max(nqubits)
    if not np.less_equal(nqubits, cmappings.shape[1]).all():
        raise ValueError("One or more circuits have more qubits than the length of the layout cycles.")
    nmappings = cmappings.shape[0]
    result_shape = (nmappings, *circuit_arr.shape)
    transpiled_circuits = np.full(result_shape, None, dtype=object)
    for i in range(nmappings):
        transpiled_circuits[i] = circutils.transpile_circuits_with_consistent_layout(circuit_arr, backend, layout=cmappings[i][:mnqubits], optimization_level=optimization_level)
    return transpiled_circuits

def plot_mapping(backend, layout):
    if isinstance(layout, dict):
        layout = list(layout.values())
    couplings = backend.coupling_map.graph
    hardware_graph = couplings_to_undirected_graph(couplings)
    def node_color_func(n):
        if n in layout:
            return {"fillcolor": "blue", "style": "filled"}
        else:
            return {}
    im = graphviz_draw(hardware_graph, node_attr_fn=node_color_func, method="dot")
    display(im)

@dataclass
class ZNEResult:
    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float

    def __call__(self, x):
        return self.slope*x + self.intercept

def perform_linear_zne(noise_rates, measurements, sigma=None):
    noise_rates = np.array(noise_rates).flatten()
    measurements = np.array(measurements).flatten()
    if noise_rates.shape != measurements.shape:
        raise ValueError("Noise rates and measurements must have the same shape.")
    def lin_func(x, slope, intercept):
        return slope*x + intercept
    try:
        if sigma is not None:
            sigma = np.array(sigma).flatten()
            if sigma.shape != measurements.shape:
                raise ValueError("sigma must have the same shape as measurements.")
            popt, pcov = curve_fit(lin_func, noise_rates, measurements, sigma=sigma, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(lin_func, noise_rates, measurements)
        slope, intercept = float(popt[0]), float(popt[1])
        std_err = float(np.sqrt(max(pcov[1, 1], 0.0)))
        y_hat = lin_func(noise_rates, *popt)
        ss_res = float(np.sum((measurements - y_hat)**2))
        ss_tot = float(np.sum((measurements - measurements.mean())**2))
        r_value = float(np.sqrt(max(1.0 - ss_res/ss_tot, 0.0))) if ss_tot > 0 else None
        return ZNEResult(slope, intercept, r_value, None, std_err)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return ZNEResult(np.nan, np.nan, None, None, np.nan)

def perform_exponential_zne(noise_rates, measurements, p0=None, sigma=None):
    noise_rates = np.array(noise_rates).flatten()
    measurements = np.array(measurements).flatten()
    if noise_rates.shape != measurements.shape:
        raise ValueError("Noise rates and measurements must have the same shape.")
    def exp_func(x, a, b):
        return a*np.exp(-b*x)
    try:
        if sigma is not None:
            sigma = np.array(sigma).flatten()
            if sigma.shape != measurements.shape:
                raise ValueError("sigma must have the same shape as measurements.")
            popt, pcov = curve_fit(exp_func, noise_rates, measurements, p0=p0, sigma=sigma, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(exp_func, noise_rates, measurements, p0=p0)
        # intercept is a (value at x=0); std_err on intercept is sqrt(pcov[0, 0])
        return ZNEResult(float(popt[1]), float(popt[0]), None, None, float(np.sqrt(max(pcov[0, 0], 0.0))))
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return ZNEResult(np.nan, np.nan, None, None, np.nan)

def perform_polynomial_zne(noise_rates, measurements, degree=2, sigma=None):
    """Polynomial ZNE: fit y = sum_k c_k x^k of given `degree`, extrapolate to x=0.

    Returns a ZNEResult where:
      - intercept : extrapolated value at noise_rate = 0
      - std_err   : standard error on that intercept (0 if exactly determined)
      - slope     : leading-order coefficient (c_1) — the linear sensitivity at 0
      - r_value   : R of the fit (sqrt of R²)
      - p_value   : None
    """
    noise_rates = np.array(noise_rates).flatten()
    measurements = np.array(measurements).flatten()
    if noise_rates.shape != measurements.shape:
        raise ValueError("Noise rates and measurements must have the same shape.")
    n = len(noise_rates)
    if n < degree + 1:
        return ZNEResult(np.nan, np.nan, None, None, np.nan)
    # Coefficients ordered c_0, c_1, ..., c_degree (low to high in x)
    def poly_func(x, *c):
        return sum(ck * x**k for k, ck in enumerate(c))
    p0 = np.zeros(degree + 1)
    try:
        if sigma is not None:
            sigma = np.array(sigma).flatten()
            if sigma.shape != measurements.shape:
                raise ValueError("sigma must have the same shape as measurements.")
            popt, pcov = curve_fit(poly_func, noise_rates, measurements, p0=p0, sigma=sigma, absolute_sigma=True)
        elif n > degree + 1:
            popt, pcov = curve_fit(poly_func, noise_rates, measurements, p0=p0)
        else:
            popt, _ = curve_fit(poly_func, noise_rates, measurements, p0=p0)
            pcov = np.zeros((degree + 1, degree + 1))
        intercept = float(popt[0])
        std_err = float(np.sqrt(max(pcov[0, 0], 0.0)))
        slope = float(popt[1]) if degree >= 1 else 0.0
        y_hat = poly_func(noise_rates, *popt)
        ss_res = float(np.sum((measurements - y_hat)**2))
        ss_tot = float(np.sum((measurements - measurements.mean())**2))
        r_value = float(np.sqrt(max(1.0 - ss_res/ss_tot, 0.0))) if ss_tot > 0 else None
        return ZNEResult(slope, intercept, r_value, None, std_err)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        return ZNEResult(np.nan, np.nan, None, None, np.nan)

def perform_quadratic_zne(noise_rates, measurements, sigma=None):
    """Convenience wrapper: degree-2 polynomial ZNE."""
    return perform_polynomial_zne(noise_rates, measurements, degree=2, sigma=sigma)

def perform_cubic_zne(noise_rates, measurements, sigma=None):
    """Convenience wrapper: degree-3 polynomial ZNE.

    Note: with only 4 noise factors this is exactly determined (interpolating)
    and std_err will be 0. Use only when you have >= 5 noise factors.
    """
    return perform_polynomial_zne(noise_rates, measurements, degree=3, sigma=sigma)

def fit_cal_linear(noise_rates, cal_measurements):
    """Linear cal fit y = 1 + s*x with intercept pinned to 1. Returns dict with shape params."""
    x = np.array(noise_rates).flatten()
    y = np.array(cal_measurements).flatten()
    if x.shape != y.shape or len(x) < 1:
        return {"slope": np.nan}
    denom = float(np.sum(x*x))
    slope = float(np.sum(x*(y - 1.0)) / denom) if denom > 0 else np.nan
    return {"slope": slope}

def fit_cal_polynomial(noise_rates, cal_measurements, degree=2):
    """Polynomial cal fit y = 1 + sum_{k>=1} c_k x^k with intercept pinned to 1.
    Returns dict with the higher-order coeffs (constant term is fixed)."""
    x = np.array(noise_rates).flatten()
    y = np.array(cal_measurements).flatten()
    if x.shape != y.shape or len(x) < degree:
        return {"coeffs_high": np.full(degree, np.nan)}
    # Design matrix [x, x^2, ..., x^degree], regressing (y-1)
    X = np.vstack([x**k for k in range(1, degree+1)]).T
    coeffs, *_ = np.linalg.lstsq(X, y - 1.0, rcond=None)
    return {"coeffs_high": coeffs}  # order: c_1, c_2, ..., c_degree

def fit_cal_exponential(noise_rates, cal_measurements):
    """Cal fit y = exp(-b*x), intercept pinned to 1. Returns dict with shape param b."""
    x = np.array(noise_rates).flatten()
    y = np.array(cal_measurements).flatten()
    if x.shape != y.shape or len(x) < 1:
        return {"b": np.nan}
    def exp_func(xv, b):
        return np.exp(-b*xv)
    try:
        popt, _ = curve_fit(exp_func, x, y, p0=[0.3], maxfev=10000)
        return {"b": float(popt[0])}
    except (RuntimeError, ValueError):
        return {"b": np.nan}

def fit_cal_exponential_offset(noise_rates, cal_measurements):
    """Cal fit y = a*exp(-b*x) + c with constraint a + c = 1. Returns shape params (a, b)."""
    x = np.array(noise_rates).flatten()
    y = np.array(cal_measurements).flatten()
    if x.shape != y.shape or len(x) < 2:
        return {"a": np.nan, "b": np.nan}
    # With a + c = 1 ⇒ c = 1 - a; model: y = a*exp(-b*x) + (1 - a) = 1 + a*(exp(-b*x) - 1)
    def model(xv, a, b):
        return 1.0 + a*(np.exp(-b*xv) - 1.0)
    try:
        popt, _ = curve_fit(model, x, y, p0=[1.0, 0.3], maxfev=10000)
        return {"a": float(popt[0]), "b": float(popt[1])}
    except (RuntimeError, ValueError):
        return {"a": np.nan, "b": np.nan}


def apply_linear_with_slope(noise_rates, sim_measurements, cal_params, sigma=None):
    """Sim ZNE with the linear slope from the cal fit fixed.
    Refits y = a + slope*x for the intercept a only. Returns ZNEResult."""
    x = np.array(noise_rates).flatten()
    y = np.array(sim_measurements).flatten()
    s = cal_params.get("slope", np.nan)
    if not np.isfinite(s) or x.shape != y.shape or len(x) < 1:
        return ZNEResult(np.nan, np.nan, None, None, np.nan)
    residual = y - s*x
    n = len(x)
    if sigma is not None:
        sigma = np.array(sigma).flatten()
        if sigma.shape != y.shape:
            return ZNEResult(np.nan, np.nan, None, None, np.nan)
        w = 1.0 / (sigma**2)
        sw = float(np.sum(w))
        if sw <= 0:
            return ZNEResult(np.nan, np.nan, None, None, np.nan)
        intercept = float(np.sum(w*residual) / sw)
        std_err = float(np.sqrt(1.0 / sw))
    else:
        intercept = float(np.mean(residual))
        std_err = float(np.std(residual, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return ZNEResult(float(s), intercept, None, None, std_err)

def apply_polynomial_with_coeffs(noise_rates, sim_measurements, cal_params, sigma=None):
    """Sim ZNE with all higher-order polynomial coeffs fixed from cal.
    Refits only the constant term: y = a + sum_{k>=1} c_k x^k. Returns ZNEResult."""
    x = np.array(noise_rates).flatten()
    y = np.array(sim_measurements).flatten()
    c_high = np.asarray(cal_params.get("coeffs_high", []), dtype=float)
    if c_high.size == 0 or np.any(~np.isfinite(c_high)) or x.shape != y.shape:
        return ZNEResult(np.nan, np.nan, None, None, np.nan)
    poly_high = sum(c_high[k-1] * x**k for k in range(1, len(c_high)+1))
    residual = y - poly_high
    n = len(x)
    if sigma is not None:
        sigma = np.array(sigma).flatten()
        if sigma.shape != y.shape:
            return ZNEResult(np.nan, np.nan, None, None, np.nan)
        w = 1.0 / (sigma**2)
        sw = float(np.sum(w))
        if sw <= 0:
            return ZNEResult(np.nan, np.nan, None, None, np.nan)
        intercept = float(np.sum(w*residual) / sw)
        std_err = float(np.sqrt(1.0 / sw))
    else:
        intercept = float(np.mean(residual))
        std_err = float(np.std(residual, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    slope = float(c_high[0]) if len(c_high) >= 1 else 0.0
    return ZNEResult(slope, intercept, None, None, std_err)

def apply_exponential_with_rate(noise_rates, sim_measurements, cal_params, sigma=None):
    """Sim ZNE y = a*exp(-b*x) with b fixed from cal; fit a. Returns ZNEResult."""
    x = np.array(noise_rates).flatten()
    y = np.array(sim_measurements).flatten()
    b = cal_params.get("b", np.nan)
    if not np.isfinite(b) or x.shape != y.shape or len(x) < 1:
        return ZNEResult(np.nan, np.nan, None, None, np.nan)
    basis = np.exp(-b*x)
    n = len(x)
    if sigma is not None:
        sigma = np.array(sigma).flatten()
        if sigma.shape != y.shape:
            return ZNEResult(np.nan, np.nan, None, None, np.nan)
        w = 1.0 / (sigma**2)
        denom = float(np.sum(w*basis*basis))
        if denom <= 0:
            return ZNEResult(np.nan, np.nan, None, None, np.nan)
        a = float(np.sum(w*basis*y) / denom)
        std_err = float(np.sqrt(1.0 / denom))
    else:
        denom = float(np.sum(basis*basis))
        if denom <= 0:
            return ZNEResult(np.nan, np.nan, None, None, np.nan)
        a = float(np.sum(basis*y) / denom)
        residual = y - a*basis
        std_err = float(np.std(residual, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return ZNEResult(float(-a*b), a, None, None, std_err)

def apply_exponential_offset_with_params(noise_rates, sim_measurements, cal_params, sigma=None):
    """Sim ZNE y = a*exp(-b*x) + c with b fixed from cal; fit (a, c). intercept = a+c."""
    x = np.array(noise_rates).flatten()
    y = np.array(sim_measurements).flatten()
    b = cal_params.get("b", np.nan)
    if not np.isfinite(b) or x.shape != y.shape or len(x) < 2:
        return ZNEResult(np.nan, np.nan, None, None, np.nan)
    basis = np.exp(-b*x)
    X = np.vstack([basis, np.ones_like(x)]).T
    n = len(x)
    try:
        if sigma is not None:
            sigma = np.array(sigma).flatten()
            if sigma.shape != y.shape:
                return ZNEResult(np.nan, np.nan, None, None, np.nan)
            w = 1.0 / (sigma**2)
            W = np.diag(w)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            coef = np.linalg.solve(XtWX, XtWy)
            cov = np.linalg.inv(XtWX)
            a, c = float(coef[0]), float(coef[1])
            # Var(a + c) = Var(a) + Var(c) + 2*Cov(a, c)
            std_err = float(np.sqrt(max(cov[0, 0] + cov[1, 1] + 2*cov[0, 1], 0.0)))
        else:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, c = float(coef[0]), float(coef[1])
            residual = y - (a*basis + c)
            std_err = float(np.std(residual, ddof=1) / np.sqrt(n)) if n > 2 else 0.0
    except (np.linalg.LinAlgError, ValueError):
        return ZNEResult(np.nan, np.nan, None, None, np.nan)
    intercept = a + c
    return ZNEResult(float(-a*b), intercept, None, None, std_err)


# Registry mapping a `zne_func` to its (cal_fitter, sim_applier) pair for guess-mode.
# Used by postprocessing_joana when guess=True.
def _identify_zne_family(zne_func):
    """Return (cal_fitter, sim_applier) for the family of `zne_func`."""
    name = getattr(zne_func, "__name__", "")
    if name == "perform_linear_zne":
        return fit_cal_linear, apply_linear_with_slope
    if name == "perform_quadratic_zne":
        return (lambda nr, y: fit_cal_polynomial(nr, y, degree=2)), apply_polynomial_with_coeffs
    if name == "perform_cubic_zne":
        return (lambda nr, y: fit_cal_polynomial(nr, y, degree=3)), apply_polynomial_with_coeffs
    if name == "perform_polynomial_zne":
        return fit_cal_polynomial, apply_polynomial_with_coeffs
    if name == "perform_exponential_zne":
        return fit_cal_exponential, apply_exponential_with_rate
    if name == "perform_exponential_offset_zne":
        return fit_cal_exponential_offset, apply_exponential_offset_with_params
    raise ValueError(f"guess mode does not know the family of zne_func={name!r}; "
                     f"supported: perform_linear_zne, perform_quadratic_zne, "
                     f"perform_cubic_zne, perform_exponential_zne, perform_exponential_offset_zne")


def perform_exponential_offset_zne(noise_rates, measurements, p0=None, sigma=None):
    """Exponential ZNE with a free offset: y = a*exp(-b*x) + c, extrapolated to x=0.

    Same signature/return as perform_linear_zne / perform_exponential_zne so it's
    a drop-in `zne_func=` for the postprocessing pipeline.
      - intercept : a + c (value at x=0)
      - std_err   : propagated SE on (a+c) using the fit covariance
      - slope     : -a*b (derivative at x=0)
    """
    noise_rates = np.array(noise_rates).flatten()
    measurements = np.array(measurements).flatten()
    if noise_rates.shape != measurements.shape:
        raise ValueError("Noise rates and measurements must have the same shape.")
    if len(noise_rates) < 3:
        return ZNEResult(np.nan, np.nan, None, None, np.nan)

    def exp_func(x, a, b, c):
        return a*np.exp(-b*x) + c

    if p0 is None:
        c0 = float(np.min(measurements) if measurements[-1] < measurements[0] else np.max(measurements))
        a0 = float(measurements[0] - c0) if measurements[0] != c0 else 1.0
        p0 = [a0, 0.3, c0]

    try:
        if sigma is not None:
            sigma = np.array(sigma).flatten()
            if sigma.shape != measurements.shape:
                raise ValueError("sigma must have the same shape as measurements.")
            popt, pcov = curve_fit(exp_func, noise_rates, measurements, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000)
        else:
            popt, pcov = curve_fit(exp_func, noise_rates, measurements, p0=p0, maxfev=10000)
        a, b, c = popt
        intercept = float(a + c)
        # Var(a + c) = Var(a) + Var(c) + 2*Cov(a, c)
        var_intercept = float(pcov[0, 0] + pcov[2, 2] + 2*pcov[0, 2])
        std_err = float(np.sqrt(max(var_intercept, 0.0)))
        slope = float(-a*b)
        y_hat = exp_func(noise_rates, *popt)
        ss_res = float(np.sum((measurements - y_hat)**2))
        ss_tot = float(np.sum((measurements - measurements.mean())**2))
        r_value = float(np.sqrt(max(1.0 - ss_res/ss_tot, 0.0))) if ss_tot > 0 else None
        return ZNEResult(slope, intercept, r_value, None, std_err)
    except (RuntimeError, ValueError):
        return ZNEResult(np.nan, np.nan, None, None, np.nan)

def plot_layoutzne_extrapolation(all_cycles_observables, zne_layouts, noise_rates, color=None):
    zne_layouts = np.array(zne_layouts)
    noise_rates = np.array(noise_rates)
    if noise_rates.ndim > 1:
        raise ValueError("noise_rates must be a 1D array-like object")
    if noise_rates.shape[0] != zne_layouts.shape[0]:
        raise ValueError("Length of noise_rates must match number of ZNE layouts")
    if all_cycles_observables.shape[0] != zne_layouts.shape[0]*zne_layouts.shape[1]:
        raise ValueError("all_cycles_observables shape does not match zne_layouts dimensions")

    layout_mean_observables = np.zeros((zne_layouts.shape[0], *all_cycles_observables.shape[1:]))
    for i in range(noise_rates.shape[0]):
        this_layout_cycle_obs = all_cycles_observables[zne_layouts.shape[1]*i:zne_layouts.shape[1]*(i+1)]
        layout_mean_observables[i] = np.mean(this_layout_cycle_obs, axis=0)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=16)

    if color is None:
        color = plt.get_cmap("Set2")(0)

    # Perform linear ZNE extrapolation
    reg = perform_linear_zne(noise_rates, layout_mean_observables)
    plt.plot([0], [reg.intercept], "s", color=color, markersize=10, markeredgecolor="black")
    x = np.concatenate(([0], noise_rates))
    plt.plot(x, reg(x), "-", color=color, linewidth=1.6)
    plt.title(rf"$x(0) = {reg.intercept:.3f} \; / \; r = {reg.r_value:.3f}$", pad=10)

    # Plot all the points for each layout and noise rate
    all_cycle_noise_rates = np.repeat(noise_rates, zne_layouts.shape[1])
    plt.plot(all_cycle_noise_rates, all_cycles_observables, "o", color=color, markersize=6, alpha=0.5)

    # Plot mean observables for each layout
    plt.plot(noise_rates, layout_mean_observables, "s", color=color, markeredgecolor="black", markersize=8)

    plt.xlabel("Error rate")
    plt.ylabel(r"$\langle O \rangle$")
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    plt.show()
    plt.rcdefaults()