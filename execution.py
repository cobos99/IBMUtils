from hardware.utils.circs import numpy_array_of_circuits, check_and_measure_active_qubits
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_ibm_runtime import Batch, Session, SamplerV2
from qiskit.transpiler import InstructionProperties
import hardware.utils.benchmarks as benchmarks
import numpy as np
import copy

# def multirep_structure(job_elements, nrepetitions, cal_elements=None, repeat_cal=False):
#     if not nrepetitions.is_integer() or nrepetitions < 1:
#         raise ValueError("nrepetitions must be a positive integer")
#     nrepetitions = int(nrepetitions)
    
#     if not isinstance(job_elements, np.ndarray):
#         raise ValueError("job_elements must be a numpy array")

#     if (cal_elements is not None) and (not isinstance(cal_elements, np.ndarray)):
#         raise ValueError("If provided, cal_elements must be a numpy array")

#     if cal_elements is not None:
#         if job_elements.shape != cal_elements.shape:
#             raise ValueError("If provided, cal_elements must have the same shape as job_elements")

#     circ_shape = job_elements.shape

#     if len(job_elements.shape) < 2:
#         job_elements = job_elements[None]
#         if cal_elements is not None:
#             cal_elements = np.array(cal_elements)[None]

#     if cal_elements is not None:
#         circuits_to_run = []
#         for ilayout in range(job_elements.shape[0]):
#             if repeat_cal:
#                 this_layout_circuits = np.full(2*nrepetitions*circ_shape[1], {}, dtype=object)
#                 repeated_job_circuits = np.repeat(job_elements[ilayout], nrepetitions)
#                 repeated_cal_circuits = np.repeat(cal_elements[ilayout], nrepetitions)
#                 this_layout_circuits[0::2] = repeated_cal_circuits
#                 this_layout_circuits[1::2] = repeated_job_circuits
#                 circuits_to_run.append(this_layout_circuits)
#             else:
#                 this_layout_circuits = np.full((nrepetitions+1)*circ_shape[1], {}, dtype=object)
#                 for i in range(circ_shape[1]):
#                     this_layout_circuits[i*(nrepetitions+1)] = cal_elements[ilayout, i]
#                     this_layout_circuits[i*(nrepetitions+1)+1:(i+1)*(nrepetitions+1)] = [job_elements[ilayout, i]]*nrepetitions
#                 circuits_to_run.append(this_layout_circuits)
#         circuits_to_run = np.array(circuits_to_run)
#     else:
#         circuits_to_run = np.repeat(job_elements, nrepetitions, axis=1)

#     return np.array(circuits_to_run)

def multirep_structure(job_elements, nrepetitions, cal_elements=None, repeat_cal=False):
    if not nrepetitions.is_integer() or nrepetitions < 1:
        raise ValueError("nrepetitions must be a positive integer")
    nrepetitions = int(nrepetitions)
    
    job_elements = numpy_array_of_circuits(job_elements)
    if cal_elements is not None:
        cal_elements = numpy_array_of_circuits(cal_elements)
        if job_elements.shape != cal_elements.shape:
            raise ValueError("If provided, cal_elements must have the same shape as job_elements")
        
    circ_shape = job_elements.shape

    if len(job_elements.shape) < 2:
        job_elements = job_elements[None]
        if cal_elements is not None:
            cal_elements = cal_elements[None]

    if cal_elements is not None:
        circuits_to_run = []
        for ilayout in range(job_elements.shape[0]):
            if repeat_cal:
                this_layout_circuits = np.full((*circ_shape[1:-1], 2*nrepetitions*circ_shape[-1]), None, dtype=object)
                repeated_job_circuits = np.repeat(job_elements[ilayout], nrepetitions, axis=-1)
                repeated_cal_circuits = np.repeat(cal_elements[ilayout], nrepetitions, axis=-1)
                this_layout_circuits[..., 0::2] = repeated_cal_circuits
                this_layout_circuits[..., 1::2] = repeated_job_circuits
                circuits_to_run.append(this_layout_circuits)
            else:
                this_layout_circuits = np.full((*circ_shape[1:-1], (nrepetitions+1)*circ_shape[-1]), None, dtype=object)
                for i in range(circ_shape[-1]):
                    this_layout_circuits[..., i*(nrepetitions+1)] = cal_elements[ilayout, ..., i]
                    this_jobs = job_elements[ilayout, ..., i]
                    this_layout_circuits[..., i*(nrepetitions+1)+1:(i+1)*(nrepetitions+1)] = np.repeat(this_jobs.reshape(*this_jobs.shape, 1), nrepetitions, axis=-1)
                circuits_to_run.append(this_layout_circuits)
        circuits_to_run = np.array(circuits_to_run)
    else:
        circuits_to_run = np.repeat(job_elements, nrepetitions, axis=-1)
    return np.array(circuits_to_run)

# def execute_sampler_batch(backend, sampler_opt_dict, job_circuits, cal_circuits=None, repeat_cal=False, db_options=None, job_db=None, share_total_shots=False, max_shots=75000):
#     job_circuits = numpy_array_of_circuits(job_circuits)
#     if cal_circuits is not None: 
#         cal_circuits = numpy_array_of_circuits(cal_circuits)
#         if job_circuits.shape != cal_circuits.shape:
#             raise ValueError("If provided, cal_circuits must have the same shape as job_circuits")

#     if len(job_circuits.shape) < 2:
#         job_circuits = job_circuits[None]
#         if cal_circuits is not None:
#             cal_circuits = cal_circuits[None]

#     if share_total_shots and len(job_circuits.shape) > 1:
#         shots = sampler_opt_dict.get("default_shots", 1)
#         shots_per_layout = shots // job_circuits.shape[0]
#     else:
#         shots = sampler_opt_dict.get("default_shots", 1)
#         shots_per_layout = shots

#     if cal_circuits is None:
#         repeat_cal = False

#     if shots_per_layout > max_shots:
#         nrepetitions = shots_per_layout // max_shots
#         if nrepetitions == 0:
#             nrepetitions = 1
#         last_rep_shots = shots % max_shots
#         limited_options = sampler_opt_dict.copy()
#         limited_options["default_shots"] = max_shots
#         last_options = sampler_opt_dict.copy()
#         last_options["default_shots"] = last_rep_shots
#         sampler_options_per_sim = [limited_options]*(nrepetitions * (2 if repeat_cal else 1))
#         if last_rep_shots != 0:
#             sampler_options_per_sim += [last_options]*(2 if repeat_cal else 1)
#             nrepetitions += 1
#         if cal_circuits is not None and not repeat_cal:
#             sampler_options_per_sim = [sampler_options_per_sim[0]] + sampler_options_per_sim
#     else:
#         nrepetitions = 1
#         edited_sampler_options = sampler_opt_dict.copy()
#         edited_sampler_options["default_shots"] = shots_per_layout
#         sampler_options_per_sim = [edited_sampler_options]*(2 if cal_circuits is not None else 1)

#     circuits_to_run = multirep_structure(job_circuits, nrepetitions, cal_circuits, repeat_cal).flatten()
#     circuits_to_run = [check_and_measure_active_qubits(c) for c in circuits_to_run]
#     batch_options = {"jobs_shape": job_circuits.shape, "nrepetitions": nrepetitions, "cal": cal_circuits is not None, "repeat_cal": repeat_cal}
#     jobs = []
#     with Batch(backend=backend) as batch:
#         batch_options["batch_id"] = batch.session_id
#         for i, circuit in enumerate(circuits_to_run):
#             this_options = sampler_options_per_sim[i % len(sampler_options_per_sim)]
#             sampler = SamplerV2(mode=batch, options=this_options)
#             jobs.append(sampler.run([circuit]))

#     if job_db is not None:
#         job_ids = [job.job_id() for job in jobs]
#         all_options = (db_options if db_options is not None else {}) | batch_options | sampler_opt_dict
#         job_db.add(all_options, circuits_to_run, "Sampler", job_ids)

#     return jobs

# def execute_sampler_batch(backend, sampler_opt_dict, job_circuits, cal_circuits=None, repeat_cal=False, db_options=None, job_db=None, share_total_shots=False, max_shots=75000):
#     job_circuits = numpy_array_of_circuits(job_circuits)
#     if cal_circuits is not None: 
#         cal_circuits = numpy_array_of_circuits(cal_circuits)
    
#     if len(job_circuits.shape) < 2:
#         job_circuits = job_circuits[None]
#     if cal_circuits is not None:
#         if len(cal_circuits.shape) < 2:
#             cal_circuits = cal_circuits[None]

#     if share_total_shots and len(job_circuits.shape) > 1:
#         shots = sampler_opt_dict.get("default_shots", 1)
#         shots_per_layout = shots // job_circuits.shape[0]
#     else:
#         shots = sampler_opt_dict.get("default_shots", 1)
#         shots_per_layout = shots

#     if cal_circuits is None:
#         repeat_cal = False

#     if shots_per_layout > max_shots:
#         nrepetitions = shots_per_layout // max_shots
#         if nrepetitions == 0:
#             nrepetitions = 1
#         last_rep_shots = shots % max_shots
#         limited_options = sampler_opt_dict.copy()
#         limited_options["default_shots"] = max_shots
#         last_options = sampler_opt_dict.copy()
#         last_options["default_shots"] = last_rep_shots
#         sampler_options_per_job_circ = [limited_options]*nrepetitions
#         sampler_options_per_cal_circ = [limited_options]*(nrepetitions if repeat_cal else 1)
#         if last_rep_shots != 0:
#             sampler_options_per_job_circ += [last_options]
#             nrepetitions += 1
#             if repeat_cal:
#                 sampler_options_per_cal_circ += [last_options]
#     else:
#         nrepetitions = 1
#         edited_sampler_options = sampler_opt_dict.copy()
#         edited_sampler_options["default_shots"] = shots_per_layout
#         sampler_options_per_job_circ = [edited_sampler_options]
#         sampler_options_per_cal_circ = [edited_sampler_options]
    
#     if cal_circuits is None:
#         circuits_to_run = multirep_structure(job_circuits, nrepetitions).flatten()
#     else:
#         if job_circuits.shape == cal_circuits.shape:
#             circuits_to_run = multirep_structure(job_circuits, nrepetitions, cal_circuits, repeat_cal).flatten()
#         else:
#             circuits_to_run = multirep_structure(job_circuits, nrepetitions).flatten()
#             repeated_cal_circuits = multirep_structure(cal_circuits, nrepetitions).flatten()
#             circuits_to_run = np.concatenate([repeated_cal_circuits, circuits_to_run])

#     circuits_to_run = [check_and_measure_active_qubits(c) for c in circuits_to_run]
#     batch_options = {"job_circuits_shape": job_circuits.shape, "cal_circuits_shape": cal_circuits.shape if cal_circuits is not None else None, "nrepetitions": nrepetitions, "repeat_cal": repeat_cal}
#     jobs = []
#     with Batch(backend=backend) as batch:
#         batch_options["batch_id"] = batch.session_id
#         for i, circuit in enumerate(circuits_to_run):
#             if cal_circuits is None:
#                 this_options = [i % len(sampler_options_per_job_circ)]
#             else:
#                 if job_circuits.shape == cal_circuits.shape:
#                     if (i % 2) == 0:
#                         this_options = sampler_options_per_job_circ[(i // 2) % nrepetitions]
#                     else:
#                         this_options = sampler_options_per_cal_circ[(i // 2) % nrepetitions if repeat_cal else 0]
#                 else:
#                     if i < (len(circuits_to_run) - len(repeated_cal_circuits)):
#                         this_options = sampler_options_per_job_circ[i % nrepetitions]
#                     else:
#                         this_options = sampler_options_per_cal_circ[(i - (len(circuits_to_run) - len(repeated_cal_circuits))) % (nrepetitions if repeat_cal else 1)]
#             sampler = SamplerV2(mode=batch, options=this_options)
#             jobs.append(sampler.run([circuit]))
    
#     if job_db is not None:
#         job_ids = [job.job_id() for job in jobs]
#         all_options = (db_options if db_options is not None else {}) | batch_options | sampler_opt_dict
#         job_db.add(all_options, circuits_to_run, "Sampler", job_ids)
#     return jobs

def execute_sampler_batch(backend, sampler_opt_dict, job_circuits, cal_circuits=None, repeat_cal=False, db_options=None, job_db=None, share_total_shots=False, max_shots=75000):
    job_circuits = numpy_array_of_circuits(job_circuits)
    if cal_circuits is not None: 
        cal_circuits = numpy_array_of_circuits(cal_circuits)
    
    if len(job_circuits.shape) < 2:
        job_circuits = job_circuits[None]
    if cal_circuits is not None:
        if len(cal_circuits.shape) < 2:
            cal_circuits = cal_circuits[None]
    else:
        repeat_cal = False

    shots = sampler_opt_dict.get("default_shots", 1)
    if share_total_shots:
        shots_per_layout = shots // job_circuits.shape[0]
    else:
        shots_per_layout = shots

    if shots_per_layout > max_shots:
        nrepetitions = shots_per_layout // max_shots
        if nrepetitions == 0:
            nrepetitions = 1
        last_rep_shots = shots % max_shots
        limited_options = sampler_opt_dict.copy()
        limited_options["default_shots"] = max_shots
        if sampler_opt_dict.get("twirling", False):
            limited_options["twirling"]["shots_per_randomization"] = int(max_shots / sampler_opt_dict["twirling"]["num_randomizations"])
        last_options = sampler_opt_dict.copy()
        last_options["default_shots"] = last_rep_shots
        if sampler_opt_dict.get("twirling", False):
            last_options["twirling"]["shots_per_randomization"] = int(last_rep_shots / sampler_opt_dict["twirling"]["num_randomizations"])
        sampler_options_per_job_circ = [limited_options]*nrepetitions
        sampler_options_per_cal_circ = [limited_options]*(nrepetitions if repeat_cal else 1)
        if last_rep_shots != 0:
            sampler_options_per_job_circ += [last_options]
            nrepetitions += 1
            if repeat_cal:
                sampler_options_per_cal_circ += [last_options]
    else:
        nrepetitions = 1
        edited_sampler_options = sampler_opt_dict.copy()
        edited_sampler_options["default_shots"] = shots_per_layout
        if sampler_opt_dict.get("twirling", False):
            edited_sampler_options["twirling"]["shots_per_randomization"] = int(shots_per_layout / sampler_opt_dict["twirling"]["num_randomizations"])
        sampler_options_per_job_circ = [edited_sampler_options]
        sampler_options_per_cal_circ = [edited_sampler_options]

    if cal_circuits is None:
        circuits_to_run = multirep_structure(job_circuits, nrepetitions).flatten()
        this_options_job_func = lambda i: sampler_options_per_job_circ[i % nrepetitions]
    else:
        if job_circuits.shape == cal_circuits.shape:
            circuits_to_run = multirep_structure(job_circuits, nrepetitions, cal_circuits, repeat_cal).flatten()
            def this_options_job_func(i):
                if (i % 2) == 0:
                    return sampler_options_per_cal_circ[(i // 2) % (nrepetitions if repeat_cal else 1)]
                else:
                    return sampler_options_per_job_circ[(i // 2) % nrepetitions]
        else:
            circuits_to_run = multirep_structure(job_circuits, nrepetitions).flatten()
            repeated_cal_circuits = multirep_structure(cal_circuits, nrepetitions).flatten()
            circuits_to_run = np.concatenate([repeated_cal_circuits, circuits_to_run])
            def this_options_job_func(i):
                if i < (len(circuits_to_run) - len(repeated_cal_circuits)):
                    return sampler_options_per_job_circ[i % nrepetitions]
                else:
                    return sampler_options_per_cal_circ[(i - (len(circuits_to_run) - len(repeated_cal_circuits))) % (nrepetitions if repeat_cal else 1)]
    
    circuits_to_run = [check_and_measure_active_qubits(c) for c in circuits_to_run]
    batch_options = {"job_circuits_shape": job_circuits.shape, "cal_circuits_shape": cal_circuits.shape if cal_circuits is not None else None, "nrepetitions": nrepetitions, "repeat_cal": repeat_cal}
    jobs = []
    with Batch(backend=backend) as batch:
        batch_options["batch_id"] = batch.session_id
        for i, circuit in enumerate(circuits_to_run):
            sampler = SamplerV2(mode=batch, options=this_options_job_func(i))
            jobs.append(sampler.run([circuit]))

    if job_db is not None:
        job_ids = [job.job_id() for job in jobs]
        all_options = (db_options if db_options is not None else {}) | batch_options | sampler_opt_dict
        job_db.add(all_options, circuits_to_run, "Sampler", job_ids)
    
    return jobs

def execute_sampler_batch_choosing_layout(backend, sampler_opt_dict, job_log_circuits, cal_log_circuits=None, repeat_cal=False, db_options=None, job_db=None, share_total_shots=False, max_shots=75000):
    job_log_circuits = numpy_array_of_circuits(job_log_circuits)
    if cal_log_circuits is not None: 
        cal_circuits = numpy_array_of_circuits(cal_log_circuits)

    if len(job_log_circuits.shape) < 2:
        job_log_circuits = job_log_circuits[None]
    if cal_log_circuits is not None:
        if len(cal_circuits.shape) < 2:
            cal_log_circuits = cal_log_circuits[None]
    else:
        repeat_cal = False

    shots = sampler_opt_dict.get("default_shots", 1)
    if share_total_shots:
        shots_per_layout = shots // job_log_circuits.shape[0]
    else:
        shots_per_layout = shots

    if shots_per_layout > max_shots:
        nrepetitions = shots_per_layout // max_shots
        if nrepetitions == 0:
            nrepetitions = 1
        last_rep_shots = shots % max_shots
        limited_options = sampler_opt_dict.copy()
        limited_options["default_shots"] = max_shots
        if sampler_opt_dict.get("twirling", False):
            limited_options["twirling"]["shots_per_randomization"] = int(max_shots / sampler_opt_dict["twirling"]["num_randomizations"])
        last_options = sampler_opt_dict.copy()
        last_options["default_shots"] = last_rep_shots
        if sampler_opt_dict.get("twirling", False):
            last_options["twirling"]["shots_per_randomization"] = int(last_rep_shots / sampler_opt_dict["twirling"]["num_randomizations"])
        sampler_options_per_job_circ = [limited_options]*nrepetitions
        sampler_options_per_cal_circ = [limited_options]*(nrepetitions if repeat_cal else 1)
        if last_rep_shots != 0:
            sampler_options_per_job_circ += [last_options]
            nrepetitions += 1
            if repeat_cal:
                sampler_options_per_cal_circ += [last_options]
    else:
        nrepetitions = 1
        edited_sampler_options = sampler_opt_dict.copy()
        edited_sampler_options["default_shots"] = shots_per_layout
        if sampler_opt_dict.get("twirling", False):
            edited_sampler_options["twirling"]["shots_per_randomization"] = int(shots_per_layout / sampler_opt_dict["twirling"]["num_randomizations"])
        sampler_options_per_job_circ = [edited_sampler_options]
        sampler_options_per_cal_circ = [edited_sampler_options]

    batch_options = {"job_circuits_shape": job_circuits.shape, "cal_circuits_shape": cal_circuits.shape if cal_circuits is not None else None, "nrepetitions": nrepetitions, "repeat_cal": repeat_cal}
    jobs = []

    benchmark_batches = benchmarks.ibm_metrics_experiments(backend)
    benchmark_batches_exp = BatchExperiment(benchmark_batches, backend)
    coupling_graph = backend.coupling_map.graph.to_undirected(multigraph=False)
    one_dir_coupling_map = coupling_graph.edge_list()
    batch_options["batch_id"] = session.session_id
    instruction_2q_name = "cz"
    with Session(backend=backend) as session:
        # Run benchmarks for circuits
        layout_sampler_dict = sampler_options_per_job_circ[0]

        # Run characterization experiments
        batches_exp_data = benchmark_batches_exp.run(
            sampler=layout_sampler_dict, **{"shots": 1e3, "dynamic": False}
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
    
    # Transpile circuits choosing best layout
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    job_circuits = pm.run(job_log_circuits)
    if cal_log_circuits is not None:
        cal_circuits = pm.run(cal_log_circuits)

    if cal_circuits is None:
        circuits_to_run = multirep_structure(job_circuits, nrepetitions).flatten()
        this_options_job_func = lambda i: sampler_options_per_job_circ[i % nrepetitions]
    else:
        if job_circuits.shape == cal_circuits.shape:
            circuits_to_run = multirep_structure(job_circuits, nrepetitions, cal_circuits, repeat_cal).flatten()
            def this_options_job_func(i):
                if (i % 2) == 0:
                    return sampler_options_per_cal_circ[(i // 2) % (nrepetitions if repeat_cal else 1)]
                else:
                    return sampler_options_per_job_circ[(i // 2) % nrepetitions]
        else:
            circuits_to_run = multirep_structure(job_circuits, nrepetitions).flatten()
            repeated_cal_circuits = multirep_structure(cal_circuits, nrepetitions).flatten()
            circuits_to_run = np.concatenate([repeated_cal_circuits, circuits_to_run])
            def this_options_job_func(i):
                if i < (len(circuits_to_run) - len(repeated_cal_circuits)):
                    return sampler_options_per_job_circ[i % nrepetitions]
                else:
                    return sampler_options_per_cal_circ[(i - (len(circuits_to_run) - len(repeated_cal_circuits))) % (nrepetitions if repeat_cal else 1)]
    
    circuits_to_run = [check_and_measure_active_qubits(c) for c in circuits_to_run]
    for i, circuit in enumerate(circuits_to_run):
        sampler = SamplerV2(mode=session, options=this_options_job_func(i))
        jobs.append(sampler.run([circuit]))
    
    if job_db is not None:
        job_ids = [job.job_id() for job in jobs]
        all_options = (db_options if db_options is not None else {}) | batch_options | sampler_opt_dict
        job_db.add(all_options, circuits_to_run, "Sampler", job_ids)

    return jobs

def get_backend_best_qubit_chain(backend, nqubits):
    qlists = backend.properties().general_qlists
    for qlist in qlists:
        if qlist["name"] == f"lf_{nqubits:d}":
            return qlist["qubits"]
        
def backends_objs_to_names(backends_arr):
        if type(backends_arr) == str:
            backends_arr = [backends_arr]
        else:
            try:
                backends_arr = list(backends_arr)
            except TypeError:
                backends_arr = [backends_arr]
        backends_name_arr = []
        for backend in backends_arr:
            if type(backend) != str:
                backends_name_arr.append(backend.name)
            else:
                backends_name_arr.append(backend)
        return backends_name_arr if len(backends_name_arr) > 1 else backends_name_arr[0]