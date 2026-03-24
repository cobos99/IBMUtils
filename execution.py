from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import AddPostSelectionMeasures, AddSpectatorMeasures
from qiskit_ibm_runtime import Batch, Session, SamplerV2, Executor, NoiseLearnerV3, NoiseLearnerV3Options, QuantumProgram
from hardware.utils.circs import numpy_array_of_circuits, check_and_measure_active_qubits
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import hardware.utils.benchmarks as benchmarks
from qiskit.transpiler import PassManager
import hardware.utils.circs as circutils
import numpy as np

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

def execute_sampler_batch_choosing_layout(backend, sampler_opt_dict, job_log_circuits, cal_log_circuits=None, repeat_cal=False, db_options=None, job_db=None, share_total_shots=False, optimization_level=3, max_shots=75000):
    job_log_circuits = numpy_array_of_circuits(job_log_circuits)
    if cal_log_circuits is not None: 
        cal_log_circuits = numpy_array_of_circuits(cal_log_circuits)

    if len(job_log_circuits.shape) < 2:
        job_log_circuits = job_log_circuits[None]
    if cal_log_circuits is not None:
        if len(cal_log_circuits.shape) < 2:
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

    batch_options = {"job_circuits_shape": job_log_circuits.shape, "cal_circuits_shape": cal_log_circuits.shape if cal_log_circuits is not None else None, "nrepetitions": nrepetitions, "repeat_cal": repeat_cal}
    jobs = []

    with Session(backend=backend) as session:
        batch_options["batch_id"] = session.session_id
        target = benchmarks.update_target(backend, session)
        # Transpile circuits choosing best layout
        pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
        flatten_job_log_circuits = numpy_array_of_circuits(job_log_circuits).flatten()
        depths = np.array([circutils.depth2qb(c) for c in flatten_job_log_circuits])
        longest_circuit_ind = np.argmax(depths)
        transpiled_longest_circuit = pm.run(flatten_job_log_circuits[longest_circuit_ind])
        chosen_layout = transpiled_longest_circuit.layout.final_index_layout()
        job_circuits = circutils.transpile_circuits_with_consistent_layout(job_log_circuits, backend, chosen_layout)
        if cal_log_circuits is not None:
            cal_circuits = circutils.transpile_circuits_with_consistent_layout(cal_log_circuits, backend, chosen_layout)
        if cal_log_circuits is None:
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

def execute_sampler_batch_choosing_layout_by_batches(backend, sampler_opt_dict, job_log_circuits, batches_first_index, cal_log_circuits=None, jobcal_together=False, db_options=None, job_db=None, optimization_level=0):
    job_log_circuits = numpy_array_of_circuits(job_log_circuits)
    if cal_log_circuits is not None:
        cal_log_circuits = numpy_array_of_circuits(cal_log_circuits)
        # if len(batches_first_index) != cal_log_circuits.shape[0]:
        #     raise ValueError("batches_first_index length must match cal_log_circuits first dimension")
        if cal_log_circuits.shape[-1] != job_log_circuits.shape[-1]:
            raise ValueError("cal_log_circuits must have the same of number of circuits per layout as job_log_circuits")

    batch_options = {"job_circuits_shape": job_log_circuits.shape, "cal_circuits_shape": cal_log_circuits.shape if cal_log_circuits is not None else None, "nrepetitions": 1, "repeat_cal": False}
    jobs = []
    if cal_log_circuits is not None: cal_jobs = []
    circuits_to_run = []
    if cal_log_circuits is not None:
        cal_circuits_to_run = []
    with Session(backend=backend) as session:
        batch_options["batch_id"] = session.session_id
        sampler = SamplerV2(mode=session, options=sampler_opt_dict)
        for i in range(len(batches_first_index)):
            print("Running batch ", i+1, " out of ", len(batches_first_index))
            target = benchmarks.update_target(backend, session)
            this_job_log_circuits = job_log_circuits[batches_first_index[i]:batches_first_index[i+1] if i < len(batches_first_index)-1 else None]
            if cal_log_circuits is not None:
                if cal_log_circuits.shape[0] == job_log_circuits.shape[0]:
                    this_cal_log_circuits = cal_log_circuits[batches_first_index[i]:batches_first_index[i+1] if i < len(batches_first_index)-1 else None]
                else:
                    this_cal_log_circuits = cal_log_circuits[i:i+1]
            pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
            # pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
            flatten_job_log_circuits = numpy_array_of_circuits(this_job_log_circuits).flatten()
            depths = np.array([circutils.depth2qb(c) for c in flatten_job_log_circuits])
            longest_circuit_ind = np.argmax(depths)
            transpiled_longest_circuit = pm.run(flatten_job_log_circuits[longest_circuit_ind])
            chosen_layout = transpiled_longest_circuit.layout.final_index_layout()
            job_circuits = circutils.transpile_circuits_with_consistent_layout(this_job_log_circuits, backend, chosen_layout)
            job_circuits = [check_and_measure_active_qubits(c) for c in job_circuits.flatten()]
            if cal_log_circuits is not None:
                cal_circuits = circutils.transpile_circuits_with_consistent_layout(this_cal_log_circuits, backend, chosen_layout)
                cal_circuits = [check_and_measure_active_qubits(c) for c in cal_circuits.flatten()]
                if cal_log_circuits.shape == job_log_circuits.shape:
                    if not jobcal_together:
                        this_circuits_to_run = multirep_structure(job_circuits, nrepetitions=1, cal_elements=cal_circuits, repeat_cal=False).flatten().tolist()
                        circuits_to_run += this_circuits_to_run
                        for c in this_circuits_to_run:
                            jobs.append(sampler.run([c]))
                    else:
                        circuits_to_run += job_circuits
                        for jc, cc in zip(job_circuits, cal_circuits):
                            jobs.append(sampler.run([cc, jc]))
                else:
                    circuits_to_run += job_circuits
                    cal_circuits_to_run += cal_circuits
                    for c in cal_circuits:
                        cal_jobs.append(sampler.run([c]))
                    for c in job_circuits:
                        jobs.append(sampler.run([c]))
                    # jobs[-1].result()
            else:
                circuits_to_run += job_circuits
                for c in job_circuits:
                    jobs.append(sampler.run([c]))
    
    if (cal_log_circuits is not None) and not (cal_log_circuits.shape == job_log_circuits.shape):
        circuits_to_run = cal_circuits_to_run + circuits_to_run
        jobs = cal_jobs + jobs

    if job_db is not None:
        job_ids = [job.job_id() for job in jobs]
        all_options = (db_options if db_options is not None else {}) | batch_options | sampler_opt_dict
        job_db.add(all_options, circuits_to_run, "Sampler", job_ids)

    return jobs

def znepea_options_check(options_dict):
    required_keys = [
        "learner_num_randomizations",
        "learner_shots_per_randomization",
        "learner_layer_pair_depth",
        "learner_post_selection_enabled",
        "noise_factors",
        "pt_num_randomizations",
        "pt_shots_per_randomization",
        "noise_factors"
    ]
    for key in required_keys:
        if key not in options_dict:
            raise ValueError(f"Missing required ZNE-PEA option: {key}")

def diagobs_noise_amplification_argument_generator(samplex, parameter_values, noise_maps, noise_rates):
    samplex_accepted_arguments = [spec for spec in samplex.inputs().specs]
    samplex_arguments = {"pauli_lindblad_maps": noise_maps}
    for spec in samplex_accepted_arguments:
        if "basis_changes" in spec.name:
            samplex_arguments[spec.name] = np.full(spec.shape, 0).tolist()
        if "noise_scales" in spec.name:
            samplex_arguments[spec.name] = noise_rates
        if "parameter_values" in spec.name:
            samplex_arguments[spec.name] = parameter_values
    return samplex_arguments

def execute_znepea_choosing_layout(backend, job_log_barrier_circuits, parameters_func, db_options, uinst_filter_func=None, job_db=None, optimization_level=3):
    znepea_options_check(db_options)
    job_log_circuits = numpy_array_of_circuits(job_log_barrier_circuits)
    job_log_circuits_shape = job_log_circuits.shape
    job_log_circuits = [circutils.check_and_measure_active_qubits(c) for c in job_log_circuits.flatten()]
    batch_options = {"job_circuits_shape": job_log_circuits_shape, "cal_circuits_shape": None, "nrepetitions": 1, "repeat_cal": False}
    jobs = []
    with Session(backend=backend) as session:
        batch_options["batch_id"] = session.session_id
        
        # Run the layout benchmarks
        print("Running layout benchmarks...")
        target = benchmarks.update_target(backend, session)
        pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
        depths = np.array([circutils.depth2qb(c) for c in job_log_circuits])
        longest_circuit_ind = np.argmax(depths)
        transpiled_longest_circuit = pm.run(job_log_circuits[longest_circuit_ind])
        chosen_layout = transpiled_longest_circuit.layout.final_index_layout()
        job_circuits = circutils.transpile_circuits_with_consistent_layout(job_log_circuits, backend, chosen_layout)

        # Prepare templates, samplexes and noise injected circuits
        print("Preparing circuits for noise learning...")
        box_job_circuits = circutils.box_circuits(job_circuits,
                                                  enable_gates=True,
                                                  enable_measures=True,
                                                  twirling_strategy="active_circuit")
        box_job_circuits_unist = circutils.get_unique_boxes(box_job_circuits.tolist(), undress_boxes=True, normalize_annotations=None)
        if uinst_filter_func is not None:
            box_job_circuits_unist = uinst_filter_func(box_job_circuits_unist)
        
        ninject_job_circuits = circutils.box_circuits(job_circuits,
                                                      twirling_strategy="active_circuit",
                                                      inject_noise_strategy="individual_modification",
                                                      inject_noise_targets="gates",
                                                      measure_annotations="all")

        templates, samplexes = circutils.samplomatic_build(ninject_job_circuits)
        if db_options["learner_post_selection_enabled"]:
            # Set up post selection PM (to add PS instructions)
            post_selection_pm = PassManager(
                [
                    AddSpectatorMeasures(backend.coupling_map),
                    AddPostSelectionMeasures(x_pulse_type="rx"), #TODO: Change pulse type
                ]
            )
            templates = post_selection_pm.run(templates)

        # Run the noise learner
        print("Running noise learner...")
        noise_learner_options = NoiseLearnerV3Options(
            num_randomizations=db_options["learner_num_randomizations"],
            shots_per_randomization=db_options["learner_shots_per_randomization"],
            layer_pair_depths=db_options["learner_layer_pair_depth"],
            post_selection={
                "enable": db_options["learner_post_selection_enabled"],
                "strategy": "edge",
                "x_pulse_type": "rx"
            })
        noise_learner = NoiseLearnerV3(mode=session, options=noise_learner_options)
        noise_learner_job = noise_learner.run(instructions=box_job_circuits_unist)
        noise_maps = noise_learner_job.result().to_dict(instructions=box_job_circuits_unist, require_refs=False)
        batch_options["noise_learner_job_id"] = noise_learner_job.job_id()

        # Submit jobs
        print("Submitting jobs...")
        executor = Executor(mode=session)
        for i, (template, samplex) in enumerate(zip(templates, samplexes)):
            samplex_arguments = diagobs_noise_amplification_argument_generator(samplex, parameter_values=parameters_func(i), noise_maps=noise_maps, noise_rates=db_options["noise_rates"])
            program = QuantumProgram(shots=db_options["pt_shots_per_randomization"])
            program.append_samplex_item(
                circuit=template,
                samplex=samplex,
                samplex_arguments=samplex_arguments,
                shape=(db_options["pt_num_randomizations"],)
            )
            jobs.append(executor.run(program))
        
    if job_db is not None:
        job_ids = [job.job_id() for job in jobs]
        all_options = (db_options if db_options is not None else {}) | batch_options
        job_db.add(all_options, job_circuits, "ZNE-PEA", job_ids)

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