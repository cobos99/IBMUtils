from qiskit_ibm_runtime.exceptions import RuntimeJobFailureError, RuntimeInvalidStateError
from qiskit_ibm_runtime.quantum_program.quantum_program_result import QuantumProgramResult
from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder
from utils.samples import group_multirep_samples_dicts
import utils.execution as execution
import numpy as np
import warnings
import json
import os

class ExecDB:
    """Database for saving jobs"""
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            with open(path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = []
            with open(path, "w") as f:
                json.dump(self._data, f, indent=4)

    def reload(self):
        with open(self.path, "r") as f:
            self._data = json.load(f)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=4)

    def _search_batches_indices_by_params(self, batch_args, observable_func_name, limit=10):
        indices_to_return = []
        observable_func_name = observable_func_name.__name__ if callable(observable_func_name) else observable_func_name
        for i, batch in enumerate(self._data[::-1]):
            is_equal = ([
                batch.get(key, None) == val for key, val in batch_args.items()
            ])# + [
            #    batch["observables_func_name"] == observable_func_name
            #])
            if all(is_equal): indices_to_return.append(len(self._data) - 1 - i)
            if len(indices_to_return) >= limit: break
        return indices_to_return
    
    def _search_batch_index_by_id(self, id):
        for i, batch in enumerate(self._data):
            if batch["id"] == id:
                return i
        raise ValueError(f"No batch found with id: {id}")
    
    def _search_batch_index_by_data(self, data):
        for i, batch in enumerate(self._data):
            batch_without_id = batch.copy()
            del batch_without_id["id"]
            if batch_without_id == data:
                return i
        raise ValueError("No batch found with the provided data")

    def search_by_params(self, batch_args, observable_func_name, ibmq_service=None, limit=10):
        indices = self._search_batches_indices_by_params(batch_args, observable_func_name, limit)
        batches_to_return = [self._data[i] for i in indices]
        if ibmq_service is None:
            return batches_to_return
        else:
            jobs_objs = []
            for batch in batches_to_return:
                this_jobs_ids = [job["job_id"] for job in batch["jobs"]]
                this_jobs_objs = [ibmq_service.job(job_id=job_id) for job_id in this_jobs_ids]
                jobs_objs.append(this_jobs_objs)
            return jobs_objs

    def search_by_id(self, id, ibmq_service=None):
        data = self._data[self._search_batch_index_by_id(id)]
        if ibmq_service is None:
            return data
        else:
            job_ids = [job["job_id"] for job in data["jobs"]]
            job_objs =  [ibmq_service.job(job_id=job_id) for job_id in job_ids]
            return job_objs

    def seach_by_data(self, data, ibmq_service=None):
        data = self._search_batch_index_by_data(data)
        if ibmq_service is None:
            return data
        else:
            job_ids = [job["job_id"] for job in data["jobs"]]
            job_objs =  [ibmq_service.job(job_id=job_id) for job_id in job_ids]
            return job_objs
    
    @classmethod
    def generate_batch_db_data(cls, batch_args, physical_circuits, observable_generating_func_name, job_ids_arr):
        data_to_add = batch_args
        try:
            data_to_add["nqubits_arr"] = sorted(list({len(physical_circuit.layout.final_index_layout()) for physical_circuit in physical_circuits}))
        except AttributeError:
            pass
        data_to_add["depths_arr"] = sorted(list({physical_circuit.depth() for physical_circuit in physical_circuits}))
        if callable(observable_generating_func_name):
            data_to_add["observables_func_name"] = observable_generating_func_name.__name__
        else:
            data_to_add["observables_func_name"] = str(observable_generating_func_name)
        data_to_add["jobs"] = [{"job_id": jid, "nqubits": len(physical_circuit.layout.final_index_layout()) if physical_circuit.layout is not None else None, "depth": physical_circuit.depth(), "layout": physical_circuit.layout.final_index_layout() if physical_circuit.layout is not None else None} for jid, physical_circuit in zip(job_ids_arr, physical_circuits)]
        return data_to_add
    
    def add(self, batch_args, ran_circuits, observable_generating_func_name, job_ids_arr):
        thisid = 0 if len(self._data) == 0 else self._data[-1]["id"] + 1
        data_to_add = {"id": thisid}
        data_to_add = data_to_add | self.generate_batch_db_data(batch_args, ran_circuits, observable_generating_func_name, job_ids_arr)
        self.reload()
        self._data.append(data_to_add)
        self.save()

    def add_data(self, data):
        thisid = 0 if len(self._data) == 0 else self._data[-1]["id"] + 1
        data_contains_id = data.get("id", False)
        if data_contains_id:
            data_to_add = data.copy()
            data_to_add["id"] = thisid
        else:
            data_to_add = {"id": thisid} | data
        self.reload()
        self._data.append(data_to_add)
        self.save()

    def remove(self, id):
        index = self._search_batch_index_by_id(id)
        self._data.pop(index)
        self.save()

    def execute_sampler_batch(self, backend, sampler_opt_dict, job_circuits, cal_circuits=None, repeat_cal=False, db_options=None, share_total_shots=False, max_shots=75000):
        jobs = execution.execute_sampler_batch(backend, sampler_opt_dict, job_circuits, cal_circuits, repeat_cal, db_options, self, share_total_shots, max_shots)
        return jobs

    def execute_sampler_batch_choosing_layout(self, backend, sampler_opt_dict, job_log_circuits, cal_log_circuits=None, repeat_cal=False, db_options=None, share_total_shots=False, optimization_level=3, max_shots=75000):
        jobs = execution.execute_sampler_batch_choosing_layout(backend, sampler_opt_dict, job_log_circuits, cal_log_circuits, repeat_cal, db_options, self, share_total_shots, optimization_level, max_shots)
        return jobs

    def execute_sampler_batch_choosing_layout_by_batches(self, backend, sampler_opt_dict, job_log_circuits, batches_first_index, cal_log_circuits=None, jobcal_together=False, db_options=None, optimization_level=0):
        jobs = execution.execute_sampler_batch_choosing_layout_by_batches(backend, sampler_opt_dict, job_log_circuits, batches_first_index, cal_log_circuits, jobcal_together, db_options, self, optimization_level)
        return jobs

    def execute_znepea_choosing_layout(self, backend, job_log_barrier_circuits, parameters_func, db_options, uinst_filter_func=None, optimization_level=3):
        jobs = execution.execute_znepea_choosing_layout(backend, job_log_barrier_circuits, parameters_func, db_options, uinst_filter_func, self, optimization_level)
        return jobs

    # def load_final_samples_from_local_or_ibmq(self, batch_args, ibmq_service, jobs_result_folder, job_index=0, return_data=False, verbose=False):
    #     jobs_data = self.search_by_params(batch_args, "Sampler", ibmq_service=None, limit=job_index+1)
    #     try:
    #         jobs_data = jobs_data[job_index]
    #     except IndexError:
    #         raise ValueError(f"No batch found with the provided parameters")
    #     jobs_ids = [job["job_id"] for job in jobs_data["jobs"]]
    #     jobs_results = load_results_from_local_or_ibmq(jobs_ids, ibmq_service, jobs_result_folder, verbose=verbose)
    #     jobs_samples = [samples_from_result(job_result) for job_result in jobs_results]
    #     jobs_samples = np.array(jobs_samples, dtype=object)
    #     jobs_shape = jobs_data["jobs_shape"]
    #     nrepetitions = jobs_data["nrepetitions"]
    #     cal = jobs_data["cal"]
    #     repeat_cal = jobs_data["repeat_cal"]
    #     multirep_combined_samples = group_multirep_samples_dicts(jobs_samples, jobs_shape, nrepetitions, cal, repeat_cal)
    #     return (*multirep_combined_samples, *((jobs_data,) if return_data else ()))

    def load_final_samples_from_local_or_ibmq(self, batch_args, ibmq_service, jobs_result_folder, job_index=0, return_data=False, verbose=False):
        jobs_data = self.search_by_params(batch_args, "Sampler", ibmq_service=None, limit=job_index+1)
        try:
            jobs_data = jobs_data[job_index]
        except IndexError:
            raise ValueError(f"No batch found with the provided parameters")
        jobs_ids = [job["job_id"] for job in jobs_data["jobs"]]
        jobs_results = load_results_from_local_or_ibmq(jobs_ids, ibmq_service, jobs_result_folder, verbose=verbose)
        # jobs_samples = [samples_from_result(job_result) for job_result in jobs_results]
        jobs_samples = []
        for job_result in jobs_results:
            jobs_samples += samples_from_result(job_result)
        jobs_samples = np.array(jobs_samples, dtype=object)
        jobs_shape = jobs_data["job_circuits_shape"]
        cal_shape = jobs_data["cal_circuits_shape"]
        nrepetitions = jobs_data["nrepetitions"]
        repeat_cal = jobs_data["repeat_cal"]
        if cal_shape is None:
            multirep_combined_job_samples = group_multirep_samples_dicts(jobs_samples, jobs_shape, nrepetitions, cal=False)
        else:
            if jobs_shape == cal_shape:
                multirep_combined_job_samples, multirep_combined_cal_samples = group_multirep_samples_dicts(jobs_samples, jobs_shape, nrepetitions, cal=True, repeat_cal=repeat_cal)
            else:
                cal_nrepetitions = nrepetitions if repeat_cal else 1
                cal_samples = jobs_samples[:np.prod(cal_shape)*cal_nrepetitions].reshape(cal_shape[0]*cal_nrepetitions, cal_shape[1])
                jobs_samples = jobs_samples[np.prod(cal_shape)*cal_nrepetitions:].reshape(jobs_shape[0]*nrepetitions, jobs_shape[1])
                multirep_combined_job_samples = group_multirep_samples_dicts(jobs_samples, jobs_shape, nrepetitions, cal=False)
                multirep_combined_cal_samples = group_multirep_samples_dicts(cal_samples, cal_shape, cal_nrepetitions, cal=False)
        to_return = [multirep_combined_job_samples]
        if cal_shape is not None:
            to_return.append(multirep_combined_cal_samples)
        if return_data:
            to_return.append(jobs_data)
        return to_return

def load_job_result(filepath):
    with open(filepath, "r") as f:
        result = json.load(f, cls=RuntimeDecoder)
    return result

def save_job_result(filepath, job):
    result = job.result()
    with open(filepath, "w") as f:
        json.dump(result, f, cls=RuntimeEncoder)
    return result
    
def load_results_from_local_or_ibmq(job_ids, ibmq_service, jobs_result_folder="", verbose=False):
    job_results = []
    for job_id in job_ids:
        if verbose: print(f"\rLoading job {job_id}...".ljust(100), end="")
        if os.path.isdir(jobs_result_folder):
            job_filepath = os.path.join(jobs_result_folder, f"{job_id}.json")
            if os.path.isfile(job_filepath):
                job_result = load_job_result(job_filepath)
            else:
                job_obj = ibmq_service.job(job_id=job_id)
                try:
                    job_result = save_job_result(job_filepath, job_obj)
                except (RuntimeJobFailureError, RuntimeInvalidStateError):
                    job_result = None
        else:
            job_obj = ibmq_service.job(job_id=job_id)
            if job_obj.status() == "QUEUED":
                warnings.warn(f"Job {job_id} is still QUEUED. Waiting for it to complete...", UserWarning, stacklevel=2)
            if job_obj.status() == "RUNNING":
                warnings.warn(f"Job {job_id} is still RUNNING. Waiting for it to complete...", UserWarning, stacklevel=2)
            try:
                job_result = job_obj.result()
            except RuntimeJobFailureError:
                job_result = None
        job_results.append(job_result)
    return job_results

def samples_from_result(job_result):
    # if job_result is None:
    #     return {}
    # try:
    #     job_result = job_result[0]
    # except (IndexError, TypeError):
    #     pass
    # samples = list(dict(job_result.data).values())[0].get_counts()
    # return samples
    if job_result is None:
        return [{}]
    elif isinstance(job_result, QuantumProgramResult):
        samples = [res for res in job_result]
        return samples
    else:
        samples = [list(dict(res.data).values())[0].get_counts() for res in job_result]
        return samples