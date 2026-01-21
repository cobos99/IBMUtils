from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
import numpy as np

def combine_samples_dicts(samples_dicts):
    final_samples_dict = samples_dicts[0].copy()
    for samples_dict in samples_dicts[1::]:
        for state in samples_dict.keys():
            if final_samples_dict.get(state, None) is not None:
                final_samples_dict[state] += samples_dict[state]
            else:
                final_samples_dict[state] = samples_dict[state]
    return final_samples_dict

# def debug_combine_samples_dicts(dummy_strings):
#     combined = dummy_strings[0]
#     for i in dummy_strings[1::]:
#         combined += i
#     return combined

# def group_multirep_samples_dicts(samples_dicts, job_circuits_shape, nrepetitions, cal=False, repeat_cal=False):
#     if not nrepetitions.is_integer() or nrepetitions < 1:
#         raise ValueError("nrepetitions must be a positive integer")
#     nrepetitions = int(nrepetitions)
#     samples_dicts = np.array(samples_dicts, dtype=object).reshape((job_circuits_shape[0], -1))
#     if not cal: repeat_cal = False
#     if repeat_cal:
#         cal_samples = samples_dicts[:, 0::2].reshape((job_circuits_shape[0], job_circuits_shape[1], nrepetitions))
#         jobs_samples = samples_dicts[:, 1::2].reshape((job_circuits_shape[0], job_circuits_shape[1], nrepetitions))
#     else:
#         if cal:
#             cal_mask = np.full_like(samples_dicts, False, dtype=bool)
#             cal_mask[:, 0::(nrepetitions+1)] = True
#             job_mask = ~cal_mask
#             cal_samples = samples_dicts[cal_mask].reshape((job_circuits_shape[0], -1))
#             jobs_samples = samples_dicts[job_mask].reshape((job_circuits_shape[0], job_circuits_shape[1], nrepetitions))
#         else:
#             cal_samples = None
#             jobs_samples = samples_dicts.reshape((job_circuits_shape[0], job_circuits_shape[1], nrepetitions))
#     grouped_jobs_samples = []
#     if cal: grouped_cal_samples = []
#     for ilayout in range(jobs_samples.shape[0]):
#         for icircuit in range(jobs_samples.shape[1]):
#             grouped_jobs_samples.append(jobs_samples[ilayout, icircuit, :])
#             if cal:
#                 if repeat_cal:
#                     grouped_cal_samples.append(cal_samples[ilayout, icircuit, :])
#                 else:
#                     grouped_cal_samples.append([cal_samples[ilayout, icircuit]])

#     with mp.Pool(mp.cpu_count()) as pool:
#         final_job_samples = pool.map(combine_samples_dicts, grouped_jobs_samples)
#         if cal: final_cal_samples = pool.map(combine_samples_dicts, grouped_cal_samples)
#     final_job_samples = np.array(final_job_samples, dtype=object).reshape(job_circuits_shape)
#     if cal: final_cal_samples = np.array(final_cal_samples, dtype=object).reshape(job_circuits_shape)
#     if cal:
#         return final_job_samples, final_cal_samples
#     else:
#         return final_job_samples

def group_multirep_samples_dicts(samples_dicts, job_circuits_shape, nrepetitions, cal=False, repeat_cal=False):
    if not nrepetitions.is_integer() or nrepetitions < 1:
        raise ValueError("nrepetitions must be a positive integer")
    nrepetitions = int(nrepetitions)
    samples_dicts = np.array(samples_dicts, dtype=object)
    if not cal: repeat_cal = False
    if repeat_cal:
        cal_samples = samples_dicts[..., 0:2].reshape(-1, nrepetitions)
        job_samples = samples_dicts[..., 1::2].reshape(-1, nrepetitions)
    else:
        if cal:
            cal_mask = np.full_like(samples_dicts, False, dtype=bool)
            cal_mask[..., 0::(nrepetitions+1)] = True
            cal_samples = samples_dicts[cal_mask].reshape(-1, 1)
            job_samples = samples_dicts[~cal_mask].reshape(-1, nrepetitions)
        else:
            cal_samples = None
            job_samples = samples_dicts.reshape(-1, nrepetitions)
    # grouped_jobs_samples = np.full(job_circuits_shape, {}, dtype=object)
    # if cal: grouped_cal_samples = np.full(job_circuits_shape, {}, dtype=object)
    # for i in range(job_samples.shape[0]):
    #     grouped_jobs_samples.flat[i] = job_samples[i, :]
    #     if cal:
    #         grouped_cal_samples.flat[i] = cal_samples[i, :]

    with mp.Pool(mp.cpu_count()) as pool:
        grouped_jobs_samples = pool.map(combine_samples_dicts, [job_samples[i, :] for i in range(job_samples.shape[0])])
        if cal:
            grouped_cal_samples = pool.map(combine_samples_dicts, [cal_samples[i, :] for i in range(cal_samples.shape[0])])
    grouped_jobs_samples = np.array(grouped_jobs_samples, dtype=object).reshape(job_circuits_shape)
    if cal:
        grouped_cal_samples = np.array(grouped_cal_samples, dtype=object).reshape(job_circuits_shape)
        return grouped_jobs_samples, grouped_cal_samples
    else:
        return grouped_jobs_samples

# def multilayout_combine_samples_dicts(samples_dicts):
#     samples_dicts = np.array(samples_dicts)
#     icirc_indices = np.arange(samples_dicts.shape[1])
#     # with mp.get_context("fork").Pool(processes=int(mp.cpu_count()/2)) as pool:
#         # combined_samples_dicts = pool.starmap(combine_samples_dicts, [tuple(samples_dicts[:, i]) for i in icirc_indices])
#     combined_samples_dicts = []
#     for i in icirc_indices:
#         combined_samples_dicts.append(combine_samples_dicts(*samples_dicts[:, i]))
#     return np.array(combined_samples_dicts)

def multilayout_combine_samples_dicts(samples_dicts):
    samples_dicts = np.array(samples_dicts, dtype=object)
    final_samples_shape = samples_dicts.shape[1:]
    samples_to_combine = samples_dicts.T.reshape(-1, samples_dicts.shape[0])
    with mp.Pool(mp.cpu_count()) as pool:
        combined_samples_dicts = pool.map(combine_samples_dicts, [samples_to_combine[i, :] for i in range(samples_to_combine.shape[0])])
    combined_samples_dicts = np.array(combined_samples_dicts, dtype=object).reshape(final_samples_shape)
    return combined_samples_dicts

def threadsafe_samples_transfer(recovery_function, shared_mem_name, shared_mem_shape, registry_index, fargs):
    shmem = SharedMemory(name=shared_mem_name)
    samples = dict(np.ndarray(shared_mem_shape, dtype=object, buffer=shmem.buf)[registry_index])
    recovered_samples = recovery_function(samples, *fargs)
    return recovered_samples