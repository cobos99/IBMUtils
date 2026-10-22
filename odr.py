import warnings

import numpy as np

def compute_depolarizing_probability(measured_observable_arr, expected_observable_arr, observable_trace, nqubits):
    # PRE 104, 035309 (2021)
    try:
        return (np.array(measured_observable_arr) - np.array(expected_observable_arr))/(observable_trace/2**nqubits - np.array(expected_observable_arr)).real
    except ValueError:
        raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")

# def compute_depolarizing_probability_std(measured_observable_std_arr, expected_observable_arr, observable_trace, nqubits):
#     try:
#         return np.abs(measured_observable_std_arr/(observable_trace/2**nqubits - np.array(expected_observable_arr)))
#     except ValueError:
#         raise ValueError("measured_observable_arr expected_observable_arr and observable_trace must be broadcastable")

def renormalize_observable(measured_observable_arr, p_arr, observable_trace, nqubits, observable_range=None, observable_confidence=None, expected_observable_arr=None, nantol=0.2, calibration_confidence=None):
    """Rescale a measured observable by the depolarizing probability of its calibration circuit.

    x = (m - p t) / (1 - p),  t = observable_trace / 2**nqubits,
    p = (c - E) / (t - E)  (compute_depolarizing_probability), with c the measured calibration
    observable and E = expected_observable_arr its noiseless value.

    Confidence interval (first-order, job and calibration shots independent):
        dx/dm = 1 / (1 - p)
        dx/dc = (m - t) / ((1 - p)**2 (t - E))
        lower/upper deviation of x = sqrt( (dx/dm * dm)**2 + (dx/dc * dc)**2 ),
    where dm comes from observable_confidence and dc from calibration_confidence, each taken
    on the side that moves x in the corresponding direction.

    calibration_confidence: (..., 2) interval [lower, upper] of the calibration observable c,
        same shape as observable_confidence. If None, the job interval is used in its place
        (equal-shot approximation) and a warning is issued.

    p_arr is not modified. Entries with p = 1 return NaN.
    """
    try:
        p_arr = np.asarray(p_arr)
        p_arr = np.where(np.isclose(p_arr, 1), np.nan, p_arr)
        measured_observable_arr = np.asarray(measured_observable_arr)
        trace_term = observable_trace/2**nqubits
        candidate = np.asarray(((measured_observable_arr - p_arr*trace_term) / (1 - p_arr)).real)
        if observable_confidence is not None:
            if expected_observable_arr is None:
                raise ValueError("expected_observable_arr must be provided if observable_confidence is provided")
            observable_confidence = np.asarray(observable_confidence)
            expected_observable_arr = np.asarray(expected_observable_arr)
            if calibration_confidence is None:
                warnings.warn("renormalize_observable: no calibration_confidence given; using the job interval "
                              "for the calibration observable (equal-shot approximation).", stacklevel=2)
                calibration_confidence = observable_confidence
                calibration_center = measured_observable_arr
            else:
                calibration_confidence = np.asarray(calibration_confidence)
                # Calibration observable implied by p: c = E + p (t - E)
                calibration_center = (expected_observable_arr + p_arr*(trace_term - expected_observable_arr)).real
            new_confidence = np.zeros_like(observable_confidence)
            try:
                dx_dm = (1 / (1 - p_arr)).real
                dx_dc = ((measured_observable_arr - trace_term) / ((1 - p_arr)**2 * (trace_term - expected_observable_arr))).real
                dm_lower = np.abs(measured_observable_arr - observable_confidence[..., 0])
                dm_upper = np.abs(observable_confidence[..., 1] - measured_observable_arr)
                dc_lower = np.abs(calibration_center - calibration_confidence[..., 0])
                dc_upper = np.abs(calibration_confidence[..., 1] - calibration_center)
                # Lower bound of x: move m (and c) in the direction that decreases x
                job_lower = np.abs(dx_dm) * np.where(dx_dm >= 0, dm_lower, dm_upper)
                job_upper = np.abs(dx_dm) * np.where(dx_dm >= 0, dm_upper, dm_lower)
                cal_lower = np.abs(dx_dc) * np.where(dx_dc >= 0, dc_lower, dc_upper)
                cal_upper = np.abs(dx_dc) * np.where(dx_dc >= 0, dc_upper, dc_lower)
                left_obs_deviation = np.sqrt(job_lower**2 + cal_lower**2)
                right_obs_deviation = np.sqrt(job_upper**2 + cal_upper**2)
            except ValueError:
                raise ValueError("measured_observable_arr expected_observable_arr must be broadcastable")
            new_confidence[..., 0] = candidate - left_obs_deviation
            new_confidence[..., 1] = candidate + right_obs_deviation
        if observable_range is not None:
            mask = np.logical_or(candidate < (1+nantol)*observable_range[0] - nantol, candidate > (1+nantol)*observable_range[1] + nantol)
            candidate[mask] = np.nan
            if observable_confidence is not None:
                new_confidence[mask, :] = np.nan
        if candidate.ndim == 0:
            candidate = candidate[()]
        if observable_confidence is not None:
            return candidate, new_confidence
        else:
            return candidate
    except ValueError:
        raise ValueError("measured_observable_arr, expected_observable_arr and observable_trace must be broadcastable")
