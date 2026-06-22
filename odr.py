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

def renormalize_observable(measured_observable_arr, p_arr, observable_trace, nqubits, observable_range=None, observable_confidence=None, expected_observable_arr=None, cal_confidence=None, cal_observable_arr=None, nantol=0.2):
    # CI propagation (when observable_confidence is provided):
    #   O_clean = (O_sim - p*c) / (1 - p),   c = trace/2**n
    #   J_sim = ∂O_clean/∂O_sim = 1/(1 - p)
    #   J_p   = ∂O_clean/∂p     = (O_sim - c)/(1 - p)²
    # Cal enters via p = (O_cal - expected)/(c - expected), so
    #   δp from cal half-width δO_cal is δO_cal / (c - expected),
    # and the resulting O_clean half-width is |J_p| * δp.
    # Sim and cal half-widths are combined in quadrature per side
    # (independent bootstraps). When the cal CI is not supplied we
    # fall back to sim-only propagation rather than reusing the sim
    # CI as a cal proxy.
    try:
        try:
            one_mask = np.isclose(p_arr, 1)
            p_arr[one_mask] = np.nan
        except TypeError:
            if one_mask == 1:
                p_arr = np.nan
        c = observable_trace/2**nqubits
        candidate = ((np.array(measured_observable_arr) - np.array(p_arr)*c) / (1 - np.array(p_arr))).real
        if observable_confidence is not None:
            if expected_observable_arr is None:
                raise ValueError("expected_observable_arr must be provided if observable_confidence is provided")
            if (cal_confidence is None) != (cal_observable_arr is None):
                raise ValueError("cal_confidence and cal_observable_arr must be provided together")
            new_confidence = np.zeros_like(observable_confidence, dtype=float)
            try:
                # Coerce to float once. Upstream layers (e.g. many_measure_observables_executor)
                # sometimes hand back dtype=object arrays of native Python floats, which numpy
                # ufuncs like np.sqrt refuse to broadcast over.
                meas_f          = np.asarray(measured_observable_arr, dtype=float)
                obs_conf_f      = np.asarray(observable_confidence,   dtype=float)
                p_f             = np.asarray(p_arr,                   dtype=float)
                expected_f      = np.asarray(expected_observable_arr, dtype=float)

                # Sim-side half-widths (lo on the lower side, hi on the upper).
                sim_lo_hw = meas_f - obs_conf_f[..., 0]
                sim_hi_hw = obs_conf_f[..., 1] - meas_f
                sim_lo_hw = np.where(sim_lo_hw > 0, sim_lo_hw, 0.0)
                sim_hi_hw = np.where(sim_hi_hw > 0, sim_hi_hw, 0.0)

                inv_one_minus_p = 1.0 / (1 - p_f)
                jac_sim = inv_one_minus_p                                  # ∂O_clean/∂O_sim
                jac_p   = (meas_f - c) * inv_one_minus_p**2                # ∂O_clean/∂p

                # Sim contribution to each side (delta is non-negative).
                sim_lo_dev = np.abs(jac_sim) * sim_lo_hw
                sim_hi_dev = np.abs(jac_sim) * sim_hi_hw

                if cal_confidence is not None:
                    cal_f      = np.asarray(cal_observable_arr, dtype=float)
                    cal_conf_f = np.asarray(cal_confidence,     dtype=float)

                    cal_lo_hw = cal_f - cal_conf_f[..., 0]
                    cal_hi_hw = cal_conf_f[..., 1] - cal_f
                    cal_lo_hw = np.where(cal_lo_hw > 0, cal_lo_hw, 0.0)
                    cal_hi_hw = np.where(cal_hi_hw > 0, cal_hi_hw, 0.0)

                    # δp from each cal side. Sign of (c - expected) is fixed (>0 if expected<c, else <0)
                    # but we only need magnitudes here because the asymmetric pairing below picks
                    # which cal side feeds the lower vs upper O_clean bound.
                    inv_cal_denom = 1.0 / (c - expected_f)
                    p_lo_hw = cal_lo_hw * np.abs(inv_cal_denom)            # δp when O_cal hits its lower bound
                    p_hi_hw = cal_hi_hw * np.abs(inv_cal_denom)            # δp when O_cal hits its upper bound

                    # A higher p (driven by the cal side that increases p) pushes O_clean in the
                    # direction sign(J_p). When (c - expected) > 0, "upper cal" raises p; when
                    # (c - expected) < 0, "upper cal" lowers p. The product sign(J_p) * sign(1/(c-expected))
                    # tells us which of (cal_lo, cal_hi) ends up on the upper-O_clean side.
                    sign_jac_p   = np.sign(jac_p)
                    sign_cal_den = np.sign(inv_cal_denom)
                    upper_uses_cal_hi = (sign_jac_p * sign_cal_den) >= 0   # broadcast-safe

                    cal_dev_to_upper = np.where(upper_uses_cal_hi, np.abs(jac_p) * p_hi_hw, np.abs(jac_p) * p_lo_hw)
                    cal_dev_to_lower = np.where(upper_uses_cal_hi, np.abs(jac_p) * p_lo_hw, np.abs(jac_p) * p_hi_hw)
                else:
                    cal_dev_to_upper = 0.0
                    cal_dev_to_lower = 0.0

                # Quadrature per side (sim and cal assumed independent).
                lower_dev = np.sqrt(sim_lo_dev**2 + cal_dev_to_lower**2)
                upper_dev = np.sqrt(sim_hi_dev**2 + cal_dev_to_upper**2)
            except ValueError:
                raise ValueError("measured_observable_arr expected_observable_arr must be broadcastable")
            new_confidence[..., 0] = candidate - lower_dev
            new_confidence[..., 1] = candidate + upper_dev
        if observable_range is not None:
            mask = np.logical_or(candidate < (1+nantol)*observable_range[0] - nantol, candidate > (1+nantol)*observable_range[1] + nantol)
            candidate[mask] = np.nan
            if observable_confidence is not None:
                new_confidence[mask, :] = np.nan
                return candidate, new_confidence
            else:
                return candidate
        else:
            if observable_confidence is not None:
                return candidate, new_confidence
            else:
                return candidate
    except ValueError:
        raise ValueError("measured_observable_arr, expected_observable_arr and observable_trace must be broadcastable")