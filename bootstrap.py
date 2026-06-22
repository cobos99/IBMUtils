import numpy as np
import scipy as sp


def bootstrap_per_shot_aggregate(per_shot_contribs, aggregator, confidence=0.95, n_resamples=1000, bootstrap_unit="pooled"):
    """Percentile bootstrap CI for an aggregator on a per-shot ±1/0 tensor.

    Parameters
    ----------
    per_shot_contribs : np.ndarray
        Tensor of shape ``(p_randomizations, ..., n_shots, n_obs)``. Each entry is +1 or
        -1 if the shot was kept (signed by the observable), or 0 if the shot was rejected
        by post-selection / gauge cutoff. Rejection is the same across observables for a
        given shot, i.e. a shot is either zero in every observable column or ±1 in every
        observable column.
    aggregator : callable
        ``aggregator(per_shot_tensor) -> np.ndarray``. Returns the central expectation
        value of shape ``(..., n_obs)``. Used here only as a probe to (a) learn the
        output shape and (b) measure any constant per-observable rescaling (e.g. TREX)
        that should also be applied to the bootstrap distribution.
    bootstrap_unit : {"pooled", "randomizations"}, default "pooled"
        Resampling unit:

        * ``"pooled"`` — all kept shots in a cell form one dataset of size K; resample
          K with replacement (binomial-equivalent). Captures shot noise only.
        * ``"randomizations"`` — compute each randomization's kept-fraction-weighted
          mean once, then resample those R per-randomization means with replacement
          (with their original weights). Treats the randomization (the twirl) as the
          independent statistical unit. Captures between-randomization variability,
          which is the dominant uncertainty in twirled estimators (PEA / TREX).

    Returns
    -------
    np.ndarray
        Confidence intervals of shape ``(*aggregator_output_shape, 2)`` with the trailing
        axis ``[lower, upper]`` (percentile interval at ``confidence`` level).
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError("Confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("Number of resamples must be at least 1")
    if bootstrap_unit not in ("pooled", "randomizations"):
        raise ValueError(f"bootstrap_unit must be 'pooled' or 'randomizations', got {bootstrap_unit!r}")

    arr = np.asarray(per_shot_contribs, dtype=float)
    if arr.ndim < 3:
        raise ValueError("per_shot_contribs must have shape (p_randomizations, ..., n_shots, n_obs)")
    R = arr.shape[0]
    nobs = arr.shape[-1]
    cell_shape = arr.shape[1:-2]                                  # whatever sits between R and (S, nobs)

    # Probe the aggregator on the original tensor for output shape and TREX-style scaling.
    probe = np.asarray(aggregator(per_shot_contribs), dtype=float)
    out_shape = probe.shape

    rng = np.random.default_rng()
    alpha = (1 - confidence) / 2

    # Per-cell bootstrap. Iterate over the ...-cell axes; each iteration handles
    # all observables vectorised over the resamples and observable axes.
    ci = np.empty((*out_shape, 2), dtype=float)
    cell_iter = np.ndindex(*cell_shape) if len(cell_shape) > 0 else [()]
    for idx in cell_iter:
        # Pull this cell's per-shot tensor, shape (R, S, nobs).
        cell = arr[(slice(None),) + idx + (slice(None), slice(None))]   # (R, S, nobs)

        # Recover the per-observable TREX-style scale factor by comparing the aggregator's
        # output for this cell against the unscaled pooled (or per-rand-weighted) mean.
        agg_for_cell = probe[idx] if len(cell_shape) > 0 else probe       # (nobs,)
        agg_for_cell = np.asarray(agg_for_cell, dtype=float)

        if bootstrap_unit == "pooled":
            flat = cell.reshape(-1, nobs)                                  # (R·S, nobs)
            kept = (flat[:, 0] != 0)
            K = int(kept.sum())
            if K == 0:
                ci[idx + (slice(None),)] = np.nan
                continue

            kept_rows = flat[kept]                                         # (K, nobs)
            n_plus = (kept_rows == 1).sum(axis=0)                          # (nobs,)
            p_plus = n_plus / K                                            # (nobs,)
            pre_scale = 2.0 * p_plus - 1.0                                 # (nobs,)

            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(np.abs(pre_scale) > 1e-15, agg_for_cell / pre_scale, 1.0)

            # Vectorised binomial bootstrap of the mean of K signed draws.
            k = rng.binomial(K, p_plus, size=(n_resamples, nobs))          # (B, nobs)
            boot = (2.0 * (k / K) - 1.0) * scale                           # (B, nobs)

        else:  # bootstrap_unit == "randomizations"
            # Per-randomization signed-sum and kept-count.
            kept_per_shot = (cell[..., 0] != 0)                            # (R, S)
            nkept_r = kept_per_shot.sum(axis=-1)                           # (R,)
            signed_sum_r = cell.sum(axis=-2)                               # (R, nobs)

            # Drop randomizations with zero kept shots from the resampling pool — they
            # carry no information and would contribute NaN to the per-rand mean.
            valid = nkept_r > 0                                            # (R,)
            n_valid = int(valid.sum())
            if n_valid == 0:
                ci[idx + (slice(None),)] = np.nan
                continue

            valid_idx = np.flatnonzero(valid)                              # (n_valid,)
            nkept_v = nkept_r[valid_idx].astype(float)                     # (n_valid,)
            signed_sum_v = signed_sum_r[valid_idx]                         # (n_valid, nobs)
            per_rand_mean = signed_sum_v / nkept_v[:, None]                # (n_valid, nobs), each in [-1, 1]

            # Recover the constant scale (TREX) by comparing the aggregator's output
            # against the kept-fraction-weighted mean computed here.
            total_kept = float(nkept_v.sum())
            weights_v = nkept_v / total_kept                               # (n_valid,)
            pre_scale = (per_rand_mean * weights_v[:, None]).sum(axis=0)   # (nobs,)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(np.abs(pre_scale) > 1e-15, agg_for_cell / pre_scale, 1.0)

            # Resample R randomizations with replacement; each picked rand carries its
            # own original kept-shot weight. The replicate's mean is sum(w · mean) / sum(w),
            # using the picked rands' weights and means.
            sampled = rng.integers(0, n_valid, size=(n_resamples, n_valid))   # (B, n_valid) of indices into valid_idx
            sampled_means   = per_rand_mean[sampled]                          # (B, n_valid, nobs)
            sampled_weights = nkept_v[sampled]                                # (B, n_valid)
            wsum = sampled_weights.sum(axis=-1, keepdims=True)                # (B, 1)
            wsum_safe = np.where(wsum > 0, wsum, 1.0)
            replicate_mean = (sampled_means * sampled_weights[..., None]).sum(axis=1) / wsum_safe   # (B, nobs)
            replicate_mean = np.where(wsum > 0, replicate_mean, np.nan)
            boot = replicate_mean * scale                                  # (B, nobs)

        ci_lo = np.nanpercentile(boot, 100 * alpha,       axis=0)
        ci_hi = np.nanpercentile(boot, 100 * (1 - alpha), axis=0)
        ci[idx + (slice(None), 0)] = ci_lo
        ci[idx + (slice(None), 1)] = ci_hi

    return ci


def bootstrap_sample_dict_mean(samples_dict, string_expectation_values, confidence=0.95, n_resamples=1000, return_std=False):
    """Percentile bootstrap CI for the mean of a two-valued (e.g. ±1) statistic.

    Each column of ``string_expectation_values`` must take only two distinct values
    (the two eigenvalues of the observable). Under that constraint the bootstrap
    distribution of the mean is exactly Binomial(N, p_plus): drawing N items with
    replacement from a two-valued dataset and averaging is equivalent to flipping
    a biased coin N times. We therefore replace the per-replicate ``np.random.choice``
    over N items with one vectorised ``rng.binomial`` call, which is the same
    bootstrap distribution sampled efficiently — not an approximation.

    Returns
    -------
    np.ndarray
        Shape ``(nobs, 2)``; each row is ``[ci_lower, ci_upper]`` (percentile interval).
    """
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("Number of resamples must be at least 1")

    string_counts = np.asarray(list(samples_dict.values()))
    N = int(np.sum(string_counts))
    nobs = string_expectation_values.shape[1]
    unique_vals = np.unique(string_expectation_values)
    if len(unique_vals) == 1:
        v = float(unique_vals[0])
        ci = np.repeat([[v, v]], nobs, axis=0)
        if return_std:
            return ci, [0.0]*nobs
        return ci
    if len(unique_vals) != 2:
        raise ValueError("string_expectation_values must take at most two unique values per column")
    v_lo, v_hi = float(unique_vals[0]), float(unique_vals[1])
    span = v_hi - v_lo

    # n_hi[i] = total count of bitstrings whose i-th observable takes value v_hi.
    is_hi = (string_expectation_values == v_hi)             # (nstrings, nobs) bool
    n_hi = (is_hi * string_counts[:, None]).sum(axis=0)     # (nobs,)
    p_hi = n_hi / N                                          # (nobs,) bootstrap success prob

    rng = np.random.default_rng()
    # One draw of (n_resamples, nobs) integers from Binomial(N, p_hi[obs]).
    k = rng.binomial(N, p_hi, size=(n_resamples, nobs))      # (B, nobs)
    boot = v_lo + (k / N) * span                             # (B, nobs) — replicate means

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(boot, 100 * alpha,       axis=0)
    ci_upper = np.percentile(boot, 100 * (1 - alpha), axis=0)
    ci = np.stack([ci_lower, ci_upper], axis=-1)             # (nobs, 2)

    if return_std:
        return ci, list(np.std(boot, axis=0))
    return ci


def bootstrap_two_valued_mean(per_shot_signed, confidence=0.95, n_resamples=1000):
    """Pooled-bootstrap CI for the mean of a two-valued (±1) per-shot tensor.

    Equivalent to ``bootstrap_sample_dict_mean`` but skips the upstream
    ``Counter(map(tuple, ...))`` step: feed the per-shot ±1 tensor directly.

    Parameters
    ----------
    per_shot_signed : np.ndarray, shape (N, nobs)
        Per-shot signed contributions taking values in {-1, +1} (or any two values).
    confidence : float in (0, 1)
    n_resamples : int

    Returns
    -------
    np.ndarray
        Shape ``(nobs, 2)``; each row is ``[ci_lower, ci_upper]`` (percentile interval).
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError("Confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("Number of resamples must be at least 1")

    arr = np.asarray(per_shot_signed)
    if arr.ndim != 2:
        raise ValueError("per_shot_signed must be 2D: (N, nobs)")
    N, nobs = arr.shape
    rng = np.random.default_rng()

    # Determine each observable's two values from the column extrema. We do not assume
    # all columns share the same pair; per-column max/min handles ±1, ±phase, etc.
    col_max = arr.max(axis=0)
    col_min = arr.min(axis=0)
    span = col_max - col_min                                 # (nobs,)
    # If a column is constant, the bootstrap CI is degenerate at that constant value.
    constant_mask = (span == 0)
    p_hi = np.where(constant_mask, 0.0, ((arr == col_max) * 1.0).sum(axis=0) / N)

    k = rng.binomial(N, p_hi, size=(n_resamples, nobs))      # (B, nobs)
    boot = col_min + (k / N) * span                          # (B, nobs)

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(boot, 100 * alpha,       axis=0)
    ci_upper = np.percentile(boot, 100 * (1 - alpha), axis=0)
    # For constant columns, force the CI to that constant (percentile of constant span = same value,
    # but be defensive against any binomial draw producing exactly k=0 across a 0-span column).
    if constant_mask.any():
        ci_lower = np.where(constant_mask, col_max, ci_lower)
        ci_upper = np.where(constant_mask, col_max, ci_upper)
    return np.stack([ci_lower, ci_upper], axis=-1)            # (nobs, 2)