import numpy as np
import scipy as sp

def bootstrap_sample_dict_mean(samples_dict, string_expectation_values, confidence=0.95, n_resamples=1000, return_std=False):
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between 0 and 1")
    if n_resamples < 1:
        raise ValueError("Number of resamples must be at least 1")
    if string_expectation_values.ndim != 2:
        raise ValueError("string_expectation_values must be a 2D array")

    string_counts = np.array(list(samples_dict.values()), dtype=np.int64)
    if string_expectation_values.shape[0] != len(string_counts):
        raise ValueError("string_expectation_values must have one row per string in samples_dict")
    total_string_counts = int(np.sum(string_counts))
    if total_string_counts < 1:
        raise ValueError("samples_dict must contain at least one count")

    n_observables = string_expectation_values.shape[1]
    unique_expectation_values = np.unique(string_expectation_values)
    if len(unique_expectation_values) == 1:
        confidence_intervals = np.repeat([[unique_expectation_values[0], unique_expectation_values[0]]], n_observables, axis=0)
        if return_std:
            return confidence_intervals, np.zeros(n_observables)
        return confidence_intervals
    elif len(unique_expectation_values) != 2:
        raise ValueError("string_expectation_values must be a 2D array with two unique values")
    eig_low, eig_high = float(unique_expectation_values[0]), float(unique_expectation_values[1])

    # Counts of each of the two eigenvalues, per observable
    observables_eigenvalues_counts = np.zeros((2, n_observables), dtype=np.int64)
    observables_eigenvalues_counts[0] = string_counts @ np.equal(string_expectation_values, eig_low)
    observables_eigenvalues_counts[1] = total_string_counts - observables_eigenvalues_counts[0]

    alpha = (1 - confidence)/2
    z_alpha = sp.special.ndtri(alpha)

    confidence_intervals = []
    stds = []
    for i in range(n_observables):
        n_low = int(observables_eigenvalues_counts[0, i])
        n_high = int(observables_eigenvalues_counts[1, i])
        original_expectation_value = (n_low*eig_low + n_high*eig_high) / total_string_counts

        # Bootstrap resampling. Every resample draws total_string_counts shots from the
        # two-point empirical distribution, so its number of high eigenvalues is binomial.
        resampled_n_high = np.random.binomial(total_string_counts, n_high / total_string_counts, size=n_resamples)
        bootstrap_set = (eig_low*(total_string_counts - resampled_n_high) + eig_high*resampled_n_high) / total_string_counts
        stds.append(np.std(bootstrap_set))

        # Correct for bias
        percentile = (np.sum(np.less(bootstrap_set, original_expectation_value)) + np.sum(np.less_equal(bootstrap_set, original_expectation_value))) / 2 / n_resamples
        if percentile == 0 or percentile == 1:
            confidence_intervals.append((original_expectation_value, original_expectation_value))
            continue
        z0_hat = sp.special.ndtri(percentile)

        # Correct for acceleration. The statistic takes only two values, so the
        # total_string_counts leave-one-out replicates collapse to n_low copies of
        # jackknife_low and n_high copies of jackknife_high.
        if n_low == 0 or n_high == 0 or total_string_counts < 2:
            a_hat = 0.0
        else:
            jackknife_low = ((n_low - 1)*eig_low + n_high*eig_high) / (total_string_counts - 1)
            jackknife_high = (n_low*eig_low + (n_high - 1)*eig_high) / (total_string_counts - 1)
            jackknife_mean = (n_low*jackknife_low + n_high*jackknife_high) / total_string_counts
            U_low = (total_string_counts - 1) * (jackknife_mean - jackknife_low)
            U_high = (total_string_counts - 1) * (jackknife_mean - jackknife_high)
            num = (n_low*U_low**3 + n_high*U_high**3) / total_string_counts**3
            den = (n_low*U_low**2 + n_high*U_high**2) / total_string_counts**2
            a_hat = num / (6 * den**(3/2))

        num1 = z0_hat + z_alpha
        alpha1 = sp.special.ndtr(z0_hat + num1 / (1 - a_hat * num1))
        num2 = z0_hat - z_alpha
        alpha2 = sp.special.ndtr(z0_hat + num2 / (1 - a_hat * num2))

        ci_l = np.percentile(bootstrap_set, 100*alpha1)
        ci_u = np.percentile(bootstrap_set, 100*alpha2)

        confidence_intervals.append((ci_l, ci_u))

    if return_std:
        return np.array(confidence_intervals), np.array(stds)
    else:
        return np.array(confidence_intervals)
