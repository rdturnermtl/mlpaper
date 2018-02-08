# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np


def boot_weights(N, n_boot, epsilon=0):
    # TODO grep for multinomial occurances that can be replaced with this

    p_BS = np.ones(N) / N
    weight = np.maximum(epsilon, np.random.multinomial(N, p_BS, size=n_boot))
    assert(weight.shape == (n_boot, N))
    return weight


def stratified_boot_weights(y, n_boot, epsilon=0):
    # TODO assert weight has same type as what comes from boot weights
    weight = np.full((n_boot, y.size), epsilon)  # preserve epsilon dtype

    labels = np.unique(y)
    for ll in labels:
        idx = (y == ll)
        N = np.sum(idx)
        weight[:, idx] = boot_weights(N, n_boot, epsilon=epsilon)
    return weight


def confidence_to_percentiles(confidence):
    assert(np.ndim(confidence) == 0 and 0.0 < confidence and confidence < 1.0)
    # TODO move to util

    alpha = 0.5 * (1.0 - confidence)
    LB, UB = 100.0 * alpha, 100.0 * (1.0 - alpha)
    return LB, UB


def percentile(boot_estimates, confidence=0.95):
    assert(boot_estimates.ndim >= 1)
    assert(not np.any(np.isnan(boot_estimates)))  # NaN ordering is arbitrary

    q_levels = confidence_to_percentiles(confidence)
    LB, UB = np.percentile(boot_estimates, q_levels, axis=0)
    assert(LB.shape == boot_estimates.shape[1:])
    assert(LB.shape == UB.shape)
    return LB, UB


def basic(boot_estimates, original_estimate, confidence=0.95):
    assert(boot_estimates.ndim >= 1)
    assert(boot_estimates.shape[1:] == np.shape(original_estimate))

    LB, UB = percentile(boot_estimates, confidence)
    LB, UB = 2 * original_estimate - UB, 2 * original_estimate - LB
    return LB, UB


def error_bar(boot_estimates, original_estimate, confidence=0.95):
    assert(boot_estimates.ndim == 1)
    assert(boot_estimates.shape[1:] == np.shape(original_estimate))

    LB, UB = percentile(boot_estimates, confidence)
    # This actually ends up the same whether we use basic or percentile
    EB = np.fmax(UB - original_estimate, original_estimate - LB)
    assert(not np.any(EB < 0.0))  # Allows nans
    # NaN EB only ever occurs when ref is infinite and so are some samples
    assert(np.all(np.isfinite(original_estimate) <= ~np.isnan(EB)))
    return EB


def significance(boot_estimates, ref):
    assert(boot_estimates.ndim == 1)
    assert(np.ndim(ref) == 0 or ref.shape == boot_estimates.shape)
    assert(not np.any(np.isnan(boot_estimates)))  # NaN ordering is arbitrary
    assert(not np.any(np.isnan(ref)))  # NaN ordering is arbitrary

    pval = 2.0 * np.minimum(np.mean(boot_estimates <= ref),
                            np.mean(ref <= boot_estimates))
    pval = np.minimum(1.0, pval)  # Only needed when some auc == ref exactly
    return pval
