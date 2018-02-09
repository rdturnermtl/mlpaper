# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np


def boot_weights(N, n_boot, epsilon=0):
    '''Sample weights for data points that makes it equivalent to bootstrap
    resampling of data points.

    Parameters
    ----------
    N : int
        Number of data points must be >= 1..
    n_boot : int
        Number of bootstrap replicates, must be >= 1.
    epsilon : int or float
        Minimum weight, typically 0 unless this creates numerical problems for
        a down stream algorithm in which case a value such as 1e-10 is used.

    Returns
    -------
    weight : ndarray, shape (n_boot, N)
        Weights equivalent to resampling for bootstrap algorithm.
    '''
    assert(N >= 1)
    assert(n_boot >= 1)

    p_BS = np.ones(N) / N
    weight = np.maximum(epsilon, np.random.multinomial(N, p_BS, size=n_boot))
    assert(weight.shape == (n_boot, N))
    return weight


def confidence_to_percentiles(confidence):
    '''Convert confidence level to percentiles in sampling distribution to
    build confidence interval.

    Parameters
    ----------
    confidence : float
        Confidence level, use 0.95 for 95% interval. Must be in (0,1).

    Returns
    -------
    LB : float
        Lower end quantile in (0,1).
    UB : float
        Upper end quantile in (0,1).

    Examples
    --------
    >>> confidence_to_percentiles(0.95)
    (2.5, 97.5)
    '''
    assert(np.ndim(confidence) == 0 and 0.0 < confidence and confidence < 1.0)

    alpha = 0.5 * (1.0 - confidence)
    LB, UB = 100.0 * alpha, 100.0 * (1.0 - alpha)
    return LB, UB


def percentile(boot_estimates, confidence=0.95):
    '''Build confidence interval using percentile boostrap method.

    Parameters
    ----------
    boot_estimates : ndarray, shape (n_boot, ...)
        Estimated quantity across different bootstrap replications.
    confidence : float
        Confidence level, use 0.95 for 95% interval. Must be in (0,1).

    Returns
    -------
    LB : ndarray, shape (...)
        Lower end of confidence interval.
    UB : ndarray, shape (...)
        Upper end of confidence interval.
    '''
    assert(boot_estimates.ndim >= 1)
    assert(not np.any(np.isnan(boot_estimates)))  # NaN ordering is arbitrary

    q_levels = confidence_to_percentiles(confidence)
    LB, UB = np.percentile(boot_estimates, q_levels, axis=0)
    assert(LB.shape == boot_estimates.shape[1:])
    assert(LB.shape == UB.shape)
    return LB, UB


def basic(boot_estimates, original_estimate, confidence=0.95):
    '''Build confidence interval using basic boostrap method.

    Parameters
    ----------
    boot_estimates : ndarray, shape (n_boot, ...)
        Estimated quantity across different bootstrap replications.
    original_estimate : ndarray, shape (...)
        Quantity estimated using original (non-bootstrap) data set.
    confidence : float
        Confidence level, use 0.95 for 95% interval. Must be in (0,1).

    Returns
    -------
    LB : ndarray, shape (...)
        Lower end of confidence interval.
    UB : ndarray, shape (...)
        Upper end of confidence interval.
    '''
    assert(boot_estimates.ndim >= 1)
    assert(boot_estimates.shape[1:] == np.shape(original_estimate))

    LB, UB = percentile(boot_estimates, confidence)
    LB, UB = 2 * original_estimate - UB, 2 * original_estimate - LB
    return LB, UB


def error_bar(boot_estimates, original_estimate, confidence=0.95):
    '''Build error bar using boostrap method. The results is the same
    regardless of whether the percentile or basic boostrap is used for CIs.

    Parameters
    ----------
    boot_estimates : ndarray, shape (n_boot,)
        Estimated quantity across different bootstrap replications.
    original_estimate : float
        Quantity estimated using original (non-bootstrap) data set.
    confidence : float
        Confidence level, use 0.95 for 95% interval. Must be in (0,1).

    Returns
    -------
    EB : float
        Error bar around the original estimate.
    '''
    assert(boot_estimates.ndim == 1)
    assert(np.ndim(original_estimate) == 0)

    LB, UB = percentile(boot_estimates, confidence)
    # This actually ends up the same whether we use basic or percentile
    EB = np.fmax(UB - original_estimate, original_estimate - LB)
    assert(not np.any(EB < 0.0))  # Allows nans
    # NaN EB only ever occurs when ref is infinite and so are some samples
    assert(np.all(np.isfinite(original_estimate) <= ~np.isnan(EB)))
    return EB


def significance(boot_estimates, ref):
    '''Perform a two-sided bootstrap based hypothesis test on whether the
    unknown quantity is equal to some reference.

    Parameters
    ----------
    boot_estimates : ndarray, shape (n_boot,)
        Estimated quantity across different bootstrap replications.
    ref : float of ndarray of shape (n_boot,)
        Reference value is in hypothesis test. Use a scalar value for a known
        reference value or a array of n_boot bootstraped value to perform a
        paired test against another unknown quantity.

    Returns
    -------
    pval : float
        Resulting p-value of hypothesis test in (0,1).
    '''
    assert(boot_estimates.ndim == 1)
    assert(np.ndim(ref) == 0 or ref.shape == boot_estimates.shape)
    assert(not np.any(np.isnan(boot_estimates)))  # NaN ordering is arbitrary
    assert(not np.any(np.isnan(ref)))  # NaN ordering is arbitrary

    pval = 2.0 * np.minimum(np.mean(boot_estimates <= ref),
                            np.mean(ref <= boot_estimates))
    # Only needed when some boot_estimates == ref exactly:
    pval = np.minimum(1.0, pval)
    return pval
