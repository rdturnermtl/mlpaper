# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, absolute_import, division
from builtins import range
import numpy as np
import pandas as pd
import scipy.stats as ss
from benchmark_tools.constants import (
    METHOD, METRIC, STAT, STD_STATS, PAIRWISE_DEFAULT)
import benchmark_tools.perf_curves as pc

# ============================================================================
# Statistical util functions
# ============================================================================

def ttest1(x, nan_on_zero=False):
    '''Perform a standard t-test to test if the values in `x` are sampled from
    a distribution with a zero mean.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        array of data points to test.
    nan_on_zero : bool
        If True, return a p-value of NaN for samples with zero variances,
        otherwise return a p-value of 1.0.

    Returns
    -------
    pval : float
        p-value (in [0,1]) from t-test on `x`.
    '''
    assert(np.ndim(x) == 1 and len(x) > 0)

    if np.std(x) == 0.0:
        pval = np.nan if nan_on_zero else 1.0
    else:
        _, pval = ss.ttest_1samp(x, 0.0)
    return pval


def t_EB(x, confidence=0.95):
    '''Get t statistic based error bars on mean of `x`.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Data points to estimate mean. Must not be empty or contain NaNs.
    confidence : float
        Confidence probability (in (0, 1)) to construct confidence interval
        from t statistic.

    Returns
    -------
    EB : float
        Size of error bar on mean (>= 0). The confidence interval is
        ``[mean(x) - EB, mean(x) + EB]``. EB is inf when ``len(x) = 1``.
    '''
    assert(np.ndim(x) == 1 and (not np.any(np.isnan(x))))
    assert(np.ndim(confidence) == 0)
    assert(0.0 < confidence and confidence < 1.0)

    N = len(x)
    if N <= 1:
        return np.inf

    # loc cancels out when we just want EB anyway
    LB, UB = ss.t.interval(confidence, N - 1, loc=0.0, scale=1.0)
    assert(not (LB > UB))
    # Just multiplying scale=ss.sem(x) is better for when scale=0
    EB = 0.5 * ss.sem(x) * (UB - LB)
    assert(np.ndim(EB) == 0 and not (EB < 0.0))
    return EB


def bernstein_EB(x, lower, upper, confidence=0.95):
    '''Bernstein version of t_EB, not yet used.'''
    assert(np.ndim(x) == 1 and (not np.any(np.isnan(x))))
    assert(np.all(lower <= x) and np.all(x <= upper))
    assert(np.ndim(confidence) == 0)
    assert(0.0 < confidence and confidence < 1.0)

    N = x.size
    range_ = upper - lower
    if N == 0:
        return range_

    delta = 1.0 - confidence
    A = np.log(3.0 / delta)
    EB = np.std(x) * np.sqrt((2.0 * A) / N) + (3.0 * A * range_) / N

    # Also get worst-case bound
    mu = np.mean(x)
    EB_worst_case = np.maximum(upper - mu, mu - lower)
    EB = np.minimum(EB, EB_worst_case)

    assert(np.ndim(EB) == 0 and not (EB < 0.0))
    return EB


def boot_EB(x, confidence=0.95, n_boot=1000):
    '''Bootstrap version of t_EB, not yet used.'''
    assert(np.ndim(x) == 1 and (not np.any(np.isnan(x))))
    N = x.size
    if N == 0:
        return np.inf

    # Setup levels for percentile function
    alpha = 0.5 * (1.0 - confidence)
    q_levels = (100.0 * alpha, 100.0 * (1.0 - alpha))

    p_BS = np.ones(N) / N
    weight = np.random.multinomial(N, p_BS, size=n_boot).T

    xw = np.mean(x[:, None] * weight, axis=0)
    LB = np.percentile(xw, q_levels[0])
    UB = np.percentile(xw, q_levels[1])

    mu = np.mean(x)
    EB = np.maximum(UB - mu, mu - LB)
    assert(np.ndim(EB) == 0 and not (EB < 0.0))
    return EB


def get_mean_and_EB(loss, loss_ref=0.0, confidence=0.95, min_EB=0.0,
                    lower=-np.inf, upper=np.inf, method='t'):
    '''Get mean loss and estimated error bar.

    Parameters
    ----------
    loss : ndarray, shape (n_samples,)
        Array of loss value where each entry is an independent prediction.
    loss_ref : float or array-like of shape (n_samples,)
        Reference values for losses. This may be the losses of another method
        on the same datapoints. If 1d must be of same size as loss. The error
        bars are constructed from ``loss - loss_ref`` (like paired test) which
        results in smaller error bars is the losses are positively correlated.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.
    min_EB : float
        Minimum size of resulting error bar regardless of the data in `loss`.
    lower : float
        Theoretically lowest possible value in `loss`. Used for construction of
        Bernstein bounds.
    upper : float
        Theoretically highest possible value in `loss`. Used for construction
        of Bernstein bounds.
    method : {'t', 'bernstein', 'boot'}
        Method to use for building error bar.

    Returns
    -------
    mu : float
        Estimated mean loss.
    EB : float
        Size of error bar on mean loss (EB > 0). The confidence interval is
        [mu - EB, mu + EB]. EB is inf when len(loss) = 1.
    '''
    assert(loss.ndim == 1 and np.ndim(loss_ref) <= 1)
    assert(np.ndim(min_EB) == 0)
    # EB subroutines all check for presence of nans
    mu = np.mean(loss)

    # Note we are computing CI on delta to reference!
    delta = loss - loss_ref
    if method == 't':
        EB = t_EB(delta, confidence=confidence)
    elif method == 'bernstein':
        EB = bernstein_EB(delta, lower, upper, confidence=confidence)
    elif method == 'boot':
        EB = boot_EB(delta, confidence=confidence)
    else:
        assert(False)

    EB = np.maximum(EB, min_EB)
    return mu, EB

# ============================================================================
# Loss summary: the main purpose of this file.
# ============================================================================


def loss_summary_table(loss_table, ref_method,
                       pairwise_CI=PAIRWISE_DEFAULT, confidence=0.95):
    '''Build table with mean and error bar summaries from a loss table that
    contains losses on a per data point basis.

    Parameters
    ----------
    loss_tbl : DataFrame, shape (n_samples, n_metrics * n_methods)
        DataFrame with loss of each method according to each loss function on
        each data point. The rows are the data points in `y` (that is the index
        matches `log_pred_prob_table`). The columns are a hierarchical index
        that is the cartesian product of loss x method. That is, the loss of
        method foo's prediction of ``y[5]`` according to loss function bar is
        stored in ``loss_tbl.loc[5, ('bar', 'foo')]``.
    ref_method : str
        Name of method that is used as reference point in paired statistical
        tests. This is usually some some of baseline method. `ref_method` must
        be found in the 2nd level of the columns of `loss_tbl`.
    pairwise_CI : bool
        If True, compute error bars on the mean of ``loss - loss_ref`` instead
        of just the mean of `loss`. This typically gives smaller error bars.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.

    Returns
    -------
    perf_tbl : DataFrame, shape (n_methods, n_metrics * 3)
        DataFrame with mean loss of each method according to each loss
        function. The rows are the methods. The columns are a hierarchical
        index that is the cartesian product of
        loss x (mean, error bar, p-value). That is,
        ``perf_tbl.loc['foo', 'bar']`` is a pandas series with
        (mean loss of foo on bar, corresponding error bar, statistical sig)
        The statistical significance is a p-value from a two-sided hypothesis
        test on the hypothesis H0 that foo has the same mean loss as the
        reference method `ref_method`.
    '''
    assert(loss_table.columns.names == (METRIC, METHOD))
    metrics, methods = loss_table.columns.levels
    assert(ref_method in methods)  # ==> len(methods) >= 1
    assert(len(loss_table) >= 1 and len(metrics) >= 1)
    # Could also test these are cartesian product if we wanted to be exhaustive

    col_names = pd.MultiIndex.from_product([metrics, STD_STATS],
                                           names=[METRIC, STAT])
    perf_tbl = pd.DataFrame(index=methods, columns=col_names, dtype=float)
    perf_tbl.index.name = METHOD
    for metric in metrics:
        loss_ref = loss_table.loc[:, (metric, ref_method)]
        assert(loss_ref.ndim == 1)  # Weird stuff happens if names not unique
        for method in methods:
            loss = loss_table.loc[:, (metric, method)]
            assert(loss.ndim == 1)  # Weird stuff happens if names not unique
            assert(not np.any(np.isnan(loss)))  # Would let method cheat

            # get_mean_and_EB() supports other EB metrics already, they could
            # be added here if needed.
            if pairwise_CI:
                mu, EB = get_mean_and_EB(loss, loss_ref, confidence)
                EB = np.nan if method == ref_method else EB
            else:
                mu, EB = get_mean_and_EB(loss, confidence)

            # This is two-sided, could include one-sided option too.
            pval = ttest1(loss - loss_ref, nan_on_zero=(method == ref_method))
            assert((method == ref_method) == np.isnan(pval))
            perf_tbl.loc[method, metric] = (mu, EB, pval)
    return perf_tbl
