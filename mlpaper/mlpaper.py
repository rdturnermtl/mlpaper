# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import scipy.stats as ss

import mlpaper.boot_util as bu
from mlpaper.constants import ERR_COL, MEAN_COL, METHOD, METRIC, PAIRWISE_DEFAULT, PVAL_COL, STAT, STD_STATS
from mlpaper.util import clip_chk

N_BOOT = 1000  # Default number of bootstrap replications

# ============================================================================
# Statistical util functions
# ============================================================================


def clip_EB(mu, EB, lower=-np.inf, upper=np.inf, min_EB=0.0):
    """Clip error bars to both a minimum uncertainty level and a maximum level
    determined by trivial error bars from the a prior known limits of the
    unknown parameter `theta`. Similar to `np.clip`, but for error bars.

    Parameters
    ----------
    mu : float
        Point estimate of unknown parameter `theta` around which error bars are
        based.
    EB : float
        Size of error bar around `mu` (``EB > 0``). The confidence interval on
        `theta` is ``[mu - EB, mu + EB]``.
    lower : float
        A priori known theoretical lower limit on unknown parameter `theta`.
        For instance, for mean zero-one loss, ``lower=0``.
    upper : float
        A priori known theoretical upper limit on unknown parameter `theta`.
        For instance, for mean zero-one loss, ``upper=1``.
    min_EB : float
        Minimum size beleivable size of error bar. Typically, leave
        ``min_EB=0`` for simplicity.

    Returns
    -------
    EB : float
        Error bar after possible clipping.
    """
    assert np.ndim(mu) == 0 and np.ndim(EB) == 0
    assert np.ndim(lower) == 0 and np.ndim(upper) == 0
    assert upper - lower >= 0.0  # Also catch (inf, inf) or nans
    assert np.ndim(min_EB) == 0
    assert 0.0 <= min_EB and min_EB < np.inf

    # Note: These conditions are designed to pass when NaNs are supplied.
    if lower > mu or mu > upper:
        raise ValueError("mu %f outside of given limits (%f, %f)" % (mu, lower, upper))
    if 2 * min_EB > upper - lower:
        raise ValueError("min error bar %f too small for limits (%f, %f)" % (min_EB, lower, upper))

    with np.errstate(invalid="ignore"):  # expect non-finite here
        EB_trivial = np.fmax(upper - mu, mu - lower)
    assert not (min_EB > EB_trivial)  # Let NaNs pass
    EB = np.clip(EB, min_EB, EB_trivial)
    return EB


def t_test(x):
    """Perform a standard t-test to test if the values in `x` are sampled from
    a distribution with a zero mean.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        array of data points to test.

    Returns
    -------
    pval : float
        p-value (in [0,1]) from t-test on `x`.
    """
    assert np.ndim(x) == 1 and (not np.any(np.isnan(x)))

    if (len(x) <= 1) or (not np.all(np.isfinite(x))):
        return 1.0  # Can't say anything about scale => p=1

    _, pval = ss.ttest_1samp(x, 0.0)
    if np.isnan(pval):
        # Should only be possible if scale underflowed to zero:
        assert np.var(x, ddof=1) <= 1e-100
        # It is debatable if the condition should be ``np.mean(x) == 0.0`` or
        # ``np.all(x == 0.0)``. Should not matter in practice.
        pval = np.float(np.mean(x) == 0.0)
    assert 0.0 <= pval and pval <= 1.0
    return pval


def t_EB(x, confidence=0.95):
    """Get t statistic based error bars on mean of `x`.

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
        ``[mean(x) - EB, mean(x) + EB]``. `EB` is inf when ``len(x) <= 1``.
    """
    assert np.ndim(x) == 1 and (not np.any(np.isnan(x)))
    assert np.ndim(confidence) == 0
    assert 0.0 < confidence and confidence < 1.0

    N = len(x)
    if (N <= 1) or (not np.all(np.isfinite(x))):
        return np.inf

    # loc cancels out when we just want EB anyway
    LB, UB = ss.t.interval(confidence, N - 1, loc=0.0, scale=1.0)
    assert not (LB > UB)
    # Just multiplying scale=ss.sem(x) is better for when scale=0
    EB = 0.5 * ss.sem(x) * (UB - LB)
    assert np.ndim(EB) == 0 and EB >= 0.0
    return EB


def bernstein_test(x, lower, upper):
    """Perform Bernstein bound-based test to test if the values in `x` are
    sampled from a distribution with a zero mean. This test makes no
    distributional or central limit theorem assumption on `x`.

    As a result the bound may be loose and the p-value will not be sampled from
    a uniform distribution under H0 (E[x] = 0), but rather be skewed larger
    than uniform.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        array of data points to test.
    lower : float
        A priori known theoretical lower limit on unknown mean. For instance,
        for mean zero-one loss, ``lower=0``.
    upper : float
        A priori known theoretical upper limit on unknown mean. For instance,
        for mean zero-one loss, ``upper=1``.

    Returns
    -------
    pval : float
        p-value (in [0,1]) from t-test on `x`.
    """
    assert np.ndim(x) == 1 and (not np.any(np.isnan(x)))
    assert np.ndim(lower) == 0 and np.ndim(upper) == 0
    range_ = upper - lower
    assert range_ >= 0.0  # Also catch (inf, inf) or nans
    assert np.all(lower <= x) and np.all(x <= upper)

    if (len(x) <= 1) or (not np.all(np.isfinite(x))):
        return 1.0  # Can't say anything about scale => p=1
    if (range_ == 0.0) or (range_ == np.inf):
        # If range_ = inf, we could use p=0, if 0 is outside of [lower, upper],
        # but it is unclear if there is any advantage to the extra hassle.
        # If range_ = 0, then roots not invertible and distn on data x is a
        # point mass => everything has p=1.
        return 1.0

    # Get the moments
    N = len(x)
    mu = np.mean(x)
    std = np.std(x, ddof=0)

    coef = [(3.0 * range_) / N, std * np.sqrt(2.0 / N), -np.abs(mu)]
    assert np.all(np.isfinite(coef))  # Should have caught non-finite cases
    coef_roots = np.roots(coef)
    assert len(coef_roots) == 2
    assert coef_roots.dtype.kind == "f"  # Appears roots are always real
    # Appears to always be one neg and one pos root, but we looking for square
    # root so the positive one is the correct one. The roots can be zero.
    assert np.sum(coef_roots <= 0.0) >= 1
    assert np.sum(coef_roots >= 0.0) >= 1
    B = np.max(coef_roots) ** 2  # Bernstein test statistic
    # Sampling CDF is bounded by exponential for any true distn on x.
    delta = 3.0 * np.exp(-B)

    pval = np.minimum(1.0, delta)  # Can cap at 1 to make p-value
    assert 0.0 <= pval and pval <= 1.0
    return pval


def bernstein_EB(x, lower, upper, confidence=0.95):
    """Get Bernstein bound based error bars on mean of `x`. This error bar
    makes no distributional or central limit theorem assumption on `x`.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Data points to estimate mean. Must not be empty or contain NaNs.
    lower : float
        A priori known theoretical lower limit on unknown mean. For instance,
        for mean zero-one loss, ``lower=0``.
    upper : float
        A priori known theoretical upper limit on unknown mean. For instance,
        for mean zero-one loss, ``upper=1``.
    confidence : float
        Confidence probability (in (0, 1)) to construct confidence interval
        from t statistic.

    Returns
    -------
    EB : float
        Size of error bar on mean (>= 0). The confidence interval is
        ``[mean(x) - EB, mean(x) + EB]``. ``EB = upper - lower`` is inf when
        ``len(x) = 0``.

    Notes
    -----
    This does not do clipping of to trivial error bars, i.e., `EB` could be
    larger than ``upper - lower``. However, `clip_EB` can be called to enforce
    trivial error bar limits.

    References
    ----------
    Audibert, Jean-Yves, Remi Munos, and Csaba Szepesvari.
    "Exploration-exploitation tradeoff using variance estimates in multi-armed
    bandits." Theoretical Computer Science 410.19 (2009): 1876-1902.
    """
    assert np.ndim(x) == 1 and (not np.any(np.isnan(x)))
    assert np.ndim(lower) == 0 and np.ndim(upper) == 0
    range_ = upper - lower
    assert range_ >= 0.0  # Also catch (inf, inf) or nans
    assert np.all(lower <= x) and np.all(x <= upper)
    assert np.ndim(confidence) == 0
    assert 0.0 < confidence and confidence < 1.0

    N = x.size
    if (N <= 1) or (not np.all(np.isfinite(x))):
        return range_

    # From Thm 1 of Audibert et. al. (2009), must use MLE for std ==> ddof=0
    delta = 1.0 - confidence
    A = np.log(3.0 / delta)
    EB = np.std(x, ddof=0) * np.sqrt((2.0 * A) / N) + (3.0 * A * range_) / N
    assert np.ndim(EB) == 0 and EB >= 0.0
    return EB


def _boot_EB_and_test(x, *, f=None, confidence=0.95, n_boot=N_BOOT, return_EB=True, return_test=True, return_CI=False):
    """Internal helper function to compute both bootstrap EB and significance
    using the same random bootstrap weights, which saves computation and
    guarantees the results are coherent with each other."""
    assert np.ndim(x) >= 1 and (not np.any(np.isnan(x)))
    # confidence is checked by bu.error_bar

    N = x.shape[0]
    if (N <= 1) or (not np.all(np.isfinite(x))):
        return np.inf, 1.0, (-np.inf, np.inf)

    weight = bu.boot_weights(N, n_boot)
    if f is None:
        assert np.ndim(x) == 1
        mu = np.mean(x).item()
        mu_boot = np.mean(x * weight, axis=1)
    else:
        assert np.ndim(x) == 2
        n_sample, n_stat = x.shape
        mu = f(np.sum(x, axis=0, keepdims=True), n_sample).item()
        mu_boot = np.matmul(weight, x)
        assert mu_boot.shape == (n_boot, n_stat)
        # Perhaps more intuitively expressed as below. We can remove once we are confident in correctness here.
        assert np.allclose(mu_boot, np.sum(x[None, :, :] * weight[:, :, None], axis=1))
        mu_boot = f(mu_boot, n_sample)
    assert isinstance(mu, float)
    assert mu_boot.shape == (n_boot,)
    assert not np.any(np.isnan(mu_boot))

    pval = bu.significance(mu_boot, ref=0.0) if return_test else 1.0

    EB = np.inf
    if return_EB:
        EB = bu.error_bar(mu_boot, mu, confidence=confidence)

    # Useful in test:
    CI = -np.inf, np.inf
    if return_CI:
        CI = bu.percentile(mu_boot, confidence=confidence)
    return EB, pval, CI


def boot_test(x, n_boot=N_BOOT):
    """Perform a bootstrap-based test to test if the values in `x` are sampled
    from a distribution with a zero mean.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        array of data points to test.
    n_boot : int
        Number of bootstrap iterations to perform.

    Returns
    -------
    pval : float
        p-value (in [0,1]) from t-test on `x`.
    """
    _, pval, _ = _boot_EB_and_test(x, n_boot=n_boot, return_EB=False, return_test=True)
    assert 0.0 <= pval and pval <= 1.0
    return pval


def boot_EB(x, confidence=0.95, n_boot=N_BOOT):
    """Get bootstrap bound based error bars on mean of `x`.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Data points to estimate mean. Must not be empty or contain NaNs.
    confidence : float
        Confidence probability (in (0, 1)) to construct confidence interval
        from t statistic.
    n_boot : int
        Number of bootstrap iterations to perform.

    Returns
    -------
    EB : float
        Size of error bar on mean (>= 0). The confidence interval is
        ``[mean(x) - EB, mean(x) + EB]``. `EB` is inf when ``len(x) <= 1``.
    """
    EB, _, _ = _boot_EB_and_test(x, confidence=confidence, n_boot=n_boot, return_EB=True, return_test=False)
    assert np.ndim(EB) == 0 and EB >= 0.0
    return EB


def get_mean_and_EB(x, confidence=0.95, min_EB=0.0, lower=-np.inf, upper=np.inf, method="t"):
    """Get mean loss and estimated error bar.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Array of independent observations.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.
    min_EB : float
        Minimum size of resulting error bar regardless of the data in `x`.
    lower : float
        A priori known theoretical lower limit on unknown mean of `x`. For
        instance, for mean zero-one loss, ``lower=0``.
    upper : float
        A priori known theoretical upper limit on unknown mean of `x`. For
        instance, for mean zero-one loss, ``upper=1``.
    method : {'t', 'bernstein', 'boot'}
        Method to use for building error bar.

    Returns
    -------
    mu : float
        Estimated mean of `x`.
    EB : float
        Size of error bar on mean of `x` (``EB > 0``). The confidence interval
        is ``[mu - EB, mu + EB]``.
    """
    assert np.all(lower <= x) and np.all(x <= upper)

    if method == "t":
        EB = t_EB(x, confidence=confidence)
    elif method == "bernstein":
        EB = bernstein_EB(x, lower, upper, confidence=confidence)
    elif method == "boot":
        EB = boot_EB(x, confidence=confidence)
    else:
        assert False

    # EB subroutines already validated x for shape and nans
    mu = clip_chk(np.mean(x), lower, upper)
    EB = clip_EB(mu, EB, lower, upper, min_EB=min_EB)
    return mu, EB


def get_test(x, lower=-np.inf, upper=np.inf, method="t"):
    """Perform a statistical test to determine if the values in `x` are sampled
    from a distribution with a zero mean.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Array of independent observations.
    lower : float
        A priori known theoretical lower limit on unknown mean of `x`. For
        instance, for mean zero-one loss, ``lower=0``.
    upper : float
        A priori known theoretical upper limit on unknown mean of `x`. For
        instance, for mean zero-one loss, ``upper=1``.
    method : {'t', 'bernstein', 'boot'}
        Method to use statistical test.

    Returns
    -------
    pval : float
        p-value (in [0,1]) from statistical test on `x`.
    """
    if method == "t":
        pval = t_test(x)
    elif method == "bernstein":
        pval = bernstein_test(x, lower, upper)
    elif method == "boot":
        pval = boot_test(x)
    else:
        assert False
    return pval


def get_mean_EB_test(x, confidence=0.95, min_EB=0.0, lower=-np.inf, upper=np.inf, method="t"):
    """Get mean loss and estimated error bar. Also, perform a statistical test
    to determine if the values in `x` are sampled from a distribution with a
    zero mean.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Array of independent observations.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.
    min_EB : float
        Minimum size of resulting error bar regardless of the data in `x`.
    lower : float
        A priori known theoretical lower limit on unknown mean of `x`. For
        instance, for mean zero-one loss, ``lower=0``.
    upper : float
        A priori known theoretical upper limit on unknown mean of `x`. For
        instance, for mean zero-one loss, ``upper=1``.
    method : {'t', 'bernstein', 'boot'}
        Method to use for building error bar.

    Returns
    -------
    mu : float
        Estimated mean of `x`.
    EB : float
        Size of error bar on mean of `x` (``EB > 0``). The confidence interval
        is ``[mu - EB, mu + EB]``.
    pval : float
        p-value (in [0,1]) from statistical test on `x`.
    """
    assert np.all(lower <= x) and np.all(x <= upper)

    if method == "t":
        EB = t_EB(x, confidence=confidence)
        pval = t_test(x)
    elif method == "bernstein":
        EB = bernstein_EB(x, lower, upper, confidence=confidence)
        pval = bernstein_test(x, lower, upper)
    elif method == "boot":
        EB, pval, _ = _boot_EB_and_test(x, confidence=confidence)
    else:
        assert False

    # EB subroutines already validated x for shape and nans
    mu = clip_chk(np.mean(x), lower, upper)
    EB = clip_EB(mu, EB, lower, upper, min_EB=min_EB)
    return mu, EB, pval


def get_func_mean_EB_test(x, f, confidence=0.95, min_EB=0.0, lower=-np.inf, upper=np.inf, method="boot"):
    """Get a metric and estimated error bar. Also, perform a statistical test
    to determine if the metric is zero.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_stat)
        Array of independent observations. The mean of each column is given to `f` to compute the final metric.
    f : callable
        The function we are putting error bars on is ``f(sum(x), n_samples)``. It must have signature:
        ``(n_case,n_stat),()->(n_case)``. We use `n_case` for vectorization. For only a single estimation ``n_case=1``.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.
    min_EB : float
        Minimum size of resulting error bar regardless of the data in `x`.
    lower : float
        A priori known theoretical lower limit on metric.
    upper : float
        A priori known theoretical upper limit on metric.
    method : {'boot'}
        Method to use for building error bar.

    Returns
    -------
    mu : float
        Estimated metric.
    EB : float
        Size of error bar on metric (``EB > 0``). The confidence interval
        is ``[mu - EB, mu + EB]``.
    pval : float
        p-value (in [0,1]) from statistical test on `x`.
    """
    sample_size, _ = x.shape

    if method == "boot":
        EB, pval, _ = _boot_EB_and_test(x, f=f, confidence=confidence)
    else:
        # More methods will be added later, just bootstrap for now
        assert False

    # EB subroutines already validated x for shape and nans
    raw_mu = f(np.sum(x, axis=0, keepdims=True), sample_size).item()
    mu = clip_chk(raw_mu, lower, upper)
    EB = clip_EB(mu, EB, lower, upper, min_EB=min_EB)
    return mu, EB, pval


# ============================================================================
# Loss summary: the main purpose of this file.
# ============================================================================


def loss_summary_table(loss_table, ref_method, pairwise_CI=PAIRWISE_DEFAULT, confidence=0.95, method_EB="t", limits={}):
    """Build table with mean and error bar summaries from a loss table that
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
    method_EB : {'t', 'bernstein', 'boot'}
        Method to use for building error bar.
    limits : dict of str to (float, float)
        Dictionary mapping metric name to tuple with (lower, upper) which are
        the theoretical limits on the mean loss. For instance, zero-one loss
        should be ``(0.0, 1.0)``. If entry missing, (-inf, inf) is used.

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
    """
    assert loss_table.columns.names == (METRIC, METHOD)
    metrics, methods = loss_table.columns.levels
    assert ref_method in methods  # ==> len(methods) >= 1
    assert len(loss_table) >= 1 and len(metrics) >= 1
    # Could also test these are cartesian product if we wanted to be exhaustive

    col_names = pd.MultiIndex.from_product([metrics, STD_STATS], names=[METRIC, STAT])
    perf_tbl = pd.DataFrame(index=methods, columns=col_names, dtype=float)
    perf_tbl.index.set_names(METHOD, inplace=True)
    for metric in metrics:
        lower, upper = limits.get(metric, (-np.inf, np.inf))
        assert lower <= upper
        loss_ref = loss_table.loc[:, (metric, ref_method)].values
        assert loss_ref.ndim == 1  # Weird stuff happens if names not unique
        assert not np.any(np.isnan(loss_ref))  # Would let method cheat
        assert np.all(lower <= loss_ref) and np.all(loss_ref <= upper)
        for method in methods:
            loss = loss_table.loc[:, (metric, method)].values
            assert loss.ndim == 1  # Weird stuff happens if names not unique
            assert not np.any(np.isnan(loss))  # Would let method cheat
            assert np.all(lower <= loss) and np.all(loss <= upper)

            mu = np.mean(loss)  # This is the same in all cases

            deltas = loss - loss_ref
            range_ = upper - lower
            self_comparison = method == ref_method

            EB, pval = np.nan, np.nan
            if pairwise_CI:
                if not self_comparison:  # Otherwise leave both as nan
                    _, EB, pval = get_mean_EB_test(deltas, confidence, lower=-range_, upper=range_, method=method_EB)
            else:
                mu_, EB = get_mean_and_EB(loss, confidence=confidence, lower=lower, upper=upper, method=method_EB)
                assert mu_ == mu
                if not self_comparison:  # Otherwise pval as nan
                    pval = get_test(deltas, lower=-range_, upper=range_, method=method_EB)

            # This is two-sided, could include one-sided option too.
            perf_tbl.loc[method, metric] = (mu, EB, pval)
    return perf_tbl


def metric_summary_table(metric_table, f, *, confidence=0.95, method_EB="boot", limits={}, lower=-np.inf, upper=np.inf):
    """Build table with mean and error bar summaries from a loss table that
    contains losses on a per data point basis.

    Parameters
    ----------
    metric_table : DataFrame, shape (n_samples, n_method * n_stat)
        DataFrame with sufficient statistics for metrics of each method according to each metrics function on each data
        point.
    f : callable
        The function we are putting error bars on is ``f(sum(x), n_samples)``. It must have signature:
        ``(n_case,n_stat),()->(n_case)``. We use `n_case` for vectorization. For only a single estimation ``n_case=1``.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.
    method_EB : {'boot'}
        Method to use for building error bar.
    limits : dict of str to (float, float)
        Reserved for future use.
    lower : float
        Lower theoretical limit on the metric.
    upper : float
        Upper theoretical limit on the metric.

    Returns
    -------
    perf_tbl : DataFrame, shape (n_methods, 3)
        DataFrame summarizing the statistics for the metric on each method.
    """
    assert metric_table.columns.names == (METHOD, METRIC)
    methods, metrics = metric_table.columns.levels
    assert len(methods) >= 1
    assert len(metric_table) >= 1 and len(metrics) >= 1
    assert lower <= upper
    # Could also test these are cartesian product if we wanted to be exhaustive

    perf_tbl = pd.DataFrame(index=methods, columns=STD_STATS, dtype=float)
    perf_tbl.index.set_names(METHOD, inplace=True)
    for method in methods:
        x = metric_table[method].values
        assert x.ndim == 2
        assert not np.any(np.isnan(x))  # Would let method cheat

        mu, EB, pval = get_func_mean_EB_test(x, f, confidence=confidence, lower=lower, upper=upper, method=method_EB)

        perf_tbl.loc[method, MEAN_COL] = mu
        perf_tbl.loc[method, ERR_COL] = EB
        perf_tbl.loc[method, PVAL_COL] = pval
    return perf_tbl
