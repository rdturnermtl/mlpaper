# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, absolute_import, division
from joblib import Memory
import numpy as np
import pandas as pd
from scipy.misc import logsumexp
from benchmark_tools.benchmark_tools import loss_summary_table
from benchmark_tools.constants import (
    STAT, CURVE_STATS, STD_STATS, ERR_COL, PVAL_COL,
    METHOD, METRIC, PAIRWISE_DEFAULT)
import benchmark_tools.perf_curves as pc
from benchmark_tools.util import one_hot, normalize, interp1d, area

DEFAULT_NGRID = 100
LABEL = 'label'  # Don't put in constants since only needed for classification


def shape_and_validate(y, log_pred_prob):
    '''Validate shapes and types of predictive distribution against data and
    return the shape information.

    Parameters
    ----------
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point.
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        Array of shape ``(len(y), n_labels)``. Each row corresponds to a
        categorical distribution with *normalized* probabilities in log scale.
        Therefore, the number of columns must be at least 1.

    Returns
    -------
    n_samples : int
        Number of data points (length of `y`)
    n_labels : int
        The number of possible labels in `y`. Inferred from size of
        `log_pred_prob` and *not* from `y`.

    Notes
    -----
    This does *not* check normalization.
    '''
    n_samples, n_labels = log_pred_prob.shape
    assert(n_samples >= 1)  # Otherwise min and max confused
    assert(n_labels >= 1)  # Otherwise makes no sense
    assert(y.shape == (n_samples,) and y.dtype.kind in ('b', 'i'))
    assert(0 <= y.min() and y.max() < n_labels)
    return n_samples, n_labels

# ============================================================================
# Loss functions
# ============================================================================


def hard_loss_decision(log_pred_prob, loss_mat):
    '''Make Bayes' optimal action according to predictive probability
    distribution and loss matrix.

    Parameters
    ----------
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        Array of shape ``(len(y), n_labels)``. Each row corresponds to a
        categorical distribution with *normalized* probabilities in log scale.
        Therefore, the number of columns must be at least 1.
    loss_mat : ndarray, shape (n_labels, n_actions)
        Loss matrix to use for making decisions of size
        ``(n_labels, n_actions)``. The loss of taking action a when the true
        outcome (label) is y is found in ``loss_mat[y, a]``.

    Returns
    -------
    action : ndarray of type int, shape (n_samples,)
        Array of resulting Bayes' optimal action for each data point.
    '''
    pred_prob = np.exp(log_pred_prob)
    E_loss = np.dot(pred_prob, loss_mat)
    action = np.argmin(E_loss, axis=1)
    return action


def hard_loss(y, log_pred_prob, loss_mat=None):
    '''Loss function for making classification decisions from a loss matrix.

    This function both computes the optimal action under the predictive
    distribution and the loss matrix, and then scores that decision using the
    loss matrix.

    Parameters
    ----------
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point.
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        Array of shape ``(len(y), n_labels)``. Each row corresponds to a
        categorical distribution with *normalized* probabilities in log scale.
        Therefore, the number of columns must be at least 1.
    loss_mat : None or ndarray of shape (n_labels, n_actions)
        Loss matrix to use for making decisions of size
        ``(n_labels, n_actions)``. The loss of taking action a when the true
        outcome (label) is y is found in ``loss_mat[y, a]``. If None,
        1 - identity matrix is used to obtain the 0-1 loss function.

    Returns
    -------
    loss : ndarray, shape (n_samples,)
        Array of the resulting loss for the predictions on each point in `y`.
    '''
    n_samples, n_labels = shape_and_validate(y, log_pred_prob)
    loss_mat = (1.0 - np.eye(n_labels)) if loss_mat is None else loss_mat
    assert(np.ndim(loss_mat) == 2 and loss_mat.shape[0] == n_labels)
    assert(loss_mat.shape[1] >= 1)  # Must be least one action

    action = hard_loss_decision(log_pred_prob, loss_mat)

    assert(action.shape == y.shape and action.dtype.kind == 'i')
    loss = loss_mat[y.astype(int), action]
    assert(loss.shape == (n_samples,))
    return loss


def log_loss(y, log_pred_prob):
    '''Compute log loss (e.g, negative log likelihood or cross-entropy).

    Parameters
    ----------
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point.
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        Array of shape ``(len(y), n_labels)``. Each row corresponds to a
        categorical distribution with *normalized* probabilities in log scale.
        Therefore, the number of columns must be at least 1.

    Returns
    -------
    loss : ndarray, shape (n_samples,)
        Array of the log loss for the predictions on each data point in `y`.
    '''
    n_samples, n_labels = shape_and_validate(y, log_pred_prob)
    nll = -log_pred_prob[np.arange(n_samples), y.astype(int)]
    return nll


def brier_loss(y, log_pred_prob, rescale=True):
    '''Compute (rescaled) Brier loss.

    Parameters
    ----------
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point.
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        Array of shape ``(len(y), n_labels)``. Each row corresponds to a
        categorical distribution with *normalized* probabilities in log scale.
        Therefore, the number of columns must be at least 1.
    rescale : bool
        If True, linearly rescales lost so perfect (P=1) predictions give 0.0
        loss and a uniform prediction gives loss of 1.0. False gives the
        standard Brier loss.

    Returns
    -------
    loss : ndarray, shape (n_samples,)
        Array of the Brier loss for the predictions on each data point in `y`.
    '''
    n_samples, n_labels = shape_and_validate(y, log_pred_prob)

    y_bin = one_hot(y.astype(int), n_labels)
    loss = np.sum((np.exp(log_pred_prob) - y_bin) ** 2, axis=1)

    if rescale and n_labels > 1:
        # Linearly rescale so perfect is 0.0 and uniform gives 1.0
        loss = np.true_divide(n_labels, n_labels - 1) * loss
    return loss


def spherical_loss(y, log_pred_prob, rescale=True):
    '''Compute (rescaled) spherical loss.

    Parameters
    ----------
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point.
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        Array of shape ``(len(y), n_labels)``. Each row corresponds to a
        categorical distribution with *normalized* probabilities in log scale.
        Therefore, the number of columns must be at least 1.
    rescale : bool
        If True, linearly rescales lost so perfect (P=1) predictions give 0.0
        loss and a uniform prediction gives loss of 1.0. False gives the
        standard spherical loss, which is the negative spherical *score*.

    Returns
    -------
    loss : ndarray, shape (n_samples,)
        Array of the spherical loss for the predictions on each point in `y`.
    '''
    N, n_labels = shape_and_validate(y, log_pred_prob)

    log_normalizer = 0.5 * logsumexp(2.0 * log_pred_prob, axis=1)
    # Need to do negative of spherical score to make a loss function
    loss = -np.exp(log_pred_prob[np.arange(N), y.astype(int)] - log_normalizer)

    if rescale and n_labels > 1:
        # Linearly rescale so perfect is 0.0 and uniform gives 1.0
        c = 1.0 - 1.0 / np.sqrt(n_labels)
        loss = (1.0 + loss) / c
    return loss

# ============================================================================
# Loss summary: the main purpose of this file.
# ============================================================================


def loss_table(log_pred_prob_table, y, metrics_dict, assume_normalized=False):
    '''Compute loss table from table of probalistic predictions.

    Parameters
    ----------
    log_pred_prob_table : DataFrame, shape (n_samples, n_methods * n_labels)
        DataFrame with predictive distributions. Each row is a data point.
        The columns should be hierarchical index that is the cartesian product
        of methods x labels. For exampe, ``log_pred_prob_table.loc[5, 'foo']``
        is the categorical distribution (in log scale) prediction that method
        foo places on ``y[5]``.
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point. Must be of same length as
        DataFrame `log_pred_prob_table`.
    metrics_dict : dict of str to callable
        Dictionary mapping loss function name to function that computes loss,
        e.g., `log_loss`, `brier_loss`, ...
    assume_normalized : bool
        If False, renormalize the predictive distributions to ensure there is
        no cheating. If True, skips this step for speed.

    Returns
    -------
    loss_tbl : DataFrame, shape (n_samples, n_metrics * n_methods)
        DataFrame with loss of each method according to each loss function on
        each data point. The rows are the data points in `y` (that is the index
        matches `log_pred_prob_table`). The columns are a hierarchical index
        that is the cartesian product of loss x method. That is, the loss of
        method foo's prediction of ``y[5]`` according to loss function bar is
        stored in ``loss_tbl.loc[5, ('bar', 'foo')]``.
    '''
    methods, labels = log_pred_prob_table.columns.levels
    n_samples, n_labels = len(log_pred_prob_table), len(labels)
    assert(y.shape == (n_samples,))
    assert(n_samples >= 1 and n_labels >= 1 and len(methods) >= 1)

    col_names = pd.MultiIndex.from_product([metrics_dict.keys(), methods],
                                           names=[METRIC, METHOD])
    loss_tbl = pd.DataFrame(index=log_pred_prob_table.index,
                            columns=col_names, dtype=float)
    for method in methods:
        # Make sure the columns are in right order and we aren't mixing things
        assert(list(log_pred_prob_table[method].columns) ==
               list(range(n_labels)))

        log_pred_prob = log_pred_prob_table[method].values
        assert(log_pred_prob.shape == (n_samples, n_labels))
        assert(not np.any(np.isnan(log_pred_prob)))  # Would let method cheat

        if not assume_normalized:
            log_pred_prob = normalize(log_pred_prob)

        for metric, metric_f in metrics_dict.items():
            loss_tbl.loc[:, (metric, method)] = metric_f(y, log_pred_prob)
    return loss_tbl

# ============================================================================
# Use and summarize performance curves
# ============================================================================


def check_curve(result, singleton=False, x_grid=None):
    '''Check performance curve output matches expected format and return the
    curve after validation.

    Parameters
    ----------
    curve : result of curve function, e.g., classification.roc_curve
        Curves defined by a ROC or other curve estimation.
    singleton : bool
        If True, check that the 2d arrays have only a single columns, i.e.,
        only a single curve.
    x_grid : None or ndarray of shape (n_grid,)
        If provided, check that all the curves are defined over a wider range
        than the x_grid. So, when the functions are interpolated onto the range
        of x_grid no extrapolation is needed.

    Returns
    -------
    curve : tuple of (ndarray, ndarray, str)
        Returns same object passed in after some input checks. Each of the
        ndarrays have shape (n_thresholds, n_boot).
    '''
    curve, _ = result  # Skipping tholds (2nd arg) since not used here
    x_curve, y_curve, kind = curve

    # Check shape
    assert(x_curve.ndim == 2 and y_curve.ndim == 2)
    assert(x_curve.shape == y_curve.shape)
    assert((not singleton) or x_curve.shape[1] == 1)
    assert(x_curve.shape[0] >= 2)  # Otherwise not curve

    # Check values
    assert(np.all(np.isfinite(x_curve)))
    assert(np.all(y_curve < np.inf))  # PRG can be -inf, but all curves < inf
    assert(np.all(np.diff(x_curve, axis=0) >= 0.0))  # also check is sorted
    if x_grid is not None:  # Make sure we won't need to extrapolate for grid
        assert(np.all(x_curve[0, :] <= x_grid[0]))
        assert(np.all(x_grid[-1] <= x_curve[-1, :]))
    return curve


def confidence_to_percentiles(confidence):
    assert(np.ndim(confidence) == 0 and 0.0 < confidence and confidence < 1.0)

    alpha = 0.5 * (1.0 - confidence)
    q_levels = (100.0 * alpha, 100.0 * (1.0 - alpha))
    return q_levels


def boot_samples_to_CI(boot_samples, confidence):
    assert(boot_samples.ndim >= 1)

    q_levels = confidence_to_percentiles(confidence)
    LB, UB = np.percentile(boot_samples, q_levels, axis=-1)
    assert(LB.shape == boot_samples.shape[:-1])
    assert(LB.shape == UB.shape)
    return LB, UB


def boot_samples_to_EB(boot_samples, ref=0.0, confidence=0.95):
    assert(boot_samples.ndim == 1)
    assert(np.ndim(ref) == 0 or ref.shape == boot_samples.shape)

    delta = boot_samples - ref
    mu_delta = np.mean(delta)
    LB, UB = boot_samples_to_CI(delta, confidence)
    EB = np.fmax(UB - mu_delta, mu_delta - LB)
    assert(EB >= 0.0 or np.isnan(EB))
    # NaN EB only ever occurs when ref is infinite and so are some samples
    assert(np.all(np.isfinite(ref)) <= ~np.isnan(EB))
    return EB


def boot_samples_to_pval(boot_samples, ref):
    assert(boot_samples.ndim == 1)
    assert(np.ndim(ref) == 0 or ref.shape == boot_samples.shape)

    pval = 2.0 * np.minimum(np.mean(boot_samples <= ref),
                            np.mean(ref <= boot_samples))
    pval = np.minimum(1.0, pval)  # Only needed when some auc == ref exactly
    return pval


def interp1d_(x_grid, x_boot, y_boot, kind):
    # TODO just use np vectorize
    n_boot = x_boot.shape[1]

    y_grid_boot = np.zeros((x_grid.size, n_boot))
    for nn in range(n_boot):
        y_grid_boot[:, nn] = \
            interp1d(x_grid, x_boot[:, nn], y_boot[:, nn], kind)
    assert(y_grid_boot.shape == (x_grid.size, n_boot))
    return y_grid_boot


def curve_boot(y, log_pred_prob, ref, curve_f=pc.roc_curve, x_grid=None,
               n_boot=1000, pairwise_CI=PAIRWISE_DEFAULT, confidence=0.95):
    '''Perform boot strap analysis of performance curve, e.g., ROC or prec-rec.
    For binary classification only.

    Parameters
    ----------
    y : ndarray of type int or bool, shape (n_samples,)
        Array containing true labels, must be `bool` or {0,1}.
    log_pred_prob :  ndarray, shape (n_samples, 2)
        Array of shape ``(len(y), 2)``. Each row corresponds to a categorical
        distribution with *normalized* probabilities in log scale. However,
        many curves (e.g., ROC) are invariant to monotonic transformation and
        hence linear scale could also be used.
    ref : float or ndarray of shape (n_samples, 2)
        If `ref` is an rray of shape ``(len(y), 2)``: Same as `log_pred_prob`
        except for the reference (baseline) method if a paired statistical test
        is desired on the area under the curve. If `ref` is a scalar float:
        `curve_boot` tests the statistical significance that the area under the
        curve differs from `ref` in a non-paired test. For ROC analysis, `ref`
        is typically 0.5.
    curve_f : callable
        Function to compute the performance curve. Standard choices are:
        `perf_curves.roc_curve` or `perf_curves.recall_precision_curve`.
    x_grid : None or ndarray of shape (n_grid,)
        Grid of points to evaluate curve in results. If `None`, defaults to
        linear grid on [0,1].
    n_boot : int
        Number of bootstrap iterations to perform.
    pairwise_CI : bool
        If True, compute error bars on ``summary - summary_ref`` instead of
        just the summary. This typically results in smaller error bars.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.

    Returns
    -------
    summary : tuple of floats, shape (3,)
        Tuple containing (mu, EB, pval), where mu is the best estimate on the
        summary statistic of the curve, EB is the error bar, and pval is the
        p-value from the two-sided boot strap significance test that its value
        is the same as the reference summary value (from either
        `log_pred_prob_ref` or `default_summary_ref`).
    curve : DataFrame, shape (n_grid, 4)
        DataFrame containing four columns: `x_grid`, the curve value, the lower
        end of confidence envelope, and the upper end of the confidence
        envelope.
    '''
    N, n_labels = shape_and_validate(y, log_pred_prob)
    assert(n_labels == 2)
    assert(np.ndim(ref) == 0 or ref.shape == log_pred_prob.shape)
    assert(not np.any(np.isnan(ref)))
    assert(n_boot >= 1)
    assert(not np.any(np.isnan(log_pred_prob)))  # Would let method cheat

    # Setup constants
    epsilon = 1e-10  # Min bootstrap weight since 0 weight can cause problems
    pos_label = 1  # Label=1 of [0,1] is considered a positive case
    x_grid = np.linspace(0.0, 1.0, DEFAULT_NGRID) if x_grid is None else x_grid
    assert(np.ndim(x_grid) == 1)

    # Put everything into a vector of right type for binary classification
    y = y.astype(bool)
    log_pred_prob = log_pred_prob[:, pos_label]

    # Get estimator on original data (before boot strap)
    curve = check_curve(curve_f(y, log_pred_prob), singleton=True)
    auc, = area(*curve)
    assert(auc.ndim == 0)
    y_grid = np.squeeze(interp1d_(x_grid, *curve), axis=1)  # Use fixed grid
    assert(x_grid.shape == y_grid.shape)

    # Setup boot strap weights
    p_BS = np.ones(N) / N
    weight = np.maximum(epsilon, np.random.multinomial(N, p_BS, size=n_boot).T)

    # Get boot strapped scores
    curve_boot_ = check_curve(curve_f(y, log_pred_prob, weight))
    auc_boot = area(*curve_boot_)
    assert(auc_boot.shape == (n_boot,))
    y_grid_boot = interp1d_(x_grid, *curve_boot_)

    # Repeat area boot strap with reference predictor
    if np.ndim(ref) == 2:  # Note dim must be 0 or 2
        ref = area(*check_curve(curve_f(y, ref[:, pos_label], weight)))
        assert(ref.shape == (n_boot,))

    # Pack up standard numeric summary triple
    EB = boot_samples_to_EB(auc_boot, ref, confidence) if pairwise_CI \
        else boot_samples_to_EB(auc_boot, confidence=confidence)
    pval = boot_samples_to_pval(auc_boot, ref)
    summary = (auc, EB, pval)

    # Pack up data frame with graphical summaries (performance curves)
    y_LB, y_UB = boot_samples_to_CI(y_grid_boot, confidence)
    curve = pd.DataFrame(data=np.stack((x_grid, y_grid, y_LB, y_UB), axis=1),
                         index=range(x_grid.size), columns=CURVE_STATS,
                         dtype=float)
    return summary, curve


def curve_summary_table(log_pred_prob_table, y,
                        curve_dict, ref_method, x_grid=None,
                        n_boot=1000, pairwise_CI=PAIRWISE_DEFAULT,
                        confidence=0.95):
    '''Build table with mean and error bars of curve summaries from a table of
    probalistic predictions.

    Parameters
    ----------
    log_pred_prob_table : DataFrame, shape (n_samples, n_methods * n_labels)
        DataFrame with predictive distributions. Each row is a data point.
        The columns should be hierarchical index that is the cartesian product
        of methods x labels. For exampe, ``log_pred_prob_table.loc[5, 'foo']``
        is the categorical distribution (in log scale) prediction that method
        foo places on ``y[5]``.
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point. Must be of same length as
        DataFrame `log_pred_prob_table`.
    curve_dict : dict of str to callable
        Dictionary mapping curve name to performance curve. Standard choices:
        `perf_curves.roc_curve` or `perf_curves.recall_precision_curve`.
    ref_method : str
        Name of method that is used as reference point in paired statistical
        tests. This is usually some some of baseline method. `ref_method` must
        be found in the 1st level of the columns of `log_pred_prob_table`.
    x_grid : None or ndarray of shape (n_grid,)
        Grid of points to evaluate curve in results. If `None`, defaults to
        linear grid on [0,1].
    n_boot : int
        Number of bootstrap iterations to perform.
    pairwise_CI : bool
        If True, compute error bars on ``summary - summary_ref`` instead of
        just the summary. This typically results in smaller error bars.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.

    Returns
    -------
    curve_tbl : DataFrame, shape (n_methods, n_metrics * 3)
        DataFrame with curve summary of each method according to each curve.
        The rows are the methods. The columns are a hierarchical index that is
        the cartesian product of curve x (summary, error bar, p-value).
        That is, ``curve_tbl.loc['foo', 'bar']`` is a pandas series with
        (summary of bar curve on foo, corresponding error bar, statistical sig)
        The statistical significance is a p-value from a two-sided hypothesis
        test on the hypothesis H0 that foo has the same curve summary as the
        reference method `ref_method`.
    curve_dump : dict of (str, str) to DataFrame of shape (n_grid, 4)
        Each key is a pair of (method name, curve name) with the value being
        a pandas dataframe with the performance curve, which has four columns:
        `x_grid`, the curve value, the lower end of confidence envelope,
        and the upper end of the confidence envelope.
    '''
    methods, labels = log_pred_prob_table.columns.levels
    N, n_labels = len(log_pred_prob_table), len(labels)
    assert(y.shape == (N,))
    assert(ref_method in methods)  # ==> len(methods) >= 1
    assert(N >= 1 and n_labels >= 1 and len(curve_dict) >= 1)

    assert(list(log_pred_prob_table[ref_method].columns) ==
           list(range(n_labels)))
    log_pred_prob_ref = log_pred_prob_table[ref_method].values
    assert(log_pred_prob_ref.shape == (N, n_labels))
    # Note: Most curve methods are rank based and so normalization is not
    # needed to prevent cheating. However, if we expect non-normalized methods
    # they should be normalized before to keep consistency with loss metrics.

    col_names = pd.MultiIndex.from_product([curve_dict.keys(), STD_STATS],
                                           names=[METRIC, STAT])
    curve_tbl = pd.DataFrame(index=methods, columns=col_names, dtype=float)
    curve_tbl.index.name = METHOD

    curve_dump = {}
    for method in methods:
        assert(list(log_pred_prob_table[method].columns) ==
               list(range(n_labels)))
        log_pred_prob = log_pred_prob_table[method].values
        assert(log_pred_prob.shape == (N, n_labels))

        for curve_name, curve_f in curve_dict.items():
            R = curve_boot(y, log_pred_prob, ref=log_pred_prob_ref,
                           curve_f=curve_f, x_grid=x_grid, n_boot=n_boot,
                           pairwise_CI=pairwise_CI, confidence=confidence)
            curve_summary, curr_curve = R
            curve_tbl.loc[method, curve_name] = curve_summary
            if pairwise_CI and method == ref_method:
                curve_tbl.loc[method, (curve_name, ERR_COL)] = np.nan
            if method == ref_method:  # NaN probably makes more sense than 1
                curve_tbl.loc[method, (curve_name, PVAL_COL)] = np.nan
            curve_dump[(method, curve_name)] = curr_curve
    return curve_tbl, curve_dump


def summary_table(log_pred_prob_table, y,
                  loss_dict, curve_dict, ref_method, x_grid=None,
                  n_boot=1000, pairwise_CI=PAIRWISE_DEFAULT, confidence=0.95):
    '''Build table with mean and error bars of both loss and curve summaries
    from a table of probalistic predictions.

    Parameters
    ----------
    log_pred_prob_table : DataFrame, shape (n_samples, n_methods * n_labels)
        DataFrame with predictive distributions. Each row is a data point.
        The columns should be hierarchical index that is the cartesian product
        of methods x labels. For exampe, ``log_pred_prob_table.loc[5, 'foo']``
        is the categorical distribution (in log scale) prediction that method
        foo places on ``y[5]``.
    y : ndarray of type int or bool, shape (n_samples,)
        True labels for each classication data point. Must be of same length as
        DataFrame `log_pred_prob_table`.
    loss_dict : dict of str to callable
        Dictionary mapping loss function name to function that computes loss,
        e.g., `log_loss`, `brier_loss`, ...
    curve_dict : dict of str to callable
        Dictionary mapping curve name to performance curve. Standard choices:
        `perf_curves.roc_curve` or `perf_curves.recall_precision_curve`.
    ref_method : str
        Name of method that is used as reference point in paired statistical
        tests. This is usually some some of baseline method. `ref_method` must
        be found in the 1st level of the columns of `log_pred_prob_table`.
    x_grid : None or ndarray of shape (n_grid,)
        Grid of points to evaluate curve in results. If `None`, defaults to
        linear grid on [0,1].
    n_boot : int
        Number of bootstrap iterations to perform for performance curves.
    pairwise_CI : bool
        If True, compute error bars on ``summary - summary_ref`` instead of
        just the summary. This typically results in smaller error bars.
    confidence : float
        Confidence probability (in (0, 1)) to construct error bar.

    Returns
    -------
    full_tbl : DataFrame, shape (n_methods, (n_loss + n_curve) * 3)
        DataFrame with curve/loss summary of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (summary, error bar, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo, corresponding error bar, statistical sig)
        The statistical significance is a p-value from a two-sided hypothesis
        test on the hypothesis H0 that foo has the same metric as the reference
        method `ref_method`.
    curve_dump : dict of (str, str) to DataFrame of shape (n_grid, 4)
        Each key is a pair of (method name, curve name) with the value being
        a pandas dataframe with the performance curve, which has four columns:
        `x_grid`, the curve value, the lower end of confidence envelope,
        and the upper end of the confidence envelope. Only metrics from
        `curve_dict` and *not* from `loss_dict` are found here.
    '''
    # Do the curve metrics
    curve_summary, dump_tbl = \
        curve_summary_table(log_pred_prob_table, y, curve_dict, ref_method,
                            x_grid=x_grid,
                            n_boot=n_boot, pairwise_CI=pairwise_CI,
                            confidence=confidence)

    # Do loss based metrics
    loss_tbl = loss_table(log_pred_prob_table, y, loss_dict)
    loss_summary = loss_summary_table(loss_tbl, ref_method,
                                      pairwise_CI=pairwise_CI,
                                      confidence=confidence)

    # Return the combo
    full_tbl = pd.concat((loss_summary, curve_summary), axis=1)
    return full_tbl, dump_tbl

# ============================================================================
# Variables and functions to make getting results from sklearn objects easy
# ============================================================================

# Pre-build some standard metric dicts for the user
STD_CLASS_LOSS = {'NLL': log_loss, 'Brier': brier_loss,
                  'sphere': spherical_loss, 'zero_one': hard_loss}

STD_BINARY_CURVES = {'AUC': pc.roc_curve, 'AP': pc.recall_precision_curve,
                     'AUPRG': pc.prg_curve}


class JustNoise:
    '''Class version of iid predictor compatible with sklearn interface. Same
    as ``sklearn.dummy.DummyClassifier(strategy='prior').``'''

    def __init__(self, n_labels=2):
        self.pred = np.nan + np.zeros(n_labels)

    def fit(self, X_train, y_train):
        n_labels = len(self.pred)
        self.pred = np.log(np.mean(one_hot(y_train, n_labels), axis=0))
        assert(self.pred.shape == (n_labels,))

    def predict_log_proba(self, X_test):
        n_samples = X_test.shape[0]
        pred_log_prob = np.repeat([self.pred], n_samples, axis=0)
        return pred_log_prob


def get_pred_log_prob(X_train, y_train, X_test, n_labels, methods,
                      min_log_prob=-np.inf, verbose=False, checkpointdir=None):
    '''Get the predictive probability tables for each test point on a
    collection of classification methods.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Training set 2d feature array for classifiers. Each row is an
        indepedent data point and each column is a feature.
    y_train : ndarray of type int or bool, shape (n_train,)
        Training set 1d array of truth labels for classifiers. Must be of same
        length as `X_train`. Values must be in range [0, `n_labels`) or `bool`.
    X_test : ndarray, shape (n_test, n_features)
        Test set 2d feature array for classifiers. Each row is an indepedent
        data point and each column is a feature.
    n_labels : int
        Number of labels, must be >= 1. This is not infered from `y` because
        some labels may not be found in small data chunks.
    methods : dict of str to sklearn estimator
        Dictionary mapping method name (`str`) to object that performs training
        and test. Object must follow the interface of sklearn estimators, that
        is it has a ``fit()`` method and either a ``predict_log_proba()`` or
        ``predict_proba()`` method.
    min_log_prob : float
        Minimum value to floor the predictive log probabilities (while still
        normalizing). Must be < 0. Useful to prevent inf log loss penalties.
    verbose : bool
        If True, display which method being trained.
    checkpointdir : str (directory)
        If provided, stores checkpoint results using joblib for the train/test
        in case process interrupted. If None, no checkpointing is done.

    Returns
    -------
    log_pred_prob_table : DataFrame, shape (n_samples, n_methods * n_labels)
        DataFrame with predictive distributions. Each row is a data point.
        The columns should be hierarchical index that is the cartesian product
        of methods x labels. For exampe, ``log_pred_prob_table.loc[5, 'foo']``
        is the categorical distribution (in log scale) prediction that method
        foo places on ``y[5]``.

    Notes
    -----
    If a train/test operation is loaded from a checkpoint file, the estimator
    object in methods will not be in a fit state.
    '''
    n_test = X_test.shape[0]
    assert(n_test > 0)
    assert(X_train.ndim == 2)
    assert(y_train.shape == (X_train.shape[0],))
    assert(y_train.dtype.kind in ('b', 'i'))
    assert(0 <= y_train.min() and y_train.max() < n_labels)
    assert(X_test.ndim == 2 and X_test.shape[1] == X_train.shape[1])
    assert(X_train.dtype == X_test.dtype)  # Would be weird otherwise
    assert(min_log_prob < 0.0)  # Ensure is a log-prob

    memory = Memory(cachedir=checkpointdir, verbose=0)

    @memory.cache
    def train_predict(method_obj, X_train, y_train, X_test):
        method_obj.fit(X_train, y_train)
        try:
            pred_log_prob = method_obj.predict_log_proba(X_test)
        except:  # If there is no log proba available
            pred_log_prob = np.log(method_obj.predict_proba(X_test))
        return pred_log_prob

    col_names = pd.MultiIndex.from_product([methods.keys(), range(n_labels)],
                                           names=[METHOD, LABEL])
    log_pred_prob_table = pd.DataFrame(index=range(n_test), columns=col_names,
                                       dtype=float)
    for method_name, method_obj in methods.items():
        if verbose:
            print('Running fit/predict for {}'.format(method_name))
        pred_log_prob = train_predict(method_obj, X_train, y_train, X_test)
        assert(pred_log_prob.shape == (n_test, n_labels))

        pred_log_prob = normalize(np.maximum(min_log_prob, pred_log_prob))
        log_pred_prob_table.loc[:, method_name] = pred_log_prob
    return log_pred_prob_table


def just_benchmark(X_train, y_train, X_test, y_test, n_labels,
                   methods, loss_dict, curve_dict, ref_method,
                   min_pred_log_prob=-np.inf, pairwise_CI=PAIRWISE_DEFAULT):
    '''Simplest one-call interface to this package. Just pass it data and
    method objects and a performance summary DataFrame is returned.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Training set 2d feature array for classifiers. Each row is an
        indepedent data point and each column is a feature.
    y_train : ndarray of type int or bool, shape (n_train,)
        Training set 1d array of truth labels for classifiers. Must be of same
        length as `X_train`. Values must be in range [0, `n_labels`) or `bool`.
    X_test : ndarray, shape (n_test, n_features)
        Test set 2d feature array for classifiers. Each row is an indepedent
        data point and each column is a feature.
    y_test : ndarray of type int or bool, shape (n_test,)
        Test set 1d array of truth labels for classifiers. Must be of same
        length as `X_test`. Values must be in range [0, `n_labels`) or `bool`.
    n_labels : int
        Number of labels, must be >= 1. This is not infered from `y` because
        some labels may not be found in small data chunks.
    methods : dict of str to sklearn estimator
        Dictionary mapping method name (`str`) to object that performs training
        and test. Object must follow the interface of sklearn estimators, that
        is it has a ``fit()`` method and either a ``predict_log_proba()`` or
        ``predict_proba()`` method.
    loss_dict : dict of str to callable
        Dictionary mapping loss function name to function that computes loss,
        e.g., `log_loss`, `brier_loss`, ...
    curve_dict : dict of str to callable
        Dictionary mapping curve name to performance curve. Standard choices:
        `perf_curves.roc_curve` or `perf_curves.recall_precision_curve`.
    ref_method : str
        Name of method that is used as reference point in paired statistical
        tests. This is usually some some of baseline method. `ref_method` must
        be found in `methods` dictionary.
    min_log_prob : float
        Minimum value to floor the predictive log probabilities (while still
        normalizing). Must be < 0. Useful to prevent inf log loss penalties.
    pairwise_CI : bool
        If True, compute error bars on the mean of ``loss - loss_ref`` instead
        of just the mean of `loss`. This typically gives smaller error bars.

    Returns
    -------
    full_tbl : DataFrame, shape (n_methods, (n_loss + n_curve) * 3)
        DataFrame with curve/loss summary of each method according to each
        curve or loss function. The rows are the methods. The columns are a
        hierarchical index that is the cartesian product of
        metric x (summary, error bar, p-value), where metric can be a loss or
        a curve summary: ``full_tbl.loc['foo', 'bar']`` is a pandas series
        with (metric bar on foo, corresponding error bar, statistical sig)
        The statistical significance is a p-value from a two-sided hypothesis
        test on the hypothesis H0 that foo has the same metric as the reference
        method `ref_method`.
    curve_dump : dict of (str, str) to DataFrame of shape (n_grid, 4)
        Each key is a pair of (method name, curve name) with the value being
        a pandas dataframe with the performance curve, which has four columns:
        `x_grid`, the curve value, the lower end of confidence envelope,
        and the upper end of the confidence envelope. Only metrics from
        `curve_dict` and *not* from `loss_dict` are found here.
    '''
    assert(y_train.dtype == y_test.dtype)  # Would be weird otherwise
    pred_tbl = get_pred_log_prob(X_train, y_train, X_test, n_labels,
                                 methods, min_log_prob=min_pred_log_prob)
    full_tbl, dump = summary_table(pred_tbl, y_test, loss_dict, curve_dict,
                                   ref_method, pairwise_CI=pairwise_CI)
    return full_tbl, dump
