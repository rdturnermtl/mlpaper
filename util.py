# Ryan Turner (turnerry@iro.umontreal.ca)
import warnings
import numpy as np
import scipy.interpolate as si
from scipy.misc import logsumexp

STRICT_SPACING = False


def one_hot(y, n_labels):
    '''Same functionality `sklearn.preprocessing.OneHotEncoder` but avoids
    extra dependency.

    Parameters
    ----------
    y : ndarray of type int, shape (n_samples,)
        Integers in range ``[0, n_labels)`` to be one-hot encoded.
    n_labels : int
        Number of labels, must be >= 1. This is not infered from `y` because
        some labels may not be found in small data chunks.

    Returns
    -------
    y_bin : ndarray of type bool, shape (n_samples, n_labels)
        One hot encoding of `y`, with size ``(len(y), n_labels)``
    '''
    N, = y.shape
    assert(n_labels >= 1)
    assert(y.dtype.kind == 'i')  # bool would confuse np indexing
    assert(np.all(0 <= y) and np.all(y < n_labels))

    y_bin = np.zeros((N, n_labels), dtype=bool)
    y_bin[xrange(N), y] = True
    return y_bin


def normalize(log_pred_prob):
    '''Normalize log probability distributions for classification.

    Parameters
    ----------
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        Each row corresponds to a categorical distribution with unnormalized
        probabilities in log scale. Therefore, the number of columns must be at
        least 1.

    Returns
    -------
    log_pred_prob : ndarray, shape (n_samples, n_labels)
        A row-wise normalized (``exp(log_pred_prob)`` sums to 1 on each row)
        version of the input.
    '''
    assert(log_pred_prob.ndim == 2)
    assert(log_pred_prob.shape[1] >= 1)  # Otherwise, can't make it sum to 1

    normalizer = logsumexp(log_pred_prob, axis=1, keepdims=True)
    log_pred_prob = log_pred_prob - normalizer
    return log_pred_prob


def epsilon_noise(x, default_epsilon=1e-10, max_epsilon=1.0):
    '''Add a small amount of noise to a vector such that the output vector has
    all unique values. The ordering of the resutiling vector remains the
    same: ``argsort(output) = argsort(input)`` if input values are unique.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Input vector to be noise corrupted. Must have all finite values.
    default_epsilon : float
        Default noise to add for singleton lists, musts be > 0.0.
    max_epsilon : float
        Maximum amount of noise corruption regardless of scale found in `x`.

    Returns
    -------
    x : ndarray, shape (n_samples,)
        Noise correupted version of input. All values are unique with
        probability 1. The ordering is the same as the input if the inputs
        values are all unique.
    '''
    assert(x.ndim == 1)
    assert(np.all(np.isfinite(x)))

    u_x = np.unique(x)
    delta = default_epsilon if len(u_x) <= 1 else np.min(np.diff(u_x))
    delta = np.minimum(max_epsilon, delta)
    assert(0.0 < delta and delta <= max_epsilon)

    x = x + delta * (np.random.rand(len(x)) - 0.5)
    return x

# ============================================================================
# Interpolation utils
# Everything in here will be obsolete once various enhancements get merged
# into scipy for scipy.interpolate.interp1d.
# ============================================================================


def unique_take_last(xp, yp=None):
    """Make pairs of `xp` and `yp` vectors into proper step function. That is,
    remove NaN `xp` values and multiple steps at same location.

    Parameters
    ----------
    xp : ndarray, shape (n_samples,)
        The sample points corresponding to the y values. Must be sorted.
    yp : ndarray, shape (n_samples,)
        Values in y-axis for step function.

    Returns
    -------
    xp : ndarray, shape (m_samples,)
        Input `xp` after removing extra points. m_samples <= n_samples.
    yp : ndarray, shape (m_samples,)
        Input `yp` after removing extra points. m_samples <= n_samples.

    Notes
    -----
    Keeps last value in list when multiple steps happen at the same x-value.
    """
    # TODO update doc string
    assert(xp.ndim == 1)
    assert(yp is None or xp.shape == yp.shape)

    # TODO consider just changing to error if never needed
    idx = ~np.isnan(xp)
    if ~np.all(idx):
        warnings.warn('NaN value encountered in unique_take_last.')
        xp = xp[idx]
        yp = None if yp is None else yp[idx]

    deltas = np.diff(xp)
    assert(not np.any(deltas < 0))  # Should be sorted exactly

    idx = [] if xp.size == 0 else np.concatenate((deltas > 0, [True]))
    xp = xp[idx]
    yp = None if yp is None else yp[idx]
    return xp, yp


def cummax_strict(x, copy=True):
    '''Minimally increase array elements to make the array strictly increasing.
    '''
    # TODO doc string, tests
    assert(x.ndim == 1)

    x = np.copy(x) if copy else x
    for ii in xrange(1, len(x)):
        x[ii] = np.maximum(np.nextafter(x[ii - 1], np.inf), x[ii])
    assert(np.all(np.diff(x) > 0))
    return x  # return even though inplace


def eval_step_func(x_grid, xp, yp, ival=None,
                   assume_sorted=False, skip_unique_chk=False):
    '''Evaluate a stepwise function. Based on the ECDF class in statsmodels.
    The function is assumed to cadlag (like a CDF function).

    This is a non-OOP equivalent to class:
    `statsmodels.distributions.empirical_distribution.StepFunction`
    with ``side='right'`` option to be like a CDF.

    Parameters
    ----------
    x_grid : ndarray, shape (n_grid,)
        Values to evaluate the stepwise function at.
    xp : ndarray, shape (n_samples,)
        Points at which the step function changes. Typically of type float.
    yp : ndarray, shape (n_samples,)
        The new values at each of the steps
    ival : scalar or None
        Initial value for step function, e.g., the value of the step function
        at -inf. If None, we just require that all `x_grid` values are after
        the first step.
    assume_sorted : bool
        Set to True is `xp` is alreaded sorted in increasing order. This skips
        sorting for computational speed.
    skip_unique_chk: bool
        Assume all values in `xp` are sorted and unique. Setting to True skips
        checking this condition for speed.

    Returns
    -------
    y_grid : ndarray, shape (n_grid,)
        Step function defined by `xp` and `yp` evaluated at the points in
        `x_grid`.
    '''
    assert(x_grid.ndim == 1)
    assert(xp.ndim == 1 and xp.shape == yp.shape)

    if not assume_sorted:
        idx = np.argsort(xp)
        xp, yp = xp[idx], yp[idx]

    # Step function not well defined if xp grid has duplicates
    assert(skip_unique_chk or np.all(np.diff(xp) > 0.0))

    if ival is None:
        assert(len(x_grid) == 0 or np.all(xp[0] <= x_grid))
    else:
        assert(np.ndim(ival) == 0.0)
        xp = np.concatenate(([-np.inf], xp))
        yp = np.concatenate(([ival], yp))

    idx = np.searchsorted(xp, x_grid, side='right') - 1
    y_grid = yp[idx]
    return y_grid


def interp1d(x_grid, xp, yp, kind='linear'):
    """Wrap `scipy.interpolate.interp1d` so it can handle ``'previous'`` like
    MATLAB's `interp1` function. ``'next'`` may be added in future.

    This wrapper does not support extrapolation at the moment. Scipy ENH such
    as #6718 may be merged to scipy master in the near future and make this
    wrapper obsolete.

    Parameters
    ----------
    x_grid : ndarray, shape (n_grid,)
        Values to evaluate the stepwise function at.
    xp : ndarray, shape (n_samples,)
        Points at which the step function changes. Typically of type float.
    yp : ndarray, shape (n_samples,)
        The new values at each of the steps
    kind : str
        Type of interpolation scheme, must be ``'previous'`` or any method that
        `scipy.interpolate.interp1d` can process such as ``'linear'``.

    Returns
    -------
    y_grid : ndarray, shape (n_grid,)
        Interpolation `xp` and `yp` evaluated at the points in `x_grid`.
    """
    # TODO use np vectorize to make vectorized version
    if kind == 'previous':
        # eval_step_func does not work when x points overlap so need to call
        # unique_take_last to get final one.
        xp, yp = unique_take_last(xp, yp)
        y_grid = eval_step_func(x_grid, xp, yp, assume_sorted=True)
    elif kind == 'next':
        # It would be easy to modify eval_step_func to handle this case, but we
        # don't have any need for it right now.
        raise NotImplementedError
    else:
        # interp1d appears to do the right thing with points on top of each
        # other if assume_sorted=True and given sorted data, but can apply
        # cummax strict to make all points exactly unique to be extra safe.
        assert(np.all(np.diff(xp) >= 0))
        # TODO assert func gives same result with either spacing mode
        xp = cummax_strict(xp) if STRICT_SPACING else xp
        f = si.interp1d(xp, yp, kind=kind, assume_sorted=True)
        y_grid = f(x_grid)
    return y_grid


def area(x_curve, y_curve, kind):
    """Compute area under function in vectorized way.

    Parameters
    ----------
    x_curve : ndarray, shape (n_thresholds, n_boot)
        The sample points corresponding to the y values. Must be sorted.
    y_curve : ndarray, shape (n_thresholds, n_boot)
        Input array to integrate. Must be same size as `x_curve`. Operation
        performed independently for each column.
    kind : {'linear', 'kind'}
        Type of interpolation scheme to turn points into lines.

    Returns
    -------
    auc : ndarray, shape (n_boot,)
        Area under curve. Has same length as `x_curve` has columns.
    """
    # TODO move test for this to right file
    # TODO test that this agrees with interp1d for tight interp grid
    # TODO assert shapes
    assert(not np.any(np.isnan(x_curve)))
    assert(not np.any(np.isnan(y_curve)))

    if kind == 'previous':
        # Treat 0*inf as 0:
        with np.errstate(invalid='ignore'):
            auc = np.nansum(y_curve[:-1, :] * np.diff(x_curve, axis=0), axis=0)
    elif kind == 'linear':
        auc = np.trapz(y_curve, x_curve, axis=0)
    else:
        raise NotImplementedError
    return auc
