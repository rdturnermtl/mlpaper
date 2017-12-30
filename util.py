# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
from scipy.misc import logsumexp


def one_hot(y, n_labels):
    '''Same functionality `sklearn.preprocessing.OneHotEncoder` but avoids
    extra dependency.

    Parameters
    ----------
    y : 1d np array of int type
        Integers in range ``[0, n_labels)`` to be one-hot encoded.
    n_labels : int
        Number of labels, must be >= 1. This is not infered from `y` because
        some labels may not be found in small data chunks.

    Returns
    -------
    y_bin : 2d np array of bool type
        One hot encoding of `y`, with size ``(len(y), n_labels)``
    '''
    N, = y.shape
    assert(n_labels >= 1)
    assert(y.dtype.kind == 'i')
    assert(np.all(0 <= y) and np.all(y < n_labels))

    y_bin = np.zeros((N, n_labels), dtype=bool)
    y_bin[xrange(N), y] = True
    return y_bin


def normalize(log_pred_prob):
    '''Normalize log probability distributions for classification.

    Parameters
    ----------
    log_pred_prob : 2d np array
        Each row corresponds to a categorical distribution with unnormalized
        probabilities in log scale. Therefore, the number of columns must be at
        least 1.

    Returns
    -------
    log_pred_prob : 2d np array
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
    x : 1d np array
        Input vector to be noise corrupted. Must have all finite values.
    default_epsilon : float
        Default noise to add for singleton lists, musts be > 0.0.
    max_epsilon : float
        Maximum amount of noise corruption regardless of scale found in `x`.

    Returns
    -------
    x : 1d np array of float type
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


def eval_step_func(x_grid, xp, yp, ival=None,
                   assume_sorted=False, skip_unique_chk=False):
    '''Evaluate a stepwise function. Based on the ECDF class in statsmodels.
    The function is assumed to cadlag (like a CDF function).

    This is a non-OOP equivalent to class:
    `statsmodels.distributions.empirical_distribution.StepFunction`
    with ``side='right'`` option to be like a CDF.

    Parameters
    ----------
    x_grid : 1d np array
        Values to evaluate the stepwise function at.
    xp : 1d np array
        Points at which the step function changes. Typically of type float.
    yp : 1d np array
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
    y_grid : 1d np array
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


def make_into_step(xp, yp):
    """Make pairs of `xp` and `yp` vectors into proper step function. That is,
    remove NaN `xp` values and multiple steps at same location.

    Parameters
    ----------
    xp : 1d np array
        The sample points corresponding to the y values. Must be sorted.
    yp : 1d np array
        Values in y-axis for step function.

    Returns
    -------
    xp : 1d np array
        Input `xp` after removing extra points.
    yp : 1d np array
        Input `yp` after removing extra points.

    Notes
    -----
    Keeps last value in list when multiple steps happen at the same x-value.
    """
    assert(xp.ndim == 1 and xp.shape == yp.shape)

    idx = ~np.isnan(xp)
    xp, yp = xp[idx], yp[idx]

    deltas = np.diff(xp)
    assert(not np.any(deltas < -1e-10))

    idx = [] if xp.size == 0 else np.concatenate((deltas > 0, [True]))
    return xp[idx], yp[idx]
