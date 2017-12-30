# Ryan Turner (turnerry@iro.umontreal.ca)
from joblib import Memory
import numpy as np
import pandas as pd
import scipy.stats as ss
from constants import METHOD, METRIC
from benchmark_tools import loss_summary_table, PAIRWISE_DEFAULT

MOMENT = 'moment'  # Don't put in constants since only needed for regression


def shape_and_validate(y, mu, std):
    '''Validate shapes and types of predictive distribution against data and
    return the shape information.

    Parameters
    ----------
    y : 1d np array
        True targets for each regression data point. Typically of type `float`.
    mu : 1d np array
        Predictive mean for each regression data point. Typically of type
        `float`. Must be of same shape as `y`.
    std : 1d np array
        Predictive standard deviation for each regression data point. Typically
        of type `float`. Must be positive and of same shape as `y`.

    Returns
    -------
    N : int
        Number of data points (length of `y`)
    '''
    N, = y.shape
    assert(N >= 1)  # Otherwise min and max confused
    assert(mu.shape == (N,) and std.shape == (N,))
    assert(np.all(np.isfinite(mu)) and np.all(np.isfinite(std)))
    assert(np.all(std > 0.0))
    return N

# ============================================================================
# Loss functions
# ============================================================================


def square_loss(y, mu, std):
    '''Compute MSE of predictions vs true targets.

    Parameters
    ----------
    y : 1d np array
        True targets for each regression data point. Typically of type `float`.
    mu : 1d np array
        Predictive mean for each regression data point. Typically of type
        `float`. Must be of same shape as `y`.
    std : 1d np array
        Predictive standard deviation for each regression data point. Typically
        of type `float`. Must be positive and of same shape as `y`. Ignored in
        this function.

    Returns
    -------
    loss : np array of `float`
        Square error of target vs prediction. Same shape as `y`.
    '''
    shape_and_validate(y, mu, std)
    loss = (y - mu) ** 2
    return loss


def abs_loss(y, mu, std):
    '''Compute MAE of predictions vs true targets.

    Parameters
    ----------
    y : 1d np array
        True targets for each regression data point. Typically of type `float`.
    mu : 1d np array
        Predictive mean for each regression data point. Typically of type
        `float`. Must be of same shape as `y`.
    std : 1d np array
        Predictive standard deviation for each regression data point. Typically
        of type `float`. Must be positive and of same shape as `y`. Ignored in
        this function.

    Returns
    -------
    loss : np array of float
        Absolute error of target vs prediction. Same shape as `y`.
    '''
    shape_and_validate(y, mu, std)
    loss = np.abs(y - mu)
    return loss


def log_loss(y, mu, std):
    '''Compute log loss of Gaussian predictive distribution on target `y`.

    Parameters
    ----------
    y : 1d np array
        True targets for each regression data point. Typically of type `float`.
    mu : 1d np array
        Predictive mean for each regression data point. Typically of type
        `float`. Must be of same shape as `y`.
    std : 1d np array
        Predictive standard deviation for each regression data point. Typically
        of type `float`. Must be positive and of same shape as `y`.

    Returns
    -------
    loss : np array of float
        Log loss of Gaussian predictive distribution on target `y`. Same shape
        as `y`.
    '''
    shape_and_validate(y, mu, std)
    loss = -ss.norm.logpdf(y, loc=mu, scale=std)
    return loss

# ============================================================================
# Use and summarize loss functions
# ============================================================================


def loss_table(pred_tbl, y, metrics_dict):
    '''Compute loss table from table of Gaussian predictions.

    Parameters
    ----------
    pred_tbl : Pandas DataFrame
        DataFrame with predictive distributions. Each row is a data point.
        The columns should be hierarchical index that is the cartesian product
        of methods x moments. For exampe, ``log_pred_prob_table.loc[5, 'foo']``
        is a pandas series with (mean, std deviation) prediction that method
        foo places on ``y[5]``. Cannot be empty.
    y : 1d np array
        True targets for each regression data point. Typically of type `float`.
    metrics_dict : dict of str to func
        Dictionary mapping loss function name to function that computes loss,
        e.g., `log_loss`, `square_loss`, ...

    Returns
    -------
    loss_tbl : Pandas DataFrame
        DataFrame with loss of each method according to each loss function on
        each data point. The rows are the data points in `y` (that is the index
        matches `pred_tbl`). The columns are a hierarchical index that is the
        cartesian product of loss x method. That is, the loss of method foo's
        prediction of ``y[5]`` according to loss function bar is stored in
        ``loss_tbl.loc[5, ('bar', 'foo')]``.
    '''
    methods, moments = pred_tbl.columns.levels
    assert('mu' in moments and 'std' in moments)
    N = len(pred_tbl)
    assert(y.shape == (N,))
    assert(N >= 1 and len(methods) >= 1)

    col_names = pd.MultiIndex.from_product([metrics_dict.keys(), methods],
                                           names=[METRIC, METHOD])
    loss_tbl = pd.DataFrame(index=pred_tbl.index, columns=col_names,
                            dtype=float)
    for method in methods:
        # These get validated inside loss function
        mu = pred_tbl[(method, 'mu')].values
        std = pred_tbl[(method, 'std')].values
        for metric, metric_f in metrics_dict.iteritems():
            loss_tbl.loc[:, (metric, method)] = metric_f(y, mu, std)
    return loss_tbl

# ============================================================================
# Variables and functions to make getting results from sklearn objects easy
# ============================================================================

# Pre-build some standard metric dicts for the user
STD_REGR_LOSS = {'NLL': log_loss, 'MSE': square_loss, 'MAE': abs_loss}


class JustNoise:
    '''Class version of iid predictor compatible with sklearn interface.'''

    def __init__(self):
        self.mu = np.nan
        self.std = np.nan

    def fit(self, X_train, y_train):
        assert(y_train.ndim == 1)
        assert(len(y_train) >= 2)  # Require N >= 2 for std
        self.mu = np.mean(y_train)
        self.std = np.std(y_train, ddof=0)

    def predict(self, X_test, return_std=True):
        assert(return_std)
        N = X_test.shape[0]
        mu = np.repeat([self.mu], N, axis=0)
        std = np.repeat([self.std], N, axis=0)
        return mu, std


def get_gauss_pred(X_train, y_train, X_test, methods,
                   min_std=0.0, verbose=False, checkpointdir=None):
    '''Get the Gaussian prediction tables for each test point on a collection
    of regression methods.

    Parameters
    ----------
    X_train : 2d np array
        Training set 2d feature array for classifiers. Each row is an
        indepedent data point and each column is a feature.
    y_train : 1d np array
        True training targets for each regression data point. Typically of type
        `float`. Must be of same length as `X_train`.
    X_test : 2d np array
        Test set 2d feature array for classifiers. Each row is an indepedent
        data point and each column is a feature.
    methods : dict of str to sklearn estimator
        Dictionary mapping method name (`str`) to object that performs training
        and test. Object must follow the interface of sklearn estimators, that
        is, it has a ``fit()`` method and a ``predict()`` method that accepts
        the argument ``return_std=True``.
    min_std : float
        Minimum value to floor the predictive standard deviation. Must be >= 0.
        Useful to prevent inf log loss penalties.
    verbose : bool
        If True, display which method being trained.
    checkpointdir : str (directory)
        If provided, stores checkpoint results using joblib for the train/test
        in case process interrupted. If None, no checkpointing is done.

    Returns
    -------
    pred_tbl : Pandas DataFrame
        DataFrame with predictive distributions. Each row is a data point.
        The columns should be hierarchical index that is the cartesian product
        of methods x moments. For exampe, ``log_pred_prob_table.loc[5, 'foo']``
        is a pandas series with (mean, std deviation) prediction that method
        foo places on ``y[5]``.

    Notes
    -----
    If a train/test operation is loaded from a checkpoint file, the estimator
    object in methods will not be in a fit state.
    '''
    # TODO assert that n_test > 0
    n_test = X_test.shape[0]
    assert(X_train.ndim == 2)
    assert(y_train.shape == (X_train.shape[0],))
    assert(X_test.ndim == 2 and X_test.shape[1] == X_train.shape[1])
    assert(min_std >= 0.0)

    memory = Memory(cachedir=checkpointdir, verbose=0)

    @memory.cache
    def train_predict(method_obj, X_train, y_train, X_test):
        method_obj.fit(X_train, y_train)
        mu, std = method_obj.predict(X_test, return_std=True)
        return mu, std

    col_names = pd.MultiIndex.from_product([methods.keys(), ('mu', 'std')],
                                           names=[METHOD, MOMENT])
    pred_tbl = pd.DataFrame(index=xrange(n_test), columns=col_names,
                            dtype=float)
    for method_name, method_obj in methods.iteritems():
        if verbose:
            print 'Running fit/predict for %s' % method_name
        mu, std = train_predict(method_obj, X_train, y_train, X_test)
        assert(mu.shape == (n_test,) and std.shape == (n_test,))

        std = np.maximum(min_std, std)
        pred_tbl.loc[:, (method_name, 'mu')] = mu
        pred_tbl.loc[:, (method_name, 'std')] = std
    return pred_tbl


def just_benchmark(X_train, y_train, X_test, y_test,
                   methods, loss_dict, ref_method, min_std=0.0,
                   pairwise_CI=PAIRWISE_DEFAULT):
    '''Simplest one-call interface to this package. Just pass it data and
    method objects and a performance summary DataFrame is returned.

    Parameters
    ----------
    X_train : 2d np array
        Training set 2d feature array for classifiers. Each row is an
        indepedent data point and each column is a feature.
    y_train : 1d np array
        True training targets for each regression data point. Typically of type
        `float`. Must be of same length as `X_train`.
    X_test : 2d np array
        Test set 2d feature array for classifiers. Each row is an indepedent
        data point and each column is a feature.
    y_test : 1d np array
        True test targets for each regression data point. Typically of type
        `float`. Cannot be empty. Must be of same length as `X_test`.
    methods : dict of str to sklearn estimator
        Dictionary mapping method name (`str`) to object that performs training
        and test. Object must follow the interface of sklearn estimators, that
        is, it has a ``fit()`` method and a ``predict()`` method that accepts
        the argument ``return_std=True``.
    loss_dict : dict of str to func
        Dictionary mapping loss function name to function that computes loss,
        e.g., `log_loss`, `square_loss`, ...
    ref_method : str
        Name of method that is used as reference point in paired statistical
        tests. This is usually some some of baseline method. `ref_method` must
        be found in `methods` dictionary.
    min_std : float
        Minimum value to floor the predictive standard deviation. Must be >= 0.
        Useful to prevent inf log loss penalties.
    pairwise_CI : bool
        If True, compute error bars on the mean of ``loss - loss_ref`` instead
        of just the mean of `loss`. This typically gives smaller error bars.

    Returns
    -------
    loss_summary : Pandas DataFrame
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
    pred_tbl = get_gauss_pred(X_train, y_train, X_test, methods,
                              min_std=min_std)
    loss_tbl = loss_table(pred_tbl, y_test, loss_dict)
    loss_summary = loss_summary_table(loss_tbl, ref_method,
                                      pairwise_CI=pairwise_CI)
    return loss_summary
