# Ryan Turner (turnerry@iro.umontreal.ca)
from joblib import Memory
import numpy as np
import pandas as pd
import scipy.stats as ss
from constants import METHOD, METRIC
from benchmark_tools import loss_summary_table

MOMENT = 'moment'
PKL_EXT = '.checkpoint'

# ============================================================================
# Handy specific utils
# ============================================================================


def shape_and_validate(y, mu, std):
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
    shape_and_validate(y, mu, std)
    loss = (y - mu) ** 2
    return loss


def abs_loss(y, mu, std):
    shape_and_validate(y, mu, std)
    loss = np.abs(y - mu)
    return loss


def log_loss(y, mu, std):
    shape_and_validate(y, mu, std)
    nll = -ss.norm.logpdf(y, loc=mu, scale=std)
    return nll

# ============================================================================
# Use and summarize loss functions
# ============================================================================


def loss_table(pred_tbl, y, metrics_dict):
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
    '''Note that the method objects will not be fit if the checkpoint result
    is available.'''
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
                   methods, loss_dict, ref_method, min_std=0.0):
    pred_tbl = get_gauss_pred(X_train, y_train, X_test, methods,
                              min_std=min_std)
    loss_tbl = loss_table(pred_tbl, y_test, loss_dict)
    loss_summary = loss_summary_table(loss_tbl, ref_method)
    return loss_summary
