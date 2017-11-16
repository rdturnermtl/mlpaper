# Ryan Turner (turnerry@iro.umontreal.ca)
from joblib import Memory
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.misc import logsumexp
from constants import LABEL, METHOD, METRIC, STAT, CURVE_STATS
from constants import STD_STATS, PVAL_COL, ERR_COL
import perf_curves as pc

PAIRWISE_DEFAULT = False
N_GRID = 100
PKL_EXT = '.checkpoint'

# ============================================================================
# Functions to move to util
# ============================================================================


def one_hot(y, n_labels):
    '''Equivalent to OneHotEncoder in sklearn but avoids extra dependency.'''
    N, = y.shape
    assert(n_labels >= 1)
    assert(y.dtype.kind == 'i')
    assert(np.all(0 <= y) and np.all(y < n_labels))

    y_bin = np.zeros((N, n_labels), dtype=bool)
    y_bin[xrange(N), y] = True
    return y_bin


def normalize(log_pred_prob):
    assert(log_pred_prob.ndim == 2)
    assert(log_pred_prob.shape[1] >= 1)  # Otherwise, can't make it sum to 1

    normalizer = logsumexp(log_pred_prob, axis=1, keepdims=True)
    log_pred_prob = log_pred_prob - normalizer
    return log_pred_prob


def epsilon_noise(x, default_epsilon=1e-10, max_epsilon=1.0):
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


def ttest1(x, nan_on_zero=False):
    assert(x.size > 0)
    if np.std(x) == 0.0:
        pval = np.nan if nan_on_zero else 1.0
    else:
        _, pval = ss.ttest_1samp(x, 0.0)
    return pval


def t_EB(x, confidence=0.95):
    '''I assumed to support NaN then it became too much headache.'''
    assert(np.ndim(x) == 1 and (not np.any(np.isnan(x))))
    assert(np.ndim(confidence) == 0)
    assert(0.0 < confidence and confidence < 1.0)

    N = x.size
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
    '''I assumed to support NaN then it became too much headache.'''
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
    '''This CI is not designed to work with nans.'''
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

# ============================================================================
# Handy specific utils
# ============================================================================


def get_mean_and_EB(loss, loss_ref=0.0, confidence=0.95, min_EB=0.0,
                    lower=-np.inf, upper=np.inf, method='t'):
    assert(loss.ndim == 1 and np.ndim(loss_ref) <= 1)
    assert(np.ndim(min_EB) == 0)
    mu = np.nanmean(loss)

    delta = loss - loss_ref

    # Note we are computing CI on delta to reference!
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


def shape_and_validate(y, log_pred_prob):
    # Note: This does not check normalization
    N, n_labels = log_pred_prob.shape
    assert(N >= 1)  # Otherwise min and max confused
    assert(n_labels >= 1)  # Otherwise makes no sense
    assert(y.shape == (N,) and y.dtype.kind in ('b', 'i'))
    assert(0 <= y.min() and y.max() < n_labels)
    return N, n_labels

# ============================================================================
# Loss functions
# ============================================================================


def hard_loss_binary(y_bool, log_pred_prob, FP_cost=1.0):
    N, n_labels = shape_and_validate(y_bool, log_pred_prob)
    assert(n_labels == 2)
    assert(FP_cost > 0.0)

    FN_cost = 1.0
    thold = np.log(FP_cost / (FP_cost + FN_cost))

    yhat = log_pred_prob[:, 1] >= thold
    loss = (~y_bool * yhat) * FP_cost + (y_bool * ~yhat) * FN_cost
    return loss


def hard_loss_decision(log_pred_prob, loss_mat):
    pred_prob = np.exp(log_pred_prob)
    E_loss = np.dot(pred_prob, loss_mat)
    action = np.argmin(E_loss, axis=1)
    return action


def hard_loss(y, log_pred_prob, loss_mat=None):
    N, n_labels = shape_and_validate(y, log_pred_prob)
    loss_mat = (1.0 - np.eye(n_labels)) if loss_mat is None else loss_mat
    assert(np.ndim(loss_mat) == 2 and loss_mat.shape[0] == n_labels)
    assert(loss_mat.shape[1] >= 1)  # Must be least one action

    action = hard_loss_decision(log_pred_prob, loss_mat)

    assert(action.shape == y.shape and action.dtype.kind == 'i')
    loss = loss_mat[y.astype(int), action]
    assert(loss.shape == (N,))
    return loss


def log_loss(y, log_pred_prob):
    N, n_labels = shape_and_validate(y, log_pred_prob)
    nll = -log_pred_prob[np.arange(N), y.astype(int)]
    return nll


def brier_loss(y, log_pred_prob, rescale=True):
    N, n_labels = shape_and_validate(y, log_pred_prob)

    y_bin = one_hot(y.astype(int), n_labels)
    loss = np.sum((np.exp(log_pred_prob) - y_bin) ** 2, axis=1)

    if rescale and n_labels > 1:
        # Linearly rescale so perfect is 0.0 and uniform gives 1.0
        loss = np.true_divide(n_labels, n_labels - 1) * loss
    return loss


def spherical_loss(y, log_pred_prob, rescale=True):
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
# Use and summarize loss functions
# ============================================================================


def loss_table(log_pred_prob_table, y, metrics_dict, assume_normalized=False):
    methods, labels = log_pred_prob_table.columns.levels
    N, n_labels = len(log_pred_prob_table), len(labels)
    assert(y.shape == (N,))
    assert(N >= 1 and n_labels >= 1 and len(methods) >= 1)

    col_names = pd.MultiIndex.from_product([metrics_dict.keys(), methods],
                                           names=[METRIC, METHOD])
    loss_tbl = pd.DataFrame(index=log_pred_prob_table.index,
                            columns=col_names, dtype=float)
    for method in methods:
        log_pred_prob = log_pred_prob_table[method].values
        assert(log_pred_prob.shape == (N, n_labels))
        assert(not np.any(np.isnan(log_pred_prob)))  # Would let method cheat

        if not assume_normalized:
            log_pred_prob = normalize(log_pred_prob)

        for metric, metric_f in metrics_dict.iteritems():
            loss_tbl.loc[:, (metric, method)] = metric_f(y, log_pred_prob)
    return loss_tbl


def loss_summary_table(loss_table, ref_method,
                       pairwise_CI=PAIRWISE_DEFAULT, confidence=0.95):
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

# ============================================================================
# Use and summarize performance curves
# ============================================================================


def check_curve(curve):
    x_curve, y_curve, _ = curve  # Skipping tholds since not used here
    assert(x_curve.ndim == 2 and y_curve.ndim == 2)
    assert(x_curve.shape == y_curve.shape)
    assert(np.all(np.isfinite(x_curve)))
    # PRG can be -inf, but all curves should be < inf
    assert(np.all(y_curve < np.inf))
    # x should be sorted
    assert(np.all(np.diff(x_curve, axis=0) >= 0.0))
    return curve


def check_summary(cs):
    assert(cs.ndim == 1)
    # PRG can be -inf, but all curves should be < inf
    assert(np.all(cs < np.inf))
    return cs


def curve_boot(y, log_pred_prob,
               log_pred_prob_ref=None, default_summary_ref=np.nan,
               curve_f=pc.roc_curve, summary_f=pc.auc_trapz, x_grid=None,
               n_boot=1000, pairwise_CI=PAIRWISE_DEFAULT, confidence=0.95):
    N, n_labels = shape_and_validate(y, log_pred_prob)
    assert(n_labels == 2)
    assert(log_pred_prob_ref is None or
           log_pred_prob_ref.shape == log_pred_prob.shape)
    assert(n_boot >= 1)
    assert(np.ndim(confidence) == 0 and 0.0 < confidence and confidence < 1.0)
    assert(not np.any(np.isnan(log_pred_prob)))  # Would let method cheat

    # Set weights to at least epsilon instead of zero in bootstrap since some
    # routines have trouble with zero weight.
    epsilon = 1e-10

    x_grid = np.linspace(0.0, 1.0, N_GRID) if x_grid is None else x_grid
    assert(np.ndim(x_grid) == 1)

    # Setup levels for percentile function
    alpha = 0.5 * (1.0 - confidence)
    q_levels = (100.0 * alpha, 100.0 * (1.0 - alpha))

    # Make everything a vector
    y = y.astype(bool)
    log_pred_prob = log_pred_prob[:, 1]

    # Basic no-boot strap result
    x_curve, y_curve, _ = check_curve(curve_f(y, log_pred_prob))
    mu, = check_summary(summary_f(x_curve, y_curve))
    assert(mu.ndim == 0)

    # To really plot curve at full precision we should use union of x_curve
    # and x_grid, but if x_grid is also random the validity of the following
    # bootstrap procedure is more difficult to determine.
    x_curve_, y_curve_ = pc.make_into_step(x_curve[:, 0], y_curve[:, 0])
    y_curve = eval_step_func(x_grid, x_curve_, y_curve_)

    p_BS = np.ones(N) / N
    weight = np.maximum(epsilon, np.random.multinomial(N, p_BS, size=n_boot).T)

    xp_boot, yp_boot, _ = check_curve(curve_f(y, log_pred_prob, weight))
    curve_summary = check_summary(summary_f(xp_boot, yp_boot))

    curve_summary_ref = default_summary_ref
    if log_pred_prob_ref is not None:
        assert(not np.any(np.isnan(log_pred_prob_ref)))
        log_pred_prob_ref = log_pred_prob_ref[:, 1]

        xp_boot_ref, yp_boot_ref, _ = \
            check_curve(curve_f(y, log_pred_prob_ref, weight))
        curve_summary_ref = check_summary(summary_f(xp_boot_ref, yp_boot_ref))

    # Unclear if there is efficient way to vectorize this
    yp_boot_grid = np.zeros((x_grid.size, n_boot))
    for nn in xrange(n_boot):
        x_curve_, y_curve_ = pc.make_into_step(xp_boot[:, nn], yp_boot[:, nn])
        yp_boot_grid[:, nn] = eval_step_func(x_grid, x_curve_, y_curve_)

    # Summary stat: Get EB
    # This could create problems if curve_summary and curve_summary_ref both
    # have inf values.
    delta = curve_summary - curve_summary_ref if pairwise_CI else curve_summary
    mu_delta = np.mean(delta)
    LB, UB = np.percentile(delta, q_levels)
    # This could nan-out if everything is inf
    EB = np.fmax(UB - mu_delta, mu_delta - LB)
    assert(EB >= 0.0 or np.isnan(EB))

    # Summary stat: Get p-val (two-sided)
    pval = 2.0 * np.minimum(np.mean(curve_summary <= curve_summary_ref),
                            np.mean(curve_summary_ref <= curve_summary))
    pval = np.minimum(1.0, pval)

    summary = (mu, EB, pval)

    # Summarize the curves
    y_LB = np.percentile(yp_boot_grid, q_levels[0], axis=1)
    y_UB = np.percentile(yp_boot_grid, q_levels[1], axis=1)
    curve = pd.DataFrame(data=np.stack((x_grid, y_curve, y_LB, y_UB), axis=1),
                         index=xrange(x_grid.size), columns=CURVE_STATS,
                         dtype=float)
    return summary, curve


def curve_summary_table(log_pred_prob_table, y,
                        curve_dict, ref_method, x_grid=None,
                        n_boot=1000, pairwise_CI=PAIRWISE_DEFAULT,
                        confidence=0.95):
    methods, labels = log_pred_prob_table.columns.levels
    N, n_labels = len(log_pred_prob_table), len(labels)
    assert(y.shape == (N,))
    assert(ref_method in methods)  # ==> len(methods) >= 1
    assert(N >= 1 and n_labels >= 1 and len(curve_dict) >= 1)

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
        log_pred_prob = log_pred_prob_table[method].values
        assert(log_pred_prob.shape == (N, n_labels))

        for curve_name, curve_and_area_f in curve_dict.iteritems():
            curve_f, summary_f = curve_and_area_f
            R = curve_boot(y, log_pred_prob,
                           log_pred_prob_ref=log_pred_prob_ref,
                           curve_f=curve_f, summary_f=summary_f, x_grid=x_grid,
                           n_boot=n_boot, pairwise_CI=pairwise_CI,
                           confidence=confidence)
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
STD_MULTICLASS_LOSS = {'NLL': log_loss, 'Brier': brier_loss,
                       'sphere': spherical_loss}

STD_BINARY_LOSS = {'NLL': log_loss, 'Brier': brier_loss,
                   'sphere': spherical_loss, 'zero_one': hard_loss_binary}

STD_BINARY_CURVES = {'AUC': (pc.roc_curve, pc.auc_trapz),
                     'AP': (pc.recall_precision_curve, pc.auc_left),
                     'AUPRG': (pc.prg_curve, pc.auc_left)}


class JustNoise:
    '''Class version of iid predictor compatible with sklearn interface.'''

    def __init__(self):
        self.pred = [np.nan, np.nan]

    def fit(self, X_train, y_train):
        P = np.mean(y_train)
        self.pred = [np.log(1.0 - P), np.log(P)]

    def predict_log_proba(self, X_test):
        N = X_test.shape[0]
        pred_log_prob = np.repeat([self.pred], N, axis=0)
        return pred_log_prob


def get_pred_log_prob(X_train, y_train, X_test, n_labels, methods,
                      min_log_prob=-np.inf, verbose=False, checkpointdir=None):
    '''Note that the method objects will not be fit if the checkpoint result
    is available.'''
    n_test = X_test.shape[0]
    assert(X_train.ndim == 2)
    assert(y_train.shape == (X_train.shape[0],))
    assert(y_train.dtype.kind in ('b', 'i'))
    assert(0 <= y_train.min() and y_train.max() < n_labels)
    assert(X_test.ndim == 2 and X_test.shape[1] == X_train.shape[1])

    memory = Memory(cachedir=checkpointdir, verbose=0)

    @memory.cache
    def train_predict(method_obj, X_train, y_train, X_test):
        method_obj.fit(X_train, y_train)
        try:
            pred_log_prob = method_obj.predict_log_proba(X_test)
        except:  # If there is no log proba available
            pred_log_prob = np.log(method_obj.predict_proba(X_test))
        return pred_log_prob

    col_names = pd.MultiIndex.from_product([methods.keys(), xrange(n_labels)],
                                           names=[METHOD, LABEL])
    pred_tbl = pd.DataFrame(index=xrange(n_test), columns=col_names,
                            dtype=float)
    for method_name, method_obj in methods.iteritems():
        if verbose:
            print 'Running fit/predict for %s' % method_name
        pred_log_prob = train_predict(method_obj, X_train, y_train, X_test)
        assert(pred_log_prob.shape == (n_test, n_labels))

        pred_log_prob = normalize(np.maximum(min_log_prob, pred_log_prob))
        pred_tbl.loc[:, method_name] = pred_log_prob
    return pred_tbl


def just_benchmark(X_train, y_train, X_test, y_test, n_labels,
                   methods, loss_dict, curve_dict, ref_method,
                   min_pred_log_prob=-np.inf):
    pred_tbl = get_pred_log_prob(X_train, y_train, X_test, n_labels,
                                 methods, min_pred_log_prob=min_pred_log_prob)
    full_tbl, dump = summary_table(pred_tbl, y_test, loss_dict, curve_dict,
                                   ref_method)
    return full_tbl, dump

if __name__ == '__main__':
    import sciprint as sp

    np.random.seed(35634)

    N = 500
    n_labels = 2

    methods = ['foo', 'bar', 'baz']
    labels = xrange(n_labels)

    col_names = pd.MultiIndex.from_product([methods, labels],
                                           names=[METHOD, LABEL])
    dat = np.random.randn(N, n_labels * len(methods))
    log_pred_prob_table = pd.DataFrame(data=dat,
                                       index=xrange(N), columns=col_names,
                                       dtype=float)
    y = np.random.rand(N) <= 0.3

    loss_tbl = loss_table(log_pred_prob_table, y, metrics_dict=STD_BINARY_LOSS)
    perf_tbl = loss_summary_table(loss_tbl, 'foo')
    curve_tbl, dump1 = curve_summary_table(log_pred_prob_table, y,
                                           STD_BINARY_CURVES, 'foo')
    full_tbl, dump2 = summary_table(log_pred_prob_table, y, STD_BINARY_LOSS,
                                    STD_BINARY_CURVES, 'foo')
    print full_tbl

    print sp.just_format_it(full_tbl, shift_mod=3, unit_dict={'NLL': 'nats'},
                            crap_limit_min={'AUPRG': -1},
                            crap_limit_max={'zero_one': -1},
                            EB_limit={'AUPRG': -1},
                            non_finite_fmt={sp.NAN_STR: '{--}'}, use_tex=True)
    print sp.just_format_it(full_tbl, shift_mod=3, unit_dict={'NLL': 'nats'},
                            crap_limit_min={'AUPRG': -1},
                            crap_limit_max={'zero_one': -1},
                            non_finite_fmt={sp.NAN_STR: 'N/A'}, use_tex=False)
