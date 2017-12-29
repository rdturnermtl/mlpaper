# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function
from builtins import range
from string import ascii_letters
import constants
import numpy as np
import pandas as pd
import benchmark_tools as bt
import scipy.stats as ss
from sklearn.metrics import brier_score_loss, log_loss, zero_one_loss
from sklearn.preprocessing import OneHotEncoder, normalize
from statsmodels.distributions.empirical_distribution import StepFunction

def test_t_EB(runs=100, trials=1000):
    pval = []
    while len(pval) < runs:
        N = np.random.randint(low=0, high=10)
        confidence = np.random.rand()

        if N <= 1:
            x = np.random.randn(N)
            EB = bt.t_EB(x, confidence=confidence)
            assert(EB == np.inf)
        else:
            fail = 0
            for tt in range(trials):
                x = np.random.randn(N)

                EB = bt.t_EB(x, confidence=confidence)
                mu = np.nanmean(x)
                LB, UB = mu - EB, mu + EB
                assert(np.isfinite(LB) and np.isfinite(UB))
                fail += (0.0 < LB) or (UB < 0.0)
            pval.append(ss.binom_test(fail, trials, 1.0 - confidence))
    _, pval_agg = ss.combine_pvalues(pval)
    return pval_agg


def test_get_mean_and_EB(runs=100, trials=1000):
    pval = []
    while len(pval) < runs:
        N = np.random.randint(low=0, high=10)
        confidence = np.random.rand()

        if N <= 1:
            x = 2.0 + np.random.randn(N)
            x_ref = 1.5 + np.random.randn(N)

            mu, EB = bt.get_mean_and_EB(x, x_ref, confidence=confidence)
            assert(np.allclose(mu, np.nanmean(mu), equal_nan=True))
            assert(EB == np.inf)

            mu, EB = bt.get_mean_and_EB(x, confidence=confidence)
            mu2, EB2 = bt.get_mean_and_EB(x, np.zeros_like(x),
                                          confidence=confidence)
            assert(np.allclose(mu, np.nanmean(mu2), equal_nan=True))
            assert(np.allclose(EB, np.nanmean(EB2), equal_nan=True))
        else:
            fail = 0
            for tt in range(trials):
                x = 2.0 + np.random.randn(N)
                x_ref = 1.5 + np.random.randn(N)

                mu, EB = bt.get_mean_and_EB(x, x_ref, confidence=confidence)
                assert(np.isfinite(mu) and np.isfinite(EB))
                assert(np.allclose(mu, np.nanmean(mu), equal_nan=True))
                err = np.nanmean(x - x_ref) - 0.5
                fail += np.abs(err) > EB

                mu, EB = bt.get_mean_and_EB(x, confidence=confidence)
                mu2, EB2 = bt.get_mean_and_EB(x, np.zeros_like(x),
                                              confidence=confidence)
                assert(np.allclose(mu, np.nanmean(mu2), equal_nan=True))
                assert(np.allclose(EB, np.nanmean(EB2), equal_nan=True))
            pval.append(ss.binom_test(fail, trials, 1.0 - confidence))
    _, pval_agg = ss.combine_pvalues(pval)
    return pval_agg


def test_one_hot():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    z0 = bt.one_hot(y, n_labels)
    assert(z0.dtype.kind == 'b' and z0.shape == (N, n_labels))

    if N >= 1:
        enc = OneHotEncoder(n_values=n_labels, sparse=False, dtype=bool)
        z1 = enc.fit_transform(y[:, None])
        assert(np.all(z0 == z1))


def test_normalize():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    log_pred_prob = np.random.randn(N, n_labels)
    z0 = bt.normalize(log_pred_prob)
    assert(z0.shape == (N, n_labels))

    if N >= 1:
        z1 = np.log(normalize(np.exp(log_pred_prob), norm='l1', axis=1))
        assert(np.allclose(z0, z1))


def epsilon_noise_test():
    N = np.random.randint(low=0, high=10)
    x = np.random.randn(N)

    _, idx0 = np.unique(x, return_inverse=True)
    _, idx1 = np.unique(bt.epsilon_noise(x), return_inverse=True)
    assert(np.all(idx0 == idx1))


def eval_step_func_test():
    N = np.random.randint(low=0, high=10)

    xp = np.random.randn(N)
    yp = np.random.randn(N)
    ival = np.random.randn()

    assume_sorted = np.random.rand() <= 0.5
    if assume_sorted:
        xp = np.sort(xp)

    y_grid = bt.eval_step_func(xp, xp, yp, assume_sorted=assume_sorted)
    assert(np.allclose(y_grid, yp))

    N_test = np.random.randint(low=0, high=10)
    x_grid = np.concatenate((xp, np.random.randn(N_test)))

    y_grid = bt.eval_step_func(x_grid, xp, yp, ival)

    SF = StepFunction(xp, yp, ival=ival, side='right', sorted=assume_sorted)
    y_grid2 = SF(x_grid)

    assert(np.allclose(y_grid, y_grid2))


def hard_loss_binary_test():
    '''Also tests hard loss.'''
    n_labels = 2
    N = np.random.randint(low=1, high=10)

    y_bool = np.random.rand(N) <= 0.5
    y_pred = bt.normalize(np.random.randn(N, n_labels))
    loss = bt.hard_loss_binary(y_bool, y_pred)

    act = bt.hard_loss_decision(y_pred, 1.0 - np.eye(n_labels))
    loss2 = zero_one_loss(y_bool.astype(int), act)
    assert(np.allclose(np.mean(loss), loss2))

    loss2 = bt.hard_loss(y_bool, y_pred)
    assert(np.allclose(loss, loss2))


def hard_loss_decision_test():
    n_labels = np.random.randint(low=1, high=10)
    n_act = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    y_pred = bt.normalize(np.random.randn(N, n_labels))

    act = bt.hard_loss_decision(y_pred, 1.0 - np.eye(n_labels))
    assert(np.all(np.argmax(y_pred, axis=1) == act))

    loss_mat = np.random.rand(n_labels, n_act)
    act = bt.hard_loss_decision(y_pred, loss_mat)

    loss_mat = np.concatenate((loss_mat, np.ones((n_labels, 1))), axis=1)
    act2 = bt.hard_loss_decision(y_pred, loss_mat)
    assert(np.all(act == act2))

    loss_mat = np.concatenate((loss_mat, np.zeros((n_labels, 1))), axis=1)
    act2 = bt.hard_loss_decision(y_pred, loss_mat)
    assert(np.all(act2 == loss_mat.shape[1] - 1))


def log_loss_test():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=1, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = bt.normalize(np.random.randn(N, n_labels))

    if n_labels >= 2:
        loss = bt.log_loss(y, y_pred)
        loss2 = log_loss(y, np.exp(y_pred), labels=range(n_labels))
        assert(np.allclose(np.mean(loss), loss2))

    with np.errstate(invalid='ignore', divide='ignore'):
        pred = np.log(bt.one_hot(y, n_labels))
    loss2 = bt.log_loss(y, pred)
    assert(np.max(np.abs(loss2)) <= 1e-8)


def brier_loss_test():
    n_labels = np.random.randint(low=1, high=4)
    N = np.random.randint(low=1, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = bt.normalize(np.random.randn(N, n_labels))

    loss = bt.brier_loss(y, y_pred, rescale=False)
    # sklearn learn is dumb and gets confused when only one class passed in
    if n_labels == 2 and np.std(y) >= 1e-8:
        loss2 = brier_score_loss(y == 1, np.exp(y_pred[:, 1]), pos_label=True)
        assert(np.allclose(np.mean(loss), 2.0 * loss2))

    with np.errstate(invalid='ignore', divide='ignore'):
        pred = np.log(bt.one_hot(y, n_labels))
    loss2 = bt.brier_loss(y, pred, rescale=False)
    assert(np.max(np.abs(loss2)) <= 1e-8)


def spherical_loss_test():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=1, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = bt.normalize(np.random.randn(N, n_labels))

    loss = bt.spherical_loss(y, y_pred, rescale=False)

    # Check against the linear implementation
    pred_prob = np.exp(y_pred)
    normalizer = np.sqrt(np.sum(pred_prob ** 2, axis=1))
    loss_0 = -pred_prob[np.arange(N), y.astype(int)] / normalizer
    assert(np.allclose(loss, loss_0, equal_nan=True))

    with np.errstate(invalid='ignore', divide='ignore'):
        pred = np.log(bt.one_hot(y, n_labels))
    loss2 = bt.spherical_loss(y, pred, rescale=False)
    assert(np.max(np.abs(loss2 + 1.0)) <= 1e-8)

    if n_labels >= 2:
        with np.errstate(invalid='ignore', divide='ignore'):
            pred_prob = bt.normalize(np.log(1.0 - bt.one_hot(y, n_labels)))
        loss2 = bt.spherical_loss(y, pred_prob, rescale=False)
        assert(np.max(np.abs(loss2)) <= 1e-8)


def loss_summary_table_test():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=1, high=10)
    n_methods = np.random.randint(low=1, high=5)

    confidence = np.random.rand()
    pairwise_CI = np.random.rand() <= 0.5

    methods = np.random.choice(list(ascii_letters), n_methods, replace=False)
    ref = np.random.choice(methods)
    metrics = bt.STD_MULTICLASS_LOSS
    labels = range(n_labels)

    col_names = pd.MultiIndex.from_product([methods, labels],
                                           names=[bt.METHOD, bt.LABEL])
    dat = np.random.randn(N, n_labels * len(methods))
    tbl = pd.DataFrame(data=dat, index=range(N), columns=col_names,
                       dtype=float)

    y = np.random.randint(low=0, high=n_labels, size=N)
    loss_tbl = bt.loss_table(tbl, y, metrics_dict=metrics)
    perf_tbl = bt.loss_summary_table(loss_tbl, ref, pairwise_CI=pairwise_CI,
                                     confidence=confidence)
    for metric, metric_f in metrics.iteritems():
        loss_ref = metric_f(y, bt.normalize(tbl[ref].values))
        for method in methods:
            loss = metric_f(y, bt.normalize(tbl[method].values))
            assert(np.allclose(loss_tbl[(metric, method)].values, loss,
                               equal_nan=True))

            if pairwise_CI:
                mu, EB = bt.get_mean_and_EB(loss, loss_ref, confidence)
                if method == ref:
                    EB = np.nan
            else:
                mu, EB = bt.get_mean_and_EB(loss, confidence)

            if method == ref:
                pval = np.nan
            elif np.std(loss - loss_ref) == 0.0:
                pval = 1.0
            else:
                _, pval = ss.ttest_1samp(loss - loss_ref, 0.0)
            assert(np.allclose(perf_tbl.loc[method, metric].values,
                               [mu, EB, pval], equal_nan=True))

np.random.seed(53634)

for _ in range(constants.MC_REPEATS_1K):
    test_one_hot()
    test_normalize()
    epsilon_noise_test()
    hard_loss_binary_test()
    hard_loss_decision_test()
    log_loss_test()
    brier_loss_test()
    spherical_loss_test()
    loss_summary_table_test()
    eval_step_func_test()
print('Now running MC tests')

print(test_t_EB(trials=constants.MC_REPEATS_1K))
print(test_get_mean_and_EB(trials=constants.MC_REPEATS_1K))
