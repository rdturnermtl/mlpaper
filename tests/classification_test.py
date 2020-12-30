# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import brier_score_loss, log_loss, zero_one_loss

import mlpaper.classification as btc
from mlpaper import util
from mlpaper.test_constants import MC_REPEATS_LARGE


def hard_loss_binary(y_bool, log_pred_prob, FP_cost=1.0):
    """Special case of hard_loss."""
    N, n_labels = btc.shape_and_validate(y_bool, log_pred_prob)
    assert n_labels == 2
    assert FP_cost > 0.0

    FN_cost = 1.0
    thold = np.log(FP_cost / (FP_cost + FN_cost))

    y_bool = y_bool.astype(bool)  # So we can use ~
    yhat = log_pred_prob[:, 1] >= thold
    assert y_bool.dtype.kind == "b" and yhat.dtype.kind == "b"

    loss = (~y_bool * yhat) * FP_cost + (y_bool * ~yhat) * FN_cost
    assert np.all((loss == 0) | (loss == FN_cost) | (loss == FP_cost))
    return loss


def test_hard_loss_decision():
    n_labels = np.random.randint(low=1, high=10)
    n_act = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    y_pred = util.normalize(np.random.randn(N, n_labels))

    act = btc.hard_loss_decision(y_pred, 1.0 - np.eye(n_labels))
    assert np.all(np.argmax(y_pred, axis=1) == act)

    loss_mat = np.random.rand(n_labels, n_act)
    act = btc.hard_loss_decision(y_pred, loss_mat)

    loss_mat = np.concatenate((loss_mat, np.ones((n_labels, 1))), axis=1)
    act2 = btc.hard_loss_decision(y_pred, loss_mat)
    assert np.all(act == act2)

    loss_mat = np.concatenate((loss_mat, np.zeros((n_labels, 1))), axis=1)
    act2 = btc.hard_loss_decision(y_pred, loss_mat)
    assert np.all(act2 == loss_mat.shape[1] - 1)


def test_hard_loss_binary():
    """Also tests hard loss."""
    n_labels = 2
    N = np.random.randint(low=1, high=10)

    y_bool = np.random.rand(N) <= 0.5
    y_pred = util.normalize(np.random.randn(N, n_labels))
    loss = hard_loss_binary(y_bool, y_pred)

    act = btc.hard_loss_decision(y_pred, 1.0 - np.eye(n_labels))
    loss2 = zero_one_loss(y_bool.astype(int), act)
    assert np.allclose(np.mean(loss), loss2)

    loss2 = btc.hard_loss(y_bool, y_pred)
    assert np.allclose(loss, loss2)


def test_log_loss():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=1, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = util.normalize(np.random.randn(N, n_labels))

    if n_labels >= 2:
        loss = btc.log_loss(y, y_pred)

        loss2 = log_loss(y, np.exp(y_pred), labels=range(n_labels))
        assert np.allclose(np.mean(loss), loss2)

    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.log(util.one_hot(y, n_labels))
    loss2 = btc.log_loss(y, pred)
    assert np.max(np.abs(loss2)) <= 1e-8


def test_brier_loss():
    n_labels = np.random.randint(low=1, high=4)
    N = np.random.randint(low=1, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = util.normalize(np.random.randn(N, n_labels))

    loss = btc.brier_loss(y, y_pred, rescale=False)
    # sklearn learn is dumb and gets confused when only one class passed in
    if n_labels == 2 and np.std(y) >= 1e-8:
        loss2 = brier_score_loss(y == 1, np.exp(y_pred[:, 1]), pos_label=True)
        assert np.allclose(np.mean(loss), 2.0 * loss2)

    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.log(util.one_hot(y, n_labels))
    loss2 = btc.brier_loss(y, pred, rescale=False)
    assert np.max(np.abs(loss2)) <= 1e-8
    loss2 = btc.brier_loss(y, pred, rescale=True)
    assert np.max(np.abs(loss2)) <= 1e-8

    pred = util.normalize(np.zeros((N, n_labels)))
    loss2 = btc.brier_loss(y, pred, rescale=True)
    if n_labels >= 2:
        assert np.max(np.abs(loss2 - 1.0)) <= 1e-8
    else:
        assert np.max(np.abs(loss2)) <= 1e-8


def test_spherical_loss():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=1, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = util.normalize(np.random.randn(N, n_labels))

    loss = btc.spherical_loss(y, y_pred, rescale=False)

    # Check against the linear implementation
    pred_prob = np.exp(y_pred)
    normalizer = np.sqrt(np.sum(pred_prob ** 2, axis=1))
    loss_0 = -pred_prob[np.arange(N), y.astype(int)] / normalizer
    assert np.allclose(loss, loss_0, equal_nan=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.log(util.one_hot(y, n_labels))
    loss2 = btc.spherical_loss(y, pred, rescale=False)
    assert np.max(np.abs(loss2 + 1.0)) <= 1e-8
    loss2 = btc.spherical_loss(y, pred, rescale=True)
    assert np.max(np.abs(loss2)) <= 1e-8

    if n_labels >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            pred_prob = util.normalize(np.log(1.0 - util.one_hot(y, n_labels)))
        loss2 = btc.spherical_loss(y, pred_prob, rescale=False)
        assert np.max(np.abs(loss2)) <= 1e-8

        loss2 = btc.spherical_loss(y, util.normalize(np.zeros((N, n_labels))), rescale=True)
        assert np.max(np.abs(loss2 - 1.0)) <= 1e-8


def test_rce():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=2, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = util.normalize(np.random.randn(N, n_labels))

    stat = btc.rce_stat(y, y_pred)
    stat_sum = np.sum(stat, axis=0)
    rce = btc.rce_agg(stat_sum[None, :], N)
    rce = rce.item()

    y_one_hot = util.one_hot(y, n_labels)

    loss = -np.mean(np.sum(y_pred * y_one_hot, axis=1))
    baseline = entropy(np.mean(y_one_hot, axis=0))
    rce_ = 100.0 * (1.0 - np.true_divide(loss, baseline))
    rce_ = rce_.item()

    assert np.isclose(rce, rce_)


def test_dawid():
    n_labels = 2
    N = np.random.randint(low=2, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    y_pred = util.normalize(np.random.randn(N, n_labels))

    stat = btc.dawid_stat(y, y_pred)
    stat_sum = np.sum(stat, axis=0)
    Z = btc.dawid_agg(stat_sum[None, :], N)
    Z = Z.item()

    yp = np.exp(y_pred)
    Z_ = np.sum(yp[:, 1] - y) / np.sqrt(np.sum(yp[:, 0] * yp[:, 1]))
    Z_ = Z_.item()

    assert np.isclose(Z, Z_)


# TODO test (stat, agg) combo against integrated
# test all vectorize

# TODO metric_summary
# test index names, shape

if __name__ == "__main__":
    np.random.seed(845412)

    for _ in range(MC_REPEATS_LARGE):
        test_hard_loss_binary()
        test_hard_loss_decision()
        test_log_loss()
        test_brier_loss()
        test_spherical_loss()
    print("passed")
