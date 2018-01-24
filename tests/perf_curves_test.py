# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, division
from builtins import range

import numpy as np
from sklearn.metrics import auc
from sklearn.metrics.ranking import _binary_clf_curve
from sklearn.metrics.ranking import roc_curve, precision_recall_curve
import benchmark_tools.constants as constants
import benchmark_tools.perf_curves as pc

## @TODO(rdturnermtl): move MC tests into the respective test functions
# ============================================================================
# Non-vectorized versions of routines in perf_curves for testing.
# ============================================================================

def _nv_add_pseudo_points(fps, tps):
    if fps[-1] == 0:
        fps = pc.EPSILON * tps
        tps = tps.astype(fps.dtype)

    if tps[-1] == 0:
        tps = pc.EPSILON * fps
        fps = fps.astype(tps.dtype)
    return fps, tps


def _nv_binary_clf_curve(y_true, y_score, sample_weight=None):
    assert(y_true.ndim == 1 and y_true.dtype.kind == 'b')
    assert(y_score.shape == y_true.shape and np.all(np.isfinite(y_score)))
    assert(y_true.size >= 1)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score, y_true = y_score[desc_score_indices], y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    if sample_weight is None:
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        assert(fps[-1] == np.sum(~y_true) and tps[-1] == np.sum(y_true))
    else:
        assert(sample_weight.shape == y_true.shape)
        assert(np.all(np.isfinite(sample_weight)))
        # Negative weight makes no sense, 0 can violate assumps. of other funcs
        assert(np.all(sample_weight > 0))

        weight = sample_weight[desc_score_indices]
        tps = np.cumsum(y_true * weight)[threshold_idxs]
        fps = np.cumsum(weight)[threshold_idxs] - tps
        assert(np.allclose((fps[-1], tps[-1]),
                           (np.sum(weight[~y_true]), np.sum(weight[y_true]))))

    # Now put in the (0, 0) coord (y_score >= np.inf)
    assert(not (tps[0] == 0 and fps[0] == 0))
    fps, tps = np.r_[0, fps], np.r_[0, tps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    # Clean up corner case
    fps, tps = _nv_add_pseudo_points(fps, tps)
    assert(fps[-1] > 0 and tps[-1] > 0)
    assert(fps.dtype == tps.dtype)

    # Remove any decreases due to numerics
    fps = np.maximum.accumulate(fps)
    assert(np.all((np.diff(fps) >= 0.0) & (np.diff(tps) >= 0.0)))
    return fps, tps, thresholds

# ============================================================================
# Non-vectorized versions of routines in perf_curves for testing.
# ============================================================================

def _nv_roc_curve(y_true, y_score, sample_weight=None):
    fps, tps, thresholds = _nv_binary_clf_curve(y_true, y_score,
                                                sample_weight=sample_weight)
    fpr = np.true_divide(fps, fps[-1])
    tpr = np.true_divide(tps, tps[-1])
    return fpr, tpr, thresholds


def _nv_recall_precision_curve(y_true, y_score, sample_weight=None):
    fps, tps, thresholds = _nv_binary_clf_curve(y_true, y_score,
                                                sample_weight=sample_weight)
    recall = np.true_divide(tps, tps[-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.true_divide(tps, tps + fps)
    precision[0] = precision[1]
    assert(np.all(0.0 <= precision) and np.all(precision <= 1.0))
    return recall, precision, thresholds


def _nv_prg_curve(y_true, y_score, sample_weight=None):
    fps, tps, thresholds = _nv_binary_clf_curve(y_true, y_score,
                                                sample_weight=sample_weight)
    n_neg, n_pos = fps[-1], tps[-1]
    fns = n_pos - tps

    den = n_neg * tps
    with np.errstate(divide='ignore', invalid='ignore'):
        rec_gain = 1.0 - np.true_divide(n_pos * fns, den)
        prec_gain = 1.0 - np.true_divide(n_pos * fps, den)
    # interpolate backward just like in PR curve
    prec_gain[0] = prec_gain[1]
    assert(np.all(rec_gain <= 1.0) and np.all(prec_gain <= 1.0))

    # Find index to put everything in the box
    with np.errstate(invalid='ignore'):
        assert(not np.any(np.diff(rec_gain) < 0.0))
    idx = np.searchsorted(rec_gain, 0.0, side='right')
    assert(idx == np.where(rec_gain > 0.0)[0][0])
    assert(idx > 0)  # Not selecting first point
    # Bring forward most recent negative point as point at 0
    rec_gain = np.concatenate(([0.0], rec_gain[idx:]))
    prec_gain = prec_gain[idx - 1:]
    thresholds = thresholds[idx - 1:]
    return rec_gain, prec_gain, thresholds

# ============================================================================
# Now the actual tests
# ============================================================================

def test_nv_binary_clf_curve():
    N = np.random.randint(low=1, high=10)

    y_bool = np.random.rand(N) <= 0.5
    y_pred = np.random.rand(N)

    sample_weight = None
    if np.random.rand() <= 0.2:
        sample_weight = np.abs(np.random.randn(N))
    if np.random.rand() <= 0.2:
        sample_weight = 1 + np.random.multinomial(N, np.ones(N) / N)
    if np.random.rand() <= 0.2:
        sample_weight = np.maximum(np.random.multinomial(N, np.ones(N) / N),
                                   1e-6)

    fps, tps, thresholds = _nv_binary_clf_curve(y_bool, y_pred, sample_weight)
    assert(fps.shape == tps.shape and fps.shape == thresholds.shape)
    assert(np.all(np.isfinite(fps)))
    assert(np.all(np.isfinite(tps)))
    assert(np.all(np.isfinite(thresholds[1:])))
    assert(fps[0] == 0 and tps[0] == 0 and thresholds[0] == np.inf)
    if sample_weight is None:
        assert(np.abs(fps[-1] - np.sum(~y_bool)) <= 1e-8)
        assert(np.abs(tps[-1] - np.sum(y_bool)) <= 1e-8)
    else:
        assert(np.abs(fps[-1] - np.sum(sample_weight * ~y_bool)) <= 1e-8)
        assert(np.abs(tps[-1] - np.sum(sample_weight * y_bool)) <= 1e-8)
    assert(np.all((np.diff(fps) >= 0.0) & (np.diff(tps) >= 0.0)))
    assert(np.all((np.diff(fps) > 0) | (np.diff(tps) > 0)))
    assert(np.all(np.diff(thresholds) < 0.0))

    fpr, tpr, thresholds_roc = _nv_roc_curve(y_bool, y_pred, sample_weight)
    assert(fpr.shape == tpr.shape and fpr.shape == thresholds_roc.shape)
    assert(np.all(np.isfinite(fpr)))
    assert(np.all(np.isfinite(tpr)))
    assert(np.all(np.isfinite(thresholds_roc[1:])))
    assert(fpr[0] == 0.0 and tpr[0] == 0.0)
    assert(fpr[-1] == 1.0 and tpr[-1] == 1.0)
    assert(np.all((np.diff(fpr) >= 0.0) & (np.diff(tpr) >= 0.0)))
    assert(np.all((np.diff(fpr) > 0.0) | (np.diff(tpr) > 0.0)))
    assert(np.all(np.diff(thresholds_roc) < 0.0))

    rec, prec, thresholds_pr = _nv_recall_precision_curve(y_bool, y_pred,
                                                          sample_weight)
    assert(rec.shape == prec.shape and rec.shape == thresholds_pr.shape)
    assert(np.all(np.isfinite(rec)))
    assert(np.all(np.isfinite(prec)))
    assert(np.all(np.isfinite(thresholds_pr[1:])))
    assert(rec[0] == 0.0 and rec[-1] == 1.0)
    assert(len(prec) >= 2 and prec[0] == prec[1])
    b_rate = np.mean(y_bool) if sample_weight is None else \
        np.true_divide(np.sum(sample_weight * y_bool), np.sum(sample_weight))
    assert(np.max(np.abs(prec[-1] - b_rate)) <= 1e-8)
    # Note: may have repeats in PR curve
    assert(np.all(np.diff(rec) >= 0.0))
    assert(np.all(np.diff(thresholds_pr) < 0.0))

    rec_gain, prec_gain, thresholds_prg = _nv_prg_curve(y_bool, y_pred,
                                                        sample_weight)
    assert(rec_gain.shape == prec_gain.shape)
    assert(rec_gain.shape == thresholds_prg.shape)
    assert(np.all(np.isfinite(thresholds_prg[1:])))
    assert(rec_gain[0] == 0.0 and rec_gain[-1] == 1.0)
    assert(np.all(rec_gain <= 1.0) and np.all(prec_gain <= 1.0))
    assert(np.all(np.diff(rec_gain) >= 0.0))
    assert(np.allclose(prec_gain[-1], 0.0))

    if np.all(y_bool) or (not np.any(y_bool)):
        assert(np.allclose(0.5, np.trapz(fpr, tpr)))
        assert(np.allclose(np.mean(y_bool), np.sum(prec[:-1] * np.diff(rec))))
        assert(np.allclose(0.0, np.sum(prec_gain[:-1] * np.diff(rec_gain))))
        return

    fps2, tps2, thresholds2 = _binary_clf_curve(y_bool, y_pred, pos_label=True,
                                                sample_weight=sample_weight)
    assert(np.allclose(fps[1:], fps2))
    assert(np.allclose(tps[1:], tps2))
    assert(np.allclose(thresholds[1:], thresholds2))

    fpr2, tpr2, thresholds2 = roc_curve(y_bool, y_pred, pos_label=True,
                                        sample_weight=sample_weight,
                                        drop_intermediate=False)
    # sklearn inconsistent on including origin ==> need if statement
    if len(fpr) == len(fpr2):
        assert(np.allclose(fpr, fpr2))
        assert(np.allclose(tpr, tpr2))
        assert(np.allclose(thresholds_roc[1:], thresholds2[1:]))
    else:
        assert(np.allclose(fpr[1:], fpr2))
        assert(np.allclose(tpr[1:], tpr2))
        assert(np.allclose(thresholds_roc[1:], thresholds2))

    prec2, rec2, thresholds2 = \
        precision_recall_curve(y_bool, y_pred, pos_label=True,
                               sample_weight=sample_weight)
    prec2, rec2, thresholds2 = prec2[::-1], rec2[::-1], thresholds2[::-1]
    prec2[0] = prec2[1]
    err = rec[len(rec2):] - 1.0
    assert(len(err) == 0 or np.max(np.abs(err)) <= 1e-8)
    assert(np.allclose(rec[:len(rec2)], rec2))
    assert(np.allclose(prec[:len(rec2)], prec2))
    assert(np.allclose(thresholds_pr[1:len(rec2)], thresholds2))

    with np.errstate(divide='ignore', invalid='ignore'):
        rec_gain2 = (rec - b_rate) / ((1.0 - b_rate) * rec)
        prec_gain2 = (prec - b_rate) / ((1.0 - b_rate) * prec)
    idx = rec_gain2 > 0.0
    assert(np.allclose(rec_gain[1:], rec_gain2[idx]))
    assert(np.allclose(prec_gain[1:], prec_gain2[idx]))
    assert(np.allclose(thresholds_prg[1:], thresholds_pr[idx]))
    assert(np.allclose(rec_gain[0], 0.0))
    idx0 = np.where(~idx)[0][-1]
    assert(np.allclose(prec_gain[0], prec_gain2[idx0]))
    assert(np.allclose(thresholds_prg[0], thresholds_pr[idx0]))


def auc_trapz_test(x_curve, y_curve):
    auc0 = pc.auc_trapz(x_curve, y_curve)
    for ii in range(x_curve.shape[1]):
        auc1 = auc(x_curve[:, ii], y_curve[:, ii])
        assert(np.allclose(auc0[ii], auc1))

        y_ave = y_curve[1:, ii] + y_curve[:-1, ii]
        auc2 = np.sum(0.5 * np.diff(x_curve[:, ii]) * y_ave)
        assert(np.allclose(auc0[ii], auc2))


def auc_left_test(x_curve, y_curve):
    auc0 = pc.auc_left(x_curve, y_curve)
    for ii in range(x_curve.shape[1]):
        delta = np.diff(x_curve[:, ii])
        yv = y_curve[:-1, ii]

        idx = delta > 0.0
        delta, yv = delta[idx], yv[idx]

        auc1 = np.sum(delta * yv)
        assert(np.allclose(auc0[ii], auc1))


def test_binary_clf_curve():
    N = np.random.randint(low=1, high=10)
    n_boot = np.random.randint(low=1, high=10)

    y_bool = np.random.rand(N) <= 0.5
    y_pred = np.random.rand(N)

    p = np.ones(N) / N
    sample_weight = None
    if np.random.rand() <= 0.2:
        sample_weight = np.abs(np.random.randn(N, n_boot))
    if np.random.rand() <= 0.2:
        sample_weight = 1 + np.random.multinomial(N, p, size=n_boot).T
    if np.random.rand() <= 0.2:
        sample_weight = np.maximum(np.random.multinomial(N, p, size=n_boot).T,
                                   1e-6)

    fps, tps, thresholds = pc._binary_clf_curve(y_bool, y_pred, sample_weight)
    fpr, tpr, thresholds_roc = pc.roc_curve(y_bool, y_pred, sample_weight)
    rec, prec, thresholds_pr = pc.recall_precision_curve(y_bool, y_pred,
                                                         sample_weight)
    rec_gain, prec_gain, thresholds_prg = pc.prg_curve(y_bool, y_pred,
                                                       sample_weight)

    auc_trapz_test(fpr, tpr)
    auc_left_test(rec, prec)
    auc_left_test(rec_gain, prec_gain)

    if sample_weight is None:
        fps2, tps2, thresholds2 = \
            _nv_binary_clf_curve(y_bool, y_pred, np.ones(N, dtype=int))
        assert(np.all(fps2 == fps[:, 0]))
        assert(np.all(tps2 == tps[:, 0]))
        assert(np.all(thresholds2 == thresholds))

        fps2, tps2, thresholds2 = _nv_binary_clf_curve(y_bool, y_pred)
        assert(np.all(fps2 == fps[:, 0]))
        assert(np.all(tps2 == tps[:, 0]))
        assert(np.all(thresholds2 == thresholds))

        fpr2, tpr2, thresholds_roc2 = _nv_roc_curve(y_bool, y_pred)
        assert(np.allclose(fpr2, fpr[:, 0]))
        assert(np.allclose(tpr2, tpr[:, 0]))
        assert(np.allclose(thresholds_roc2, thresholds))

        rec2, prec2, thresholds_pr2 = \
            _nv_recall_precision_curve(y_bool, y_pred)
        assert(np.allclose(rec2, rec[:, 0]))
        assert(np.allclose(prec2, prec[:, 0]))
        assert(np.allclose(thresholds_pr2, thresholds_pr))

        rec_gain2, prec_gain2, thresholds_prg2 = _nv_prg_curve(y_bool, y_pred)
        assert(np.all(rec_gain[:-len(rec_gain2), 0] == 0.0))
        assert(np.allclose(rec_gain2, rec_gain[-len(rec_gain2):, 0]))
        assert(np.allclose(prec_gain2, prec_gain[-len(rec_gain2):, 0]))
        assert(np.allclose(thresholds_prg2, thresholds_prg[-len(rec_gain2):]))
        return

    for ii in range(n_boot):
        weight_curr = sample_weight[:, ii]

        fpr2, tpr2, thresholds_roc2 = \
            _nv_roc_curve(y_bool, y_pred, weight_curr)
        assert(np.allclose(fpr2, fpr[:, ii]))
        assert(np.allclose(tpr2, tpr[:, ii]))
        assert(np.allclose(thresholds_roc2, thresholds))

        rec2, prec2, thresholds_pr2 = \
            _nv_recall_precision_curve(y_bool, y_pred, weight_curr)
        assert(np.allclose(rec2, rec[:, ii]))
        assert(np.allclose(prec2, prec[:, ii]))
        assert(np.allclose(thresholds_pr2, thresholds_pr))

        rec_gain2, prec_gain2, thresholds_prg2 = \
            _nv_prg_curve(y_bool, y_pred, weight_curr)
        assert(np.all(rec_gain[:-len(rec_gain2), ii] == 0.0))
        assert(np.allclose(rec_gain2, rec_gain[-len(rec_gain2):, ii]))
        assert(np.allclose(prec_gain2, prec_gain[-len(rec_gain2):, ii]))
        assert(np.allclose(thresholds_prg2, thresholds_prg[-len(rec_gain2):]))

if __name__ == '__main__':
    np.random.seed(89254)

    for rr in range(constants.MC_REPEATS_LARGE):
        test_nv_binary_clf_curve()
        test_binary_clf_curve()
    print('passed')
