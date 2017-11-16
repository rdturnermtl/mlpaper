# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics.ranking import _binary_clf_curve
from sklearn.metrics.ranking import roc_curve, precision_recall_curve
import perf_curves as pc


def eval_step_func_test():
    N = np.random.randint(low=0, high=10)

    xp = np.sort(np.random.choice(np.random.randn(N + 1),
                                  size=N, replace=True))
    yp = np.random.randn(N)
    D = {xp[ii]: yp[ii] for ii in xrange(N)}

    xp2, yp2 = pc.make_into_step(xp, yp)
    assert(xp2.shape == yp2.shape)
    assert(np.all(np.unique(xp) == xp2))
    D2 = {xp2[ii]: yp2[ii] for ii in xrange(len(xp2))}
    assert(D == D2)


def nv_binary_clf_curve_test():
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

    fps, tps, thresholds = pc._nv_binary_clf_curve(y_bool, y_pred,
                                                   sample_weight)
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

    fpr, tpr, thresholds_roc = pc._nv_roc_curve(y_bool, y_pred, sample_weight)
    assert(fpr.shape == tpr.shape and fpr.shape == thresholds_roc.shape)
    assert(np.all(np.isfinite(fpr)))
    assert(np.all(np.isfinite(tpr)))
    assert(np.all(np.isfinite(thresholds_roc[1:])))
    assert(fpr[0] == 0.0 and tpr[0] == 0.0)
    assert(fpr[-1] == 1.0 and tpr[-1] == 1.0)
    assert(np.all((np.diff(fpr) >= 0.0) & (np.diff(tpr) >= 0.0)))
    assert(np.all((np.diff(fpr) > 0.0) | (np.diff(tpr) > 0.0)))
    assert(np.all(np.diff(thresholds_roc) < 0.0))

    rec, prec, thresholds_pr = pc._nv_recall_precision_curve(y_bool, y_pred,
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

    rec_gain, prec_gain, thresholds_prg = pc._nv_prg_curve(y_bool, y_pred,
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
    for ii in xrange(x_curve.shape[1]):
        auc1 = auc(x_curve[:, ii], y_curve[:, ii])
        assert(np.allclose(auc0[ii], auc1))

        y_ave = y_curve[1:, ii] + y_curve[:-1, ii]
        auc2 = np.sum(0.5 * np.diff(x_curve[:, ii]) * y_ave)
        assert(np.allclose(auc0[ii], auc2))


def auc_left_test(x_curve, y_curve):
    auc0 = pc.auc_left(x_curve, y_curve)
    for ii in xrange(x_curve.shape[1]):
        delta = np.diff(x_curve[:, ii])
        yv = y_curve[:-1, ii]

        idx = delta > 0.0
        delta, yv = delta[idx], yv[idx]

        auc1 = np.sum(delta * yv)
        assert(np.allclose(auc0[ii], auc1))


def binary_clf_curve_test():
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
            pc._nv_binary_clf_curve(y_bool, y_pred, np.ones(N, dtype=int))
        assert(np.all(fps2 == fps[:, 0]))
        assert(np.all(tps2 == tps[:, 0]))
        assert(np.all(thresholds2 == thresholds))

        fps2, tps2, thresholds2 = pc._nv_binary_clf_curve(y_bool, y_pred)
        assert(np.all(fps2 == fps[:, 0]))
        assert(np.all(tps2 == tps[:, 0]))
        assert(np.all(thresholds2 == thresholds))

        fpr2, tpr2, thresholds_roc2 = pc._nv_roc_curve(y_bool, y_pred)
        assert(np.allclose(fpr2, fpr[:, 0]))
        assert(np.allclose(tpr2, tpr[:, 0]))
        assert(np.allclose(thresholds_roc2, thresholds))

        rec2, prec2, thresholds_pr2 = \
            pc._nv_recall_precision_curve(y_bool, y_pred)
        assert(np.allclose(rec2, rec[:, 0]))
        assert(np.allclose(prec2, prec[:, 0]))
        assert(np.allclose(thresholds_pr2, thresholds_pr))

        rec_gain2, prec_gain2, thresholds_prg2 = \
            pc._nv_prg_curve(y_bool, y_pred)
        assert(np.all(rec_gain[:-len(rec_gain2), 0] == 0.0))
        assert(np.allclose(rec_gain2, rec_gain[-len(rec_gain2):, 0]))
        assert(np.allclose(prec_gain2, prec_gain[-len(rec_gain2):, 0]))
        assert(np.allclose(thresholds_prg2, thresholds_prg[-len(rec_gain2):]))
        return

    for ii in xrange(n_boot):
        weight_curr = sample_weight[:, ii]

        fpr2, tpr2, thresholds_roc2 = \
            pc._nv_roc_curve(y_bool, y_pred, weight_curr)
        assert(np.allclose(fpr2, fpr[:, ii]))
        assert(np.allclose(tpr2, tpr[:, ii]))
        assert(np.allclose(thresholds_roc2, thresholds))

        rec2, prec2, thresholds_pr2 = \
            pc._nv_recall_precision_curve(y_bool, y_pred, weight_curr)
        assert(np.allclose(rec2, rec[:, ii]))
        assert(np.allclose(prec2, prec[:, ii]))
        assert(np.allclose(thresholds_pr2, thresholds_pr))

        rec_gain2, prec_gain2, thresholds_prg2 = \
            pc._nv_prg_curve(y_bool, y_pred, weight_curr)
        assert(np.all(rec_gain[:-len(rec_gain2), ii] == 0.0))
        assert(np.allclose(rec_gain2, rec_gain[-len(rec_gain2):, ii]))
        assert(np.allclose(prec_gain2, prec_gain[-len(rec_gain2):, ii]))
        assert(np.allclose(thresholds_prg2, thresholds_prg[-len(rec_gain2):]))

np.random.seed(89254)

print 'start'
for rr in xrange(100000):
    eval_step_func_test()
    nv_binary_clf_curve_test()
    binary_clf_curve_test()
print 'done'
