# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, division
from builtins import range
import numpy as np
import scipy.stats as ss
from benchmark_tools.classification import curve_boot, DEFAULT_NGRID
import benchmark_tools.constants as cc
import benchmark_tools.perf_curves as pc
from benchmark_tools.util import area

# @TODO(rdturnermtl): this needs to be made into proper tests.
# np.random.seed(3563)

FPR = 1e-2


def fail_check_stat(fail, runs, expect_p_fail, fpr):
    pvals_2side = [ss.binom_test(ff, runs, expect_p_fail) for ff in fail]
    pvals_1side = \
        [ss.binom_test(ff, runs, expect_p_fail, alternative='greater')
         for ff in fail]
    print(pvals_2side)
    assert(np.min(pvals_2side) >= fpr / len(pvals_2side))
    print(pvals_1side)
    assert(np.min(pvals_1side) >= fpr / len(pvals_1side))


def test_boot(runs=100):
    N = 201
    confidence = 0.95

    # Doesn't furt to draw some extra seeds
    # TODO accept as input
    seeds = np.nditer(np.random.randint(low=0, high=int(1e6), size=runs * 5))

    def run_trial(y_true, y_score, y_score_ref, true_value, curve_f, seed):
        curve, _ = curve_f(y_true, y_score[:, 1])
        auc_, = area(*curve)
        curve, _ = curve_f(y_true, y_score_ref[:, 1])
        auc_ref, = area(*curve)

        np.random.seed(seed)
        (auc, EB, pval), curve = \
            curve_boot(y_true, y_score, ref=true_value, curve_f=curve_f,
                       confidence=confidence)
        assert(auc_ == auc)
        fail_EB = np.abs(auc - true_value) > EB
        # Could also test distn with 1-sided KS test but this easier for now
        fail_P = pval < 1.0 - confidence
        fail_curve = ((curve[cc.YGRID].values < curve[cc.LB].values) |
                      (curve[cc.UB].values < curve[cc.YGRID].values))

        np.random.seed(seed)
        (auc, EB_, pval), curve_ = \
            curve_boot(y_true, y_score, ref=y_score_ref, curve_f=curve_f,
                       confidence=confidence, pairwise_CI=False)
        assert(auc_ == auc)
        assert(EB_ == EB)
        # Could also test distn with 1-sided KS test but this easier for now
        fail_P2 = pval < 1.0 - confidence
        assert(np.all(curve_.values == curve.values))

        np.random.seed(seed)
        (auc, EB, pval_), curve = \
            curve_boot(y_true, y_score, ref=y_score_ref, curve_f=curve_f,
                       confidence=confidence, pairwise_CI=True)
        assert(auc_ == auc)
        fail_EB2 = np.abs(auc - auc_ref) > EB
        # Could also test distn with 1-sided KS test but this easier for now
        assert(pval_ == pval)
        assert(np.all(curve_.values == curve.values))

        return fail_EB, fail_P, fail_EB2, fail_P2, fail_curve

    fail = [0] * 8
    fail_curve_roc = np.zeros(DEFAULT_NGRID, dtype=int)
    fail_curve_ap = np.zeros(DEFAULT_NGRID, dtype=int)
    for ii in range(runs):
        mu = np.random.randn(2)
        S = np.random.randn(2, 2)
        S = np.dot(S, S.T)
        p = np.random.rand()

        y_true = np.random.rand(N) <= p
        # TODO also consider resample
        y_score, y_score_ref = np.random.multivariate_normal(mu, S, size=N).T
        y_score = np.stack((np.zeros_like(y_score), y_score), axis=1)
        y_score_ref = np.stack((np.zeros_like(y_score_ref), y_score_ref), axis=1)
        print(y_true.shape)
        print(y_score.shape)
        print(y_score_ref.shape)

        fail_EB, fail_P, fail_EB2, fail_P2, fail_curve = \
            run_trial(y_true, y_score, y_score_ref,
                      0.5, pc.roc_curve, seeds.next())
        fail[0] += fail_EB
        fail[1] += fail_P
        fail[2] += fail_EB2
        fail[3] += fail_P2
        fail_curve_roc += fail_curve

        fail_EB, fail_P, fail_EB2, fail_P2, fail_curve = \
            run_trial(y_true, y_score, y_score_ref,
                      p, pc.recall_precision_curve, seeds.next())
        fail[4] += fail_EB
        fail[5] += fail_P
        fail[6] += fail_EB2
        fail[7] += fail_P2
        fail_curve_ap += fail_curve

    sub_FPR = FPR / 3.0
    expect_p_fail = 1.0 - confidence
    fail_check_stat(fail, runs, expect_p_fail, sub_FPR)
    print('ROC')
    fail_check_stat(fail_curve_roc, runs, expect_p_fail, sub_FPR)
    print('AP')
    fail_check_stat(fail_curve_ap, runs, expect_p_fail, sub_FPR)

if __name__ == '__main__':
    np.random.seed(56456)

    test_boot()
    print('passed')
