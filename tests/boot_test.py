# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, division
from builtins import range
import numpy as np
import scipy.stats as ss
from benchmark_tools.classification import curve_boot, DEFAULT_NGRID
import benchmark_tools.constants as cc
import benchmark_tools.perf_curves as pc
from benchmark_tools.util import area

FPR = 1e-2  # TODO move to constants file

# TODO make 2nd test func that tests big against small


def fail_check_stat(fail, runs, expect_p_fail, fpr):
    pvals_2side = [ss.binom_test(ff, runs, expect_p_fail) for ff in fail]
    pvals_1side = \
        [ss.binom_test(ff, runs, expect_p_fail, alternative='greater')
         for ff in fail]
    # Note that we are not going multiple comparison correction between the
    # two sided and one sided tests.
    print(pvals_2side)
    assert(np.min(pvals_2side) >= fpr / len(pvals_2side))
    print(pvals_1side)
    assert(np.min(pvals_1side) >= fpr / len(pvals_1side))


def test_boot(runs=100):
    N = 201
    confidence = 0.95

    # Drawing more seeds than we need to be safe
    seeds = np.nditer(np.random.randint(low=0, high=int(1e6), size=runs * 5))

    def run_trial(y_true, y_score, y_score_ref, true_value, curve_f, seed,
                  x_grid=None):
        curve, _ = curve_f(y_true, y_score[:, 1])
        auc, = area(*curve)
        curve, _ = curve_f(y_true, y_score_ref[:, 1])
        auc_ref, = area(*curve)

        np.random.seed(seed)
        (auc_, EB, pval), curve = \
            curve_boot(y_true, y_score, ref=true_value, curve_f=curve_f,
                       confidence=confidence, x_grid=x_grid)
        assert(auc_ == auc)
        fail_EB = np.abs(auc - true_value) > EB
        # Could also test distn with 1-sided KS test but this easier for now
        fail_P = pval < 1.0 - confidence
        fail_curve = ((curve[cc.YGRID].values < curve[cc.LB].values) |
                      (curve[cc.UB].values < curve[cc.YGRID].values))
        assert((x_grid is None) or np.all(curve[cc.XGRID].values == x_grid))

        np.random.seed(seed)
        (auc_, EB_, pval), curve_ = \
            curve_boot(y_true, y_score, ref=y_score_ref, curve_f=curve_f,
                       confidence=confidence, pairwise_CI=False, x_grid=x_grid)
        assert(auc_ == auc)
        assert(EB_ == EB)
        # Could also test distn with 1-sided KS test but this easier for now
        fail_P2 = pval < 1.0 - confidence
        assert(np.all(curve_.values == curve.values))

        np.random.seed(seed)
        (auc_, EB, pval_), curve_ = \
            curve_boot(y_true, y_score, ref=y_score_ref, curve_f=curve_f,
                       confidence=confidence, pairwise_CI=True, x_grid=x_grid)
        assert(auc_ == auc)
        fail_EB2 = np.abs(auc - auc_ref) > EB
        # Could also test distn with 1-sided KS test but this easier for now
        assert(pval_ == pval)
        assert(np.all(curve_.values == curve.values))

        return fail_EB, fail_P, fail_EB2, fail_P2, fail_curve

    fail = [0] * 12
    fail_curve_roc = np.zeros(DEFAULT_NGRID, dtype=int)
    fail_curve_ap = np.zeros(DEFAULT_NGRID, dtype=int)
    fail_curve_prg = np.zeros(DEFAULT_NGRID, dtype=int)
    for ii in range(runs):
        mu = np.random.randn(2)
        S = np.random.randn(2, 2)
        S = np.dot(S, S.T)
        p = np.random.rand()

        x_grid = None  # Can also randomize this per trial too

        y_true = np.random.rand(N) <= p
        y_score = np.random.multivariate_normal(mu, S, size=N)
        if np.random.randn() <= 0.5:  # resample to test dupes
            idx = np.random.choice(N, size=N, replace=True)
            y_score = y_score[idx, :]
        y_score, y_score_ref = y_score.T
        y_score = np.stack((np.zeros(N), y_score), axis=1)
        y_score_ref = np.stack((np.zeros(N), y_score_ref), axis=1)

        fail_EB, fail_P, fail_EB2, fail_P2, fail_curve = \
            run_trial(y_true, y_score, y_score_ref,
                      0.5, pc.roc_curve, seeds.next(), x_grid)
        fail[0] += fail_EB
        fail[1] += fail_P
        fail[2] += fail_EB2
        fail[3] += fail_P2
        fail_curve_roc += fail_curve

        fail_EB, fail_P, fail_EB2, fail_P2, fail_curve = \
            run_trial(y_true, y_score, y_score_ref,
                      p, pc.recall_precision_curve, seeds.next(), x_grid)
        fail[4] += fail_EB
        fail[5] += fail_P
        fail[6] += fail_EB2
        fail[7] += fail_P2
        fail_curve_ap += fail_curve

        # TODO switch to prg, and true_value to zero
        fail_EB, fail_P, fail_EB2, fail_P2, fail_curve = \
            run_trial(y_true, y_score, y_score_ref,
                      0.5, pc.roc_curve, seeds.next(), x_grid)
        fail[8] += fail_EB
        fail[9] += fail_P
        fail[10] += fail_EB2
        fail[11] += fail_P2
        fail_curve_prg += fail_curve
    sub_FPR = FPR / 4.0
    expect_p_fail = 1.0 - confidence
    fail_check_stat(fail, runs, expect_p_fail, sub_FPR)
    print('ROC curve')
    fail_check_stat(fail_curve_roc, runs, expect_p_fail, sub_FPR)
    print('RP curve')
    fail_check_stat(fail_curve_ap, runs, expect_p_fail, sub_FPR)
    print('PRG curve')
    fail_check_stat(fail_curve_prg, runs, expect_p_fail, sub_FPR)

if __name__ == '__main__':
    np.random.seed(56456)

    test_boot()
    print('passed')
