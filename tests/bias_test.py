# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, division
from builtins import range
import numpy as np
import scipy.stats as ss
import benchmark_tools.perf_curves as pc
from benchmark_tools.util import area

FPR = 1e-3


def inner_test_curve(runs):
    N = 6
    p = 0.3
    resample = True

    # AUPRG sucks and can sometimes be -inf so it does not make sense to test
    # the bias when it can be non-finite.
    auc = np.zeros(runs)
    ap = np.zeros(runs)
    for rr in range(runs):
        y_true = np.random.rand(N) <= p
        y_score = np.random.randn(N)
        if resample:
            y_score = np.random.choice(y_score, size=N, replace=True)

        curve, _ = pc.roc_curve(y_true, y_score)
        auc[rr], = area(*curve)
    
        curve, _ = pc.recall_precision_curve(y_true, y_score)
        ap[rr], = area(*curve)
    _, pvals_auc = ss.ttest_1samp(auc, 0.5)
    _, pvals_ap = ss.ttest_1samp(ap, p)
    return pvals_auc, pvals_ap


def loop_test(test_f):
    runs = 100
    M1 = []
    for rr in xrange(10):
        pvals = test_f(runs)
        M1.append(pvals)
    M1 = np.asarray(M1)
    pvals = [ss.combine_pvalues(M1[:, ii])[1] for ii in range(M1.shape[1])]

    print(pvals)
    assert(np.min(pvals) >= FPR / len(pvals))


def test_curve():
    loop_test(inner_test_curve)

if __name__ == '__main__':
    np.random.seed(7852)

    test_curve()
    print('passed')
