# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, division
from builtins import range
import numpy as np
import scipy.stats as ss
import benchmark_tools.perf_curves as pc
from benchmark_tools.util import area


def test_curve():
    N = 6
    p = 0.3
    resample = True
    runs = 1000

    auc = np.zeros(runs)
    ap = np.zeros(runs)
    auprg = np.zeros(runs)
    for rr in range(runs):
        y_true = np.random.rand(N) <= p
        y_score = np.random.randn(N)
        if resample:
            y_score = np.random.choice(y_score, size=N, replace=True)

        curve, _ = pc.roc_curve(y_true, y_score)
        auc[rr], = area(*curve)
    
        curve, _ = pc.recall_precision_curve(y_true, y_score)
        ap[rr], = area(*curve)

        curve, _ = pc.prg_curve(y_true, y_score)
        auprg[rr], = area(*curve)
    print(auprg)
    _, pvals_auc = ss.ttest_1samp(auc, 0.5)
    _, pvals_ap = ss.ttest_1samp(ap, p)
    _, pvals_auprg = ss.ttest_1samp(auprg, 0.0)
    print([pvals_auc, pvals_ap, pvals_auprg])

if __name__ == '__main__':
    np.random.seed(7852)

    for rr in xrange(10):
        print('---')
        test_curve()
