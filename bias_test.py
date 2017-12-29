# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function
from builtins import range
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import roc_auc_score, average_precision_score
import perf_curves as pc

import prg

np.random.seed(3563)

N = 6
p = 0.3
runs = int(1e3)
resample = True

auc = np.zeros((runs, 7))
ap = np.zeros((runs, 7))
auprg = np.zeros((runs, 7))
for rr in range(runs):
    y_true = np.random.rand(N) <= p
    y_score = np.random.randn(N)

    if resample:
        y_score = np.random.choice(y_score, size=N, replace=True)

    one_sided = np.sum(y_true) in (0, N)

    if one_sided:
        auc[rr, :] = 0.5
    else:
        fpr, tpr, _ = pc._nv_roc_curve(y_true, y_score)

        auc[rr, 0] = np.trapz(tpr, fpr)
        auc[rr, 1] = np.sum(tpr[1:] * np.diff(fpr))
        auc[rr, 2] = np.sum(tpr[:-1] * np.diff(fpr))

        idx = np.r_[True, np.diff(tpr) >= 1e-6]
        idx[-1] = True
        tpr2 = tpr[idx]
        fpr2 = fpr[idx]

        auc[rr, 3] = np.trapz(tpr2, fpr2)
        auc[rr, 4] = np.sum(tpr2[1:] * np.diff(fpr2))
        auc[rr, 5] = np.sum(tpr2[:-1] * np.diff(fpr2))

        auc[rr, 6] = roc_auc_score(y_true, y_score)

    rec, prec, _ = pc._nv_recall_precision_curve(y_true, y_score)
    ap[rr, 0] = np.trapz(prec, rec)
    ap[rr, 1] = np.sum(prec[1:] * np.diff(rec))
    ap[rr, 2] = np.sum(prec[:-1] * np.diff(rec))

    prec[0] = prec[1]
    ap[rr, 3] = np.trapz(prec, rec)
    ap[rr, 4] = np.sum(prec[1:] * np.diff(rec))
    ap[rr, 5] = np.sum(prec[:-1] * np.diff(rec))

    if one_sided:
        ap[rr, :] = np.mean(y_true)
    else:
        ap[rr, 6] = average_precision_score(y_true, y_score)

    if one_sided:
        auprg[rr, :] = 0.0
    else:
        rec_gain, prec_gain, _ = pc._nv_prg_curve(y_true, y_score)

        auprg[rr, 0] = np.trapz(prec_gain, rec_gain)
        auprg[rr, 1] = np.sum(prec_gain[1:] * np.diff(rec_gain))
        auprg[rr, 2] = np.sum(prec_gain[:-1] * np.diff(rec_gain))

        prec_gain[0] = prec_gain[1]
        auprg[rr, 3] = np.trapz(prec_gain, rec_gain)
        auprg[rr, 4] = np.sum(prec_gain[1:] * np.diff(rec_gain))
        auprg[rr, 5] = np.sum(prec_gain[:-1] * np.diff(rec_gain))

        auprg[rr, 6] = prg.calc_auprg(prg.create_prg_curve(y_true, y_score))
_, pvals_auc = ss.ttest_1samp(auc, 0.5, axis=0)
_, pvals_ap = ss.ttest_1samp(ap, p, axis=0)
_, pvals_auprg = ss.ttest_1samp(auprg, 0.0, axis=0)

cols = ['trap', 'right', 'left',
        'trap-mod', 'right-mod', 'left-mod',
        'official']
pval_df = pd.DataFrame(data=np.stack((pvals_auc, pvals_ap, pvals_auprg)),
                       index=['AUC', 'AP', 'AUPRG'], columns=cols)

print('pvals:', pval_df.to_string())

biased = pval_df <= 0.05
print('biased:',biased.to_string())
