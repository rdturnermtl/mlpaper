# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function
from builtins import range
import numpy as np
from benchmark_tools.classification import curve_boot
import benchmark_tools.constants as constants
import benchmark_tools.benchmark_tools as bt
import benchmark_tools.perf_curves as pc

np.random.seed(3563)

confidence = 0.95
N = 200
N_big = constants.MC_REPEATS_LARGE
p = 0.3
runs = constants.MC_REPEATS_1K
w = 0.1

auc = np.zeros((runs, 3))
valid = np.zeros((runs, 3))
pval = np.zeros((runs, 3))
for rr in range(runs):
    y_score = np.random.rand(N_big, 2)
    y_true = np.random.rand(N_big) <= w * y_score[:, 0] + (1-w) * y_score[:, 1]

    x_curve, y_curve, _ = pc.roc_curve(y_true[N:], y_score[N:, 1])
    ref = pc.auc_trapz(x_curve, y_curve)
    summary, _ = curve_boot(y_true[:N], y_score[:N, :],
                            default_summary_ref=ref,
                            curve_f=pc.roc_curve, summary_f=pc.auc_trapz,
                            n_boot=1000, confidence=confidence)
    auc[rr, 0], EB, pval[rr, 0] = summary
    valid[rr, 0] = np.abs(ref - auc[rr, 0]) <= EB

    x_curve, y_curve, _ = pc.recall_precision_curve(y_true[N:], y_score[N:, 1])
    ref = pc.auc_left(x_curve, y_curve)
    summary, _ = curve_boot(y_true[:N], y_score[:N, :],
                            default_summary_ref=ref,
                            curve_f=pc.recall_precision_curve,
                            summary_f=pc.auc_left,
                            n_boot=1000, confidence=confidence)
    auc[rr, 1], EB, pval[rr, 1] = summary
    valid[rr, 1] = np.abs(ref - auc[rr, 1]) <= EB

    x_curve, y_curve, _ = pc.prg_curve(y_true[N:], y_score[N:, 1])
    ref = pc.auc_left(x_curve, y_curve)
    summary, _ = curve_boot(y_true[:N], y_score[:N, :],
                            default_summary_ref=ref,
                            curve_f=pc.prg_curve, summary_f=pc.auc_left,
                            n_boot=1000, confidence=confidence)
    auc[rr, 2], EB, pval[rr, 2] = summary
    valid[rr, 2] = np.abs(ref - auc[rr, 2]) <= EB

# TODO: This is not a test.
print('against N-big')
print('P EB valid', np.mean(valid, axis=0))
print('mean AUC', np.mean(auc, axis=0))
print('P p-val sig', np.mean(pval <= 0.05, axis=0))

auc = np.zeros((runs, 3))
valid = np.zeros((runs, 3))
pval = np.zeros((runs, 3))
for rr in range(runs):
    y_true = np.random.rand(N) <= p
    y_score = np.random.randn(N, 2)

    summary, _ = curve_boot(y_true, y_score, default_summary_ref=0.5,
                            curve_f=pc.roc_curve, summary_f=pc.auc_trapz,
                            n_boot=1000, confidence=confidence)
    auc[rr, 0], EB, pval[rr, 0] = summary
    valid[rr, 0] = np.abs(0.5 - auc[rr, 0]) <= EB

    summary, _ = curve_boot(y_true, y_score, default_summary_ref=p,
                            curve_f=pc.recall_precision_curve,
                            summary_f=pc.auc_left,
                            n_boot=1000, confidence=confidence)
    auc[rr, 1], EB, pval[rr, 1] = summary
    valid[rr, 1] = np.abs(p - auc[rr, 1]) <= EB

    summary, _ = curve_boot(y_true, y_score, default_summary_ref=0.0,
                            curve_f=pc.prg_curve, summary_f=pc.auc_left,
                            n_boot=1000, confidence=confidence)
    auc[rr, 2], EB, pval[rr, 2] = summary
    valid[rr, 2] = np.abs(auc[rr, 2]) <= EB

# TODO: this is not a test
print('just noise')
print('P EB valid', np.mean(valid, axis=0))
print('mean AUC', np.mean(auc, axis=0))
print('P p-val sig', np.mean(pval <= 0.05, axis=0))
