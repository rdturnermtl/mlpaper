# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, division
import numpy as np
import scipy.stats as ss
import benchmark_tools.boot_util as bu

FPR = 1e-3


def get_boot_estimate(x, estimate_f):
    theta = estimate_f(np.random.choice(x, size=x.size, replace=True))
    return theta


def get_boot_estimate_vec(x, estimate_f):
    idx = np.random.choice(range(x.shape[0]), size=x.shape[0], replace=True)
    x_bs = x[idx, :]
    est = estimate_f(x_bs, axis=0)
    return est


def test_confidence_to_percentiles():
    confidence = np.random.rand()

    LB, UB = bu.confidence_to_percentiles(confidence)
    assert(np.allclose(confidence * 100, UB - LB))


def inner_test_boot(runs=100):
    mu = np.random.randn()
    stdev = np.abs(np.random.randn())

    N = 201
    B = 1000
    confidence = 0.95

    def run_trial(x, est_f, true_value):
        original_est = est_f(x)
        boot_estimates = [get_boot_estimate(x, est_f) for _ in range(B)]
        boot_estimates = np.asarray(boot_estimates)

        LBp, UBp = bu.percentile(boot_estimates, confidence=confidence)
        fail_perc = (true_value < LBp) or (UBp < true_value)

        LB, UB = bu.basic(boot_estimates, original_est, confidence=confidence)
        fail_basic = (true_value < LB) or (UB < true_value)

        EB = bu.error_bar(boot_estimates, original_est, confidence=confidence)
        fail_EB = np.abs(original_est - true_value) > EB

        return fail_perc, fail_basic, fail_EB

    fail = [0] * 8
    for ii in range(runs):
        x = mu + stdev * np.random.randn(N)

        fail_perc, fail_basic, fail_EB = run_trial(x, np.mean, mu)
        fail[0] += fail_perc
        fail[1] += fail_basic
        fail[2] += fail_EB

        fail_perc, fail_basic, fail_EB = run_trial(x, np.std, stdev)
        fail[3] += fail_perc
        fail[4] += fail_basic
        fail[5] += fail_EB

        # Basic boot CI seems to suck and won't work with median
        fail_perc, _, fail_EB = run_trial(x, np.median, mu)
        fail[6] += fail_perc
        fail[7] += fail_EB
    pvals_2side = [ss.binom_test(ff, runs, 1.0 - confidence) for ff in fail]
    pvals_1side = \
        [ss.binom_test(ff, runs, 1.0 - confidence, alternative='greater')
         for ff in fail]
    return pvals_2side, pvals_1side


def inner_test_paired_boot(runs=100):
    mu = np.random.randn(2)
    S = np.random.randn(2, 2)
    S = np.dot(S, S.T)

    N = 201
    B = 1000
    confidence = 0.95

    def run_trial(x, est_f, true_value):
        original_ref, original = est_f(x, axis=0)
        original_delta = original - original_ref

        boot = [get_boot_estimate_vec(x, est_f) for _ in range(B)]
        boot = np.asarray(boot)
        assert(boot.shape == (B, 2))
        delta = boot[:, 1] - boot[:, 0]

        LB, UB = bu.percentile(delta, confidence=confidence)
        fail_perc = (true_value < LB) or (UB < true_value)
        LB, UB = bu.basic(delta, original_delta, confidence=confidence)
        fail_basic = (true_value < LB) or (UB < true_value)
        EB = bu.error_bar(delta, original_delta, confidence=confidence)
        fail_EB = np.abs(original_delta - true_value) > EB
        return fail_perc, fail_basic, fail_EB

    fail = [0] * 8
    for ii in range(runs):
        x = np.random.multivariate_normal(mu, S, size=N)

        fail_perc, fail_basic, fail_EB = run_trial(x, np.mean, mu[1] - mu[0])
        fail[0] += fail_perc
        fail[1] += fail_basic
        fail[2] += fail_EB

        stdev_delta = np.sqrt(S[1, 1]) - np.sqrt(S[0, 0])
        fail_perc, fail_basic, fail_EB = run_trial(x, np.std, stdev_delta)
        fail[3] += fail_perc
        fail[4] += fail_basic
        fail[5] += fail_EB

        # Basic boot CI seems to suck and won't work with median
        fail_perc, _, fail_EB = run_trial(x, np.median, mu[1] - mu[0])
        fail[6] += fail_perc
        fail[7] += fail_EB
    pvals_2side = [ss.binom_test(ff, runs, 1.0 - confidence) for ff in fail]
    pvals_1side = \
        [ss.binom_test(ff, runs, 1.0 - confidence, alternative='greater')
         for ff in fail]
    return pvals_2side, pvals_1side


def loop_test(test_f):
    runs = 100
    M2 = []
    M1 = []
    for rr in xrange(10):
        pvals_2side, pvals_1side = test_f(runs)
        M2.append(pvals_2side)
        M1.append(pvals_1side)
    M2 = np.asarray(M2)
    M1 = np.asarray(M1)

    pvals_2side = [ss.combine_pvalues(M2[:, ii])[1] for ii in range(M2.shape[1])]
    pvals_1side = [ss.combine_pvalues(M1[:, ii])[1] for ii in range(M1.shape[1])]

    print(pvals_2side)
    assert(np.min(pvals_2side) >= FPR / len(pvals_2side))
    print(pvals_1side)
    assert(np.min(pvals_1side) >= FPR / len(pvals_1side))


def test_boot():
    loop_test(inner_test_boot)


def test_paired_boot():
    loop_test(inner_test_paired_boot)


if __name__ == '__main__':
    np.random.seed(24233)

    for rr in xrange(10):
        test_confidence_to_percentiles()
    test_boot()
    test_paired_boot()
    print('passed')
