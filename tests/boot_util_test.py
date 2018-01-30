import numpy as np
import scipy.stats as ss
import benchmark_tools.boot_util as bu


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


def test_basic():
    mu = np.random.randn()
    stdev = np.abs(np.random.randn())

    N = 201
    B = 1000
    runs = 100
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

    fail = [0] * 6
    for ii in range(runs):
        x = mu + stdev * np.random.randn(N)

        fail_perc, fail_basic, fail_EB = run_trial(x, np.mean, mu)
        fail[0] += fail_perc
        fail[1] += fail_basic
        fail[2] += fail_EB

        fail_perc, fail_basic, fail_EB = run_trial(x, np.median, mu)
        fail[3] += fail_perc
        fail[4] += fail_basic
        fail[5] += fail_EB
    print(fail)
    pval = [ss.binom_test(ff, runs, 1.0 - confidence) for ff in fail]
    print(pval)
    _, pval_agg = ss.combine_pvalues(pval)
    print(pval_agg)


def test_paired():
    mu = np.random.randn(2)
    S = np.random.randn(2, 2)
    S = np.dot(S, S.T)

    N = 201
    B = 1000
    runs = 100
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

    fail = [0] * 6
    for ii in range(runs):
        x = np.random.multivariate_normal(mu, S, size=N)

        fail_perc, fail_basic, fail_EB = run_trial(x, np.mean, mu[1] - mu[0])
        fail[0] += fail_perc
        fail[1] += fail_basic
        fail[2] += fail_EB

        fail_perc, fail_basic, fail_EB = run_trial(x, np.median, mu[1] - mu[0])
        fail[3] += fail_perc
        fail[4] += fail_basic
        fail[5] += fail_EB
    print(fail)
    pval = [ss.binom_test(ff, runs, 1.0 - confidence) for ff in fail]
    print(pval)
    _, pval_agg = ss.combine_pvalues(pval)
    print(pval_agg)

if __name__ == '__main__':
    np.random.seed(24233)

    test_basic()
    test_paired()
