# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import division, print_function

from builtins import range
from string import ascii_letters

import numpy as np
import pandas as pd
import scipy.stats as ss

import mlpaper.constants as cc
import mlpaper.mlpaper as bt
from mlpaper.test_constants import FPR, MC_REPEATS_LARGE


def close_lte(x, y):
    R = (x <= y) or np.allclose(x, y)
    return R


def fp_rnd(allow_nan=False):
    x = np.random.randn()
    if np.random.rand() <= 0.1:
        x = np.inf
    if np.random.rand() <= 0.1:
        x = -np.inf
    if allow_nan and np.random.rand() <= 0.1:
        x = np.nan
    return x


def test_clip_EB(runs=100):
    for _ in range(runs):
        mu = fp_rnd(allow_nan=True)
        EB0 = np.abs(fp_rnd(allow_nan=True))
        lower = fp_rnd()
        lower = lower if lower < np.inf else -np.inf
        upper = np.fmax(lower, fp_rnd())
        upper = upper if -np.inf < upper else np.inf
        min_EB = np.abs(np.random.randn())

        mu = np.clip(mu, lower, upper)
        min_EB = np.fmin(min_EB, 0.5 * (upper - lower))

        EB = bt.clip_EB(mu, EB0, lower=lower, upper=upper, min_EB=min_EB)

        assert np.isnan(EB0) == np.isnan(EB)
        if np.isnan(EB0):
            continue

        assert EB >= 0.0
        assert EB >= min_EB  # Sure EB is not NaN by here

        if not (np.isfinite(mu) and np.isfinite(lower) and np.isfinite(upper)):
            # Cannot clip from above
            assert np.fmax(EB0, min_EB) == EB
            continue

        if EB0 == np.inf:
            # EB=inf same as EB way bigger than range
            EB2 = bt.clip_EB(mu, 10 * (upper - lower), lower=lower, upper=upper, min_EB=min_EB)
            assert np.allclose(EB, EB2)
            continue

        # We should only have the all finite case by here
        assert np.all(np.isfinite([mu, EB0, lower, upper, min_EB]))
        assert np.isfinite(EB)

        # Make sure get trivial
        assert lower - 1e-10 <= mu - EB or mu + EB <= upper + 1e-10
        # Make didn't remove too much
        if EB < EB0:
            assert np.allclose(np.fmax(lower, mu - EB0), np.fmax(lower, mu - EB))
            assert np.allclose(np.fmin(upper, mu + EB0), np.fmin(upper, mu + EB))


def test_t_test_to_scipy():
    N = np.random.randint(low=2, high=10)
    x = np.random.randn(N)

    _, pval_ss = ss.ttest_1samp(x, 0.0)
    pval = bt.t_test(x)
    assert pval == pval_ss


def test_t_test_on_zero():
    # Also tests for small N
    N = np.random.randint(low=0, high=10)
    x = np.zeros(N)

    pval = bt.t_test(x)
    assert pval == 1.0


def test_t_test_zero_var():
    N = np.random.randint(low=2, high=10)
    x = np.random.rand() + np.zeros(N)
    scale = np.exp(np.random.randn()) * np.spacing(x[0])
    x = scale * np.random.randn(N) + x

    _, pval_ss = ss.ttest_1samp(x, 0.0)
    pval = bt.t_test(x)
    if np.isnan(pval_ss):
        assert pval == 0.0  # Not on zero w.p. 1
    else:
        assert pval == pval_ss


def test_t_test_inf():
    N = np.random.randint(low=1, high=10)
    x = np.zeros(N)
    x[0] = np.inf
    if np.random.rand() <= 0.5:
        x[0] = -1 * x[0]
    pval = bt.t_test(x)
    assert pval == 1.0


def test_t_EB_zero_var():
    # Also tests small N
    N = np.random.randint(low=0, high=10)
    x = np.random.rand() + np.zeros(N)
    confidence = np.random.rand()
    EB = bt.t_EB(x, confidence=confidence)
    if N <= 1:
        assert EB == np.inf
    else:
        assert np.allclose(EB, 0.0)


def test_t_EB_inf():
    N = np.random.randint(low=1, high=10)
    x = np.zeros(N)
    x[0] = np.inf
    if np.random.rand() <= 0.5:
        x[0] = -1 * x[0]

    confidence = np.random.rand()
    EB = bt.t_EB(x, confidence=confidence)
    assert EB == np.inf


def test_t_EB_coverage(runs=10, trials=100):
    pval = []
    while len(pval) < runs:
        N = np.random.randint(low=2, high=10)
        confidence = np.random.rand()

        fail = 0
        for tt in range(trials):
            x = np.random.randn(N)

            EB = bt.t_EB(x, confidence=confidence)
            mu = np.nanmean(x)
            LB, UB = mu - EB, mu + EB
            assert np.isfinite(LB) and np.isfinite(UB)
            fail += (0.0 < LB) or (UB < 0.0)
        pval.append(ss.binom_test(fail, trials, 1.0 - confidence))
    _, pval_agg = ss.combine_pvalues(pval)
    return pval_agg


def test_t_test_to_EB():
    N = np.random.randint(low=2, high=10)
    x = np.random.randn(N)

    pval = bt.t_test(x)
    EB = bt.t_EB(x, confidence=1.0 - pval)
    assert np.allclose(np.abs(np.mean(x)), EB)


def test_bernstein_test_inf():
    N = np.random.randint(low=1, high=10)
    x = np.zeros(N)
    x[0] = np.inf
    if np.random.rand() <= 0.5:
        x[0] = -1 * x[0]

    lower = np.minimum(np.min(x), np.random.randn())
    upper = np.maximum(np.max(x), np.random.randn())

    pval = bt.bernstein_test(x, lower, upper)
    assert pval == 1.0


def test_bernstein_EB_inf():
    N = np.random.randint(low=1, high=10)
    x = np.zeros(N)
    x[0] = np.inf
    if np.random.rand() <= 0.5:
        x[0] = -1 * x[0]

    lower = np.minimum(np.min(x), np.random.randn())
    upper = np.maximum(np.max(x), np.random.randn())

    confidence = np.random.rand()
    EB = bt.bernstein_EB(x, lower, upper, confidence=confidence)
    assert EB == np.inf


def test_bernstein_EB_coverage(runs=10, trials=100):
    pval = []
    while len(pval) < runs:
        # Crank up N to test this bound
        N = np.random.randint(low=200, high=1000)
        confidence = np.random.rand()

        lower = np.random.randn()
        upper = lower + np.abs(np.random.randn())

        fail = 0
        for tt in range(trials):
            x = np.random.uniform(lower, upper, size=N)
            true_mu = (lower + upper) / 2

            EB = bt.bernstein_EB(x, lower, upper, confidence=confidence)
            mu = np.mean(x)
            LB, UB = mu - EB, mu + EB
            assert np.isfinite(LB) and np.isfinite(UB)
            fail += (true_mu < LB) or (UB < true_mu)
        # Must use one-sided test since the bernstein bound can be loose.
        pval.append(ss.binom_test(fail, trials, 1.0 - confidence, alternative="greater"))
    _, pval_agg = ss.combine_pvalues(pval)
    return pval_agg


def test_bernstein_test_to_EB():
    N = np.random.randint(low=0, high=25)
    lower = np.random.randn()
    upper = lower + np.abs(np.random.randn())
    x = np.random.uniform(lower, upper, size=N)
    if N >= 1:
        x[0] = np.clip(0, lower, upper)
        x = np.random.choice(x, size=N, replace=True)

    EB = bt.bernstein_EB(x, lower, upper, confidence=0.95)
    pval = bt.bernstein_test(x, lower, upper)
    if N >= 1:
        mu = np.mean(x)
        LB, UB = mu - EB, mu + EB
        # Sanity check pval even if really small and not well numerically
        # invertible.
        if pval <= 0.05:
            assert close_lte(0, LB) or close_lte(UB, 0)
        else:
            assert close_lte(LB, 0) or close_lte(0, UB)

    epsilon = np.spacing(1.0)
    pval_adj = np.clip(pval, epsilon, 1.0 - epsilon)
    EB = bt.bernstein_EB(x, lower, upper, confidence=1.0 - pval_adj)
    # p-value very small, might not expect this to pass due to numerics, if
    # p-value = 1 then it was clipped and can't invert since we don't have
    # original.
    if 1e-6 <= pval and pval < 1.0:
        assert np.allclose(np.abs(np.mean(x)), EB)


def test_boot_EB_and_test():
    seed_iter = np.random.randint(0, 10 ** 6, size=MC_REPEATS_LARGE)
    for seed in seed_iter:
        N = np.random.randint(1, 20)
        x = np.random.randn(N)
        x[0] = 0
        x = np.random.choice(x, size=N, replace=True)
        x[0] = 0  # At least one, maybe more zeros
        mu = np.mean(x)
        confidence = np.random.rand()

        n_boot = 10

        np.random.seed(seed)
        EB, pval, CI = bt._boot_EB_and_test(x, confidence=confidence, return_CI=True, n_boot=n_boot)
        assert close_lte(mu - EB, CI[0])
        assert close_lte(CI[1], mu + EB)
        assert np.allclose(mu - EB, CI[0]) or np.allclose(mu + EB, CI[1])

        np.random.seed(seed)
        pval_ = bt.boot_test(x, n_boot=n_boot)
        assert pval == pval_

        np.random.seed(seed)
        EB_ = bt.boot_EB(x, confidence=confidence, n_boot=n_boot)
        assert EB == EB_

        if pval == 0.0:
            continue
        pval_adj = np.nextafter(1.0, 0.0) if pval == 1.0 else pval
        np.random.seed(seed)
        EB, pval_, CI = bt._boot_EB_and_test(x, confidence=1.0 - pval_adj, return_CI=True, n_boot=n_boot)
        assert pval == pval_
        assert close_lte(mu - EB, CI[0])
        assert close_lte(CI[1], mu + EB)
        # Can only guarantee one side will be zero if 0 is in BS replicates of
        # estimator.
        assert CI[0] <= 0.0 and 0.0 <= CI[1]


def test_boot_EB_and_test_custom_f():
    def take_col(x):
        return x[:, 0]

    seed_iter = np.random.randint(0, 10 ** 6, size=MC_REPEATS_LARGE)
    for seed in seed_iter:
        N = np.random.randint(1, 20)
        x = np.random.randn(N)
        x[0] = 0
        x = np.random.choice(x, size=N, replace=True)
        x[0] = 0  # At least one, maybe more zeros
        confidence = np.random.rand()

        n_boot = 10

        np.random.seed(seed)
        EB, pval, CI = bt._boot_EB_and_test(x, confidence=confidence, return_CI=True, n_boot=n_boot)

        np.random.seed(seed)
        EB_, pval_, CI_ = bt._boot_EB_and_test(
            x[:, None], f=take_col, confidence=confidence, return_CI=True, n_boot=n_boot
        )

        assert np.allclose(EB, EB_)
        assert np.allclose(pval, pval_)
        assert np.allclose(CI, CI_)


# TODO test func version is the same if use np.average
#    can also add linear transform version


def test_get_mean_EB_test():
    seed_iter = np.random.randint(0, 10 ** 6, size=MC_REPEATS_LARGE)
    for seed in seed_iter:
        N = np.random.randint(1, 20)
        x = np.random.randn(N)
        x[0] = 0
        x = np.random.choice(x, size=N, replace=True)
        mu = np.mean(x)
        confidence = np.random.rand()

        lower = np.min(x) - np.maximum(0.0, fp_rnd())
        upper = np.max(x) + np.maximum(0.0, fp_rnd())
        min_EB = np.clip(np.random.randn(), 0.0, 0.5 * (upper - lower))
        method = np.random.choice(["t", "bernstein", "boot"])

        np.random.seed(seed)
        mu_, EB, pval = bt.get_mean_EB_test(
            x, confidence=confidence, min_EB=min_EB, lower=lower, upper=upper, method=method
        )
        assert np.allclose(mu, mu_)

        np.random.seed(seed)
        mu_, EB_ = bt.get_mean_and_EB(x, confidence=confidence, min_EB=min_EB, lower=lower, upper=upper, method=method)
        assert np.allclose(mu, mu_)
        assert EB_ == EB

        np.random.seed(seed)
        pval_ = bt.get_test(x, lower=lower, upper=upper, method=method)
        assert pval_ == pval

        np.random.seed(seed)
        if method == "t":
            EB_ = bt.t_EB(x, confidence=confidence)
        elif method == "bernstein":
            EB_ = bt.bernstein_EB(x, lower=lower, upper=upper, confidence=confidence)
        else:
            EB_ = bt.boot_EB(x, confidence=confidence)
        assert EB_ >= EB or EB == min_EB  # EB_ is pre-clip
        assert EB == EB_ or EB == min_EB or np.allclose(mu - EB, lower) or np.allclose(mu + EB, upper)

        np.random.seed(seed)
        if method == "t":
            pval_ = bt.t_test(x)
        elif method == "bernstein":
            pval_ = bt.bernstein_test(x, lower=lower, upper=upper)
        else:
            pval_ = bt.boot_test(x)
        assert pval_ == pval


def test_get_func_mean_EB_test():
    def take_col(x):
        return x[:, 0]

    seed_iter = np.random.randint(0, 10 ** 6, size=MC_REPEATS_LARGE)
    for seed in seed_iter:
        N = np.random.randint(1, 20)
        x = np.random.randn(N)
        x[0] = 0
        x = np.random.choice(x, size=N, replace=True)
        confidence = np.random.rand()

        lower = np.min(x) - np.maximum(0.0, fp_rnd())
        upper = np.max(x) + np.maximum(0.0, fp_rnd())
        min_EB = np.clip(np.random.randn(), 0.0, 0.5 * (upper - lower))
        method = np.random.choice(["boot"])

        np.random.seed(seed)
        mu, EB, pval = bt.get_func_mean_EB_test(
            x[:, None], f=take_col, confidence=confidence, min_EB=min_EB, lower=lower, upper=upper, method=method
        )

        np.random.seed(seed)
        mu_, EB_, pval_ = bt.get_mean_EB_test(
            x, confidence=confidence, min_EB=min_EB, lower=lower, upper=upper, method=method
        )
        assert np.allclose(mu, mu_)
        assert np.allclose(EB, EB_)
        assert np.allclose(pval, pval_)


# TODO test func version is the same if use np.average
#    can also add linear transform version


def test_loss_summary_table():
    N = np.random.randint(low=1, high=10)
    n_methods = np.random.randint(low=1, high=5)
    n_metrics = np.random.randint(low=1, high=5)
    confidence = np.random.rand()
    # Would be good to test 'boot' too, but too much hassle with random seeds
    method_EB = np.random.choice(["t", "bernstein"])

    methods = np.random.choice(list(ascii_letters), n_methods, replace=False)
    ref_method = np.random.choice(methods)
    metrics = np.random.choice(list(ascii_letters), n_metrics, replace=False)

    cols = pd.MultiIndex.from_product([metrics, methods], names=[cc.METRIC, cc.METHOD])
    dat = np.random.randn(N, n_metrics * n_methods)
    tbl = pd.DataFrame(data=dat, index=range(N), columns=cols, dtype=float)

    limits = {
        mm: (np.min(tbl[mm].values) - np.maximum(0.0, fp_rnd()), np.max(tbl[mm].values) + np.maximum(0.0, fp_rnd()))
        for mm in metrics
    }
    del limits[metrics[0]]  # Also test missing

    perf_tbl = bt.loss_summary_table(
        tbl, ref_method, pairwise_CI=False, confidence=confidence, method_EB=method_EB, limits=limits
    )
    perf_tbl_p = bt.loss_summary_table(
        tbl, ref_method, pairwise_CI=True, confidence=confidence, method_EB=method_EB, limits=limits
    )

    # Test pairwise vs non-pairwise off EB
    mean_df = perf_tbl.xs(cc.MEAN_COL, axis=1, level=1)
    assert mean_df.equals(perf_tbl_p.xs(cc.MEAN_COL, axis=1, level=1))
    pval_df = perf_tbl.xs(cc.PVAL_COL, axis=1, level=1)
    assert pval_df.equals(perf_tbl_p.xs(cc.PVAL_COL, axis=1, level=1))

    # Test nan pattern
    assert not np.any(np.isnan(mean_df.values))
    assert np.all(np.isnan(pval_df.loc[ref_method, :].values))
    other_pvals = pval_df.loc[pval_df.index != ref_method, :].values
    assert np.all(0.0 <= other_pvals) and np.all(other_pvals <= 1.0)
    EB_df = perf_tbl.xs(cc.ERR_COL, axis=1, level=1)
    assert np.all(EB_df >= 0.0)
    EB_df = perf_tbl_p.xs(cc.ERR_COL, axis=1, level=1)
    assert np.all(np.isnan(EB_df.loc[ref_method, :].values))
    other_EB = EB_df.loc[pval_df.index != ref_method, :].values
    assert np.all(0.0 <= other_EB)

    # Now non-vectorized test
    for metric in metrics:
        lower, upper = limits.get(metric, (-np.inf, np.inf))
        range_ = upper - lower
        loss_sub = tbl[metric]
        ref_x = loss_sub[ref_method].values
        for method in methods:
            x = loss_sub[method].values

            mu, EB, pval = perf_tbl.loc[method, metric].values
            mu_p, EB_p, pval_p = perf_tbl_p.loc[method, metric].values
            assert mu == np.mean(x)
            assert mu == mu_p

            # Non-pairwise EB
            _, EB_ = bt.get_mean_and_EB(x, confidence=confidence, lower=lower, upper=upper, method=method_EB)
            assert EB == EB_

            # Pairwise EB
            if method == ref_method:
                assert np.isnan(EB_p)
            else:
                _, EB_ = bt.get_mean_and_EB(
                    x - ref_x, confidence=confidence, lower=-range_, upper=range_, method=method_EB
                )
                assert EB_p == EB_

            # P-vals
            if method == ref_method:
                assert np.isnan(pval)
                assert np.isnan(pval_p)
            else:
                pval_ = bt.get_test(x - ref_x, lower=-range_, upper=range_, method=method_EB)
                assert pval == pval_p
                assert pval == pval_


if __name__ == "__main__":
    np.random.seed(85634)

    # Already have for-loop built in
    test_boot_EB_and_test()
    test_get_mean_EB_test()

    for rr in range(MC_REPEATS_LARGE):
        test_clip_EB()
        test_t_test_to_scipy()
        test_t_test_on_zero()
        test_t_test_zero_var()
        test_t_test_inf()
        test_t_EB_zero_var()
        test_t_EB_inf()
        test_t_test_to_EB()
        test_bernstein_test_inf()
        test_bernstein_EB_inf()
        test_bernstein_test_to_EB()
        # This is a big one, we could put in loop with less iters:
        test_loss_summary_table()
        print(rr)

    print("Now running MC tests")
    test_list = [test_t_EB_coverage, test_bernstein_EB_coverage]
    for test_f in test_list:
        pval = test_f(trials=MC_REPEATS_LARGE)
        print(pval)
        assert pval >= FPR / len(test_list)
    print("passed")
