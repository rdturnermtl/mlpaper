# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, division
from builtins import range
import warnings
from string import ascii_letters
import numpy as np
import pandas as pd
import scipy.stats as ss
import benchmark_tools.benchmark_tools as bt
import benchmark_tools.classification as btc
from benchmark_tools import util
from benchmark_tools.test_constants import MC_REPEATS_LARGE, FPR


def fp_rnd():
    x = np.random.randn()
    if np.random.rand() <= 0.1:
        x = np.inf
    if np.random.rand() <= 0.1:
        x = -np.inf
    if np.random.rand() <= 0.1:
        x = np.nan
    return x


def test_clip_EB(runs=100):
    for _ in range(runs):
        mu = fp_rnd()
        EB0 = np.abs(fp_rnd())
        lower = fp_rnd()
        lower = lower if lower < np.inf else -np.inf
        upper = np.fmax(lower, fp_rnd())
        upper = upper if -np.inf < upper else np.inf
        min_EB = np.abs(np.random.randn())

        mu = np.clip(mu, lower, upper)
        min_EB = np.fmin(min_EB, 0.5 * (upper - lower))

        EB = bt.clip_EB(mu, EB0, lower=lower, upper=upper, min_EB=min_EB)

        assert(np.isnan(EB0) == np.isnan(EB))
        if np.isnan(EB0):
            continue

        assert(EB >= 0.0)
        assert(EB >= min_EB)  # Sure EB is not NaN by here

        if not (np.isfinite(mu) and np.isfinite(lower) and np.isfinite(upper)):
            # Cannot clip from above
            assert(np.fmax(EB0, min_EB) == EB)
            continue

        if EB0 == np.inf:
            # EB=inf same as EB way bigger than range
            EB2 = bt.clip_EB(mu, 10 * (upper - lower),
                             lower=lower, upper=upper, min_EB=min_EB)
            assert(np.allclose(EB, EB2))
            continue

        # We should only have the all finite case by here
        assert(np.all(np.isfinite([mu, EB0, lower, upper, min_EB])))
        assert(np.isfinite(EB))

        # Make sure get trivial
        assert(lower - 1e-10 <= mu - EB or mu + EB <= upper + 1e-10)
        # Make didn't remove too much
        if EB < EB0:
            assert(np.allclose(np.fmax(lower, mu - EB0),
                               np.fmax(lower, mu - EB)))
            assert(np.allclose(np.fmin(upper, mu + EB0),
                               np.fmin(upper, mu + EB)))


def test_ttest1():
    N = np.random.randint(low=1, high=10)
    x = np.random.randn(N)
    x = np.random.choice(np.concatenate(([0], x)), size=N, replace=True)

    all_same = np.all(x[0] == x)

    pval0 = bt.ttest1(x, nan_on_zero=False)
    pval1 = bt.ttest1(x, nan_on_zero=True)

    if all_same:
        if N <= 1 or x[0] == 0:
            assert(pval0 == 1.0)
        else:
            # Not guaranteed to cause problem in scipy ttest which over-rides
            # to make p-value 0
            assert(np.allclose(pval0, 0.0))
        assert(np.isnan(pval1))
    else:
        assert(pval0 == pval1)

        if N <= 1:
            assert(pval0 == 1.0)
        else:
            _, pval_ss = ss.ttest_1samp(x, 0.0)
            assert(pval0 == pval_ss)
            assert(0.0 < pval0 and pval0 < 1.0)
            mu = bt.t_EB(x, confidence=1.0 - pval0)
            assert(np.allclose(mu, np.abs(np.mean(x))))

    # Now make sure infs work
    if N >= 1:
        x[0] = np.random.choice([-np.inf, np.inf])
        pval0 = bt.ttest1(x, nan_on_zero=False)
        pval1 = bt.ttest1(x, nan_on_zero=True)
        assert(pval0 == 1.0)
        all_same = N == 0 or np.all(x[0] == x)
        if all_same:
            assert(np.isnan(pval1))
        else:
            assert(pval1 == 1.0)


def test_t_EB(runs=10, trials=100):
    pval = []
    while len(pval) < runs:
        N = np.random.randint(low=0, high=10)
        confidence = np.random.rand()

        if N <= 1:
            x = np.random.randn(N)
            EB = bt.t_EB(x, confidence=confidence)
            assert(EB == np.inf)
        else:
            fail = 0
            for tt in range(trials):
                x = np.random.randn(N)

                EB = bt.t_EB(x, confidence=confidence)
                mu = np.nanmean(x)
                LB, UB = mu - EB, mu + EB
                assert(np.isfinite(LB) and np.isfinite(UB))
                fail += (0.0 < LB) or (UB < 0.0)
            pval.append(ss.binom_test(fail, trials, 1.0 - confidence))
    _, pval_agg = ss.combine_pvalues(pval)
    return pval_agg


def test_bernstein_EB(runs=10, trials=100):
    pval = []
    while len(pval) < runs:
        N = np.random.randint(low=0, high=10)
        confidence = np.random.rand()

        lower = np.random.randn()
        upper = lower + np.abs(np.random.randn())

        if N <= 1:
            x = np.random.uniform(lower, upper, size=N)
            EB = bt.bernstein_EB(x, lower, upper, confidence=confidence)
            assert(np.allclose(EB, upper - lower))
        else:
            fail = 0
            for tt in range(trials):
                # Crank up N to test this bound
                x = np.random.uniform(lower, upper, size=100 * N)
                true_mu = (lower + upper) / 2

                EB = bt.bernstein_EB(x, lower, upper, confidence=confidence)
                mu = np.mean(x)
                LB, UB = mu - EB, mu + EB
                assert(np.isfinite(LB) and np.isfinite(UB))
                fail += (true_mu < LB) or (UB < true_mu)
            pval.append(ss.binom_test(fail, trials, 1.0 - confidence,
                                      alternative='greater'))
    _, pval_agg = ss.combine_pvalues(pval)
    return pval_agg


# TODO test other forms of EB
# will need to use one sided test for bernstein
# maybe put in slow_tests

def test_get_mean_and_EB(runs=10, trials=100):
    # TODO test bernstein and boot as well, will need one sided

    pval = []
    while len(pval) < runs:
        N = np.random.randint(low=0, high=10)
        confidence = np.random.rand()

        if N <= 1:
            x = 2.0 + np.random.randn(N)
            x_ref = 1.5 + np.random.randn(N)

            with warnings.catch_warnings():  # expect warning for N=0
                warnings.simplefilter('ignore', RuntimeWarning)
                mu, EB = bt.get_mean_and_EB(x, x_ref, confidence=confidence)
                assert(np.allclose(mu, np.nanmean(mu), equal_nan=True))
                assert(EB == np.inf)

                mu, EB = bt.get_mean_and_EB(x, confidence=confidence)
                mu2, EB2 = bt.get_mean_and_EB(x, np.zeros_like(x),
                                          confidence=confidence)
                assert(np.allclose(mu, np.nanmean(mu2), equal_nan=True))
                assert(np.allclose(EB, np.nanmean(EB2), equal_nan=True))
        else:
            fail = 0
            for tt in range(trials):
                x = 2.0 + np.random.randn(N)
                x_ref = 1.5 + np.random.randn(N)

                # TODO test again that min_EB increases it to min_EB
                # also try throwing in lower and upper near where EB is

                mu, EB = bt.get_mean_and_EB(x, x_ref, confidence=confidence)
                assert(np.isfinite(mu) and np.isfinite(EB))
                assert(np.allclose(mu, np.nanmean(mu), equal_nan=True))
                err = np.nanmean(x - x_ref) - 0.5
                fail += np.abs(err) > EB

                mu, EB = bt.get_mean_and_EB(x, confidence=confidence)
                mu2, EB2 = bt.get_mean_and_EB(x, np.zeros_like(x),
                                              confidence=confidence)
                assert(np.allclose(mu, np.nanmean(mu2), equal_nan=True))
                assert(np.allclose(EB, np.nanmean(EB2), equal_nan=True))
            pval.append(ss.binom_test(fail, trials, 1.0 - confidence))
    _, pval_agg = ss.combine_pvalues(pval)
    return pval_agg


def loss_summary_table_test():
    # TODO test other EB methods here
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=1, high=10)
    n_methods = np.random.randint(low=1, high=5)

    confidence = np.random.rand()
    pairwise_CI = np.random.rand() <= 0.5

    methods = np.random.choice(list(ascii_letters), n_methods, replace=False)
    ref = np.random.choice(methods)
    metrics = btc.STD_CLASS_LOSS
    labels = range(n_labels)

    col_names = pd.MultiIndex.from_product([methods, labels],
                                           names=[bt.METHOD, btc.LABEL])
    dat = np.random.randn(N, n_labels * len(methods))
    tbl = pd.DataFrame(data=dat, index=range(N), columns=col_names,
                       dtype=float)

    y = np.random.randint(low=0, high=n_labels, size=N)
    loss_tbl = btc.loss_table(tbl, y, metrics_dict=metrics)
    perf_tbl = bt.loss_summary_table(loss_tbl, ref, pairwise_CI=pairwise_CI,
                                     confidence=confidence)
    for metric, metric_f in metrics.items():
        loss_ref = metric_f(y, util.normalize(tbl[ref].values))
        for method in methods:
            loss = metric_f(y, util.normalize(tbl[method].values))
            assert(np.allclose(loss_tbl[(metric, method)].values, loss,
                               equal_nan=True))

            if pairwise_CI:
                mu, EB = bt.get_mean_and_EB(loss=loss, loss_ref=loss_ref,
                                            confidence=confidence)
                if method == ref:
                    EB = np.nan
            else:
                mu, EB = bt.get_mean_and_EB(loss=loss, confidence=confidence)

            delta = loss - loss_ref
            if method == ref:
                pval = np.nan
            elif len(delta) == 1:
                pval = 1.0
            elif np.std(delta) == 0.0:
                pval = np.float(np.all(delta == 0.0))
            else:
                _, pval = ss.ttest_1samp(delta, 0.0)
            assert(np.allclose(perf_tbl.loc[method, metric].values,
                               [mu, EB, pval], equal_nan=True))

if __name__ == '__main__':
    np.random.seed(53634 + 199)

    for _ in range(MC_REPEATS_LARGE):
        test_clip_EB()
        test_ttest1()
        loss_summary_table_test()

    print('Now running MC tests')
    test_list = [test_t_EB, test_bernstein_EB, test_get_mean_and_EB]
    for test_f in test_list:
        pval = test_f(trials=MC_REPEATS_LARGE)
        print(pval)
        assert(pval >= FPR / len(test_list))
    print('passed')
