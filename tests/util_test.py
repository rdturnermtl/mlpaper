# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import division, print_function

from builtins import range

import numpy as np
import scipy.interpolate as si
from sklearn.preprocessing import OneHotEncoder, normalize
from statsmodels.distributions.empirical_distribution import StepFunction

from benchmark_tools import util


def test_one_hot():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    z0 = util.one_hot(y, n_labels)
    assert z0.dtype.kind == "b" and z0.shape == (N, n_labels)

    if N >= 1:
        enc = OneHotEncoder(categories=[list(range(n_labels))], drop=None, sparse=False, dtype=bool)
        z1 = enc.fit_transform(y[:, None])
        assert np.all(z0 == z1)


def test_normalize():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    log_pred_prob = np.random.randn(N, n_labels)
    z0 = util.normalize(log_pred_prob)
    assert z0.shape == (N, n_labels)

    if N >= 1:
        z1 = np.log(normalize(np.exp(log_pred_prob), norm="l1", axis=1))
        assert np.allclose(z0, z1)


def test_epsilon_noise():
    N = np.random.randint(low=0, high=10)
    x = np.random.randn(N)

    _, idx0 = np.unique(x, return_inverse=True)
    _, idx1 = np.unique(util.epsilon_noise(x), return_inverse=True)
    assert np.all(idx0 == idx1)


def test_unique_take_last():
    N = np.random.randint(low=0, high=10)

    xp = np.sort(np.random.choice(np.random.randn(N + 1), size=N, replace=True))
    yp = np.random.randn(N)
    D = {xp[ii]: yp[ii] for ii in range(N)}

    xp2, yp2 = util.unique_take_last(xp, yp)
    assert xp2.shape == yp2.shape
    assert np.all(np.unique(xp) == xp2)
    D2 = {xp2[ii]: yp2[ii] for ii in range(len(xp2))}
    assert D == D2

    xp3, yp3 = util.unique_take_last(xp)
    assert np.all(xp2 == xp3)
    assert yp3 is None

    # Func basically same as using unique, but unique returns first occurance
    xp_rev = xp[::-1]
    _, idx = np.unique(xp_rev, return_index=True)
    xp4 = xp_rev[idx]
    yp4 = yp[::-1][idx]
    assert np.all(xp2 == xp4)
    assert np.all(yp2 == yp4)


def test_cummax_strict():
    N = np.random.randint(low=0, high=10)
    x = np.random.randn(N)
    resample = (N > 0) and (np.random.rand() <= 0.5)

    if resample:  # make non-unique
        x = np.random.choice(x, size=N, replace=True)

    x.sort()
    x2 = np.copy(x)

    y = util.cummax_strict(x)

    assert np.all(x == x2)  # not modified
    assert np.all(np.diff(y) > 0.0)
    assert np.all(y >= x)
    # Delta should round off to zero
    assert np.all(x == (x + (y - x) / (2.0 * N)))
    assert resample or np.all(x == y)

    x3 = util.cummax_strict(x, copy=False)
    assert x is x3
    assert np.all(x == y)  # modified


def test_eval_step_func():
    N = np.random.randint(low=0, high=10)

    xp = np.random.randn(N)
    yp = np.random.randn(N)
    ival = np.random.randn()

    assume_sorted = np.random.rand() <= 0.5
    if assume_sorted:
        xp = np.sort(xp)

    y_grid = util.eval_step_func(xp, xp, yp, assume_sorted=assume_sorted)
    assert np.allclose(y_grid, yp)

    N_test = np.random.randint(low=0, high=10)
    x_grid = np.concatenate((xp, np.random.randn(N_test)))

    y_grid = util.eval_step_func(x_grid, xp, yp, ival)

    SF = StepFunction(xp, yp, ival=ival, side="right", sorted=assume_sorted)
    y_grid2 = SF(x_grid)

    assert np.allclose(y_grid, y_grid2)


def interp1d_vec(x_grid, x_boot, y_boot, kind):
    n_boot = x_boot.shape[0]

    y_grid_boot = np.zeros((n_boot, x_grid.size))
    for nn in range(n_boot):
        y_grid_boot[nn, :] = util._interp1d(x_grid, x_boot[nn, :], y_boot[nn, :], kind)
    assert y_grid_boot.shape == (n_boot, x_grid.size)
    return y_grid_boot


def test_interp1d_vec():
    kind_list = ["linear", "previous"]
    kind = np.random.choice(kind_list)

    N = np.random.randint(low=2, high=10)
    n_boot = np.random.randint(low=0, high=10)
    n_grid = np.random.randint(low=0, high=10)

    xp = np.random.randn(n_boot, N)

    if n_boot == 0:
        LB, UB = 0.0, 1.0
    else:
        LB, UB = np.min(xp), np.max(xp)

    for ii in range(n_boot):
        xp[ii, :] = np.random.choice(xp[ii, :], size=N, replace=True)
    xp.sort(axis=1)
    xp[:, 0] = LB
    xp[:, -1] = UB

    yp = np.random.randn(n_boot, N)
    x_grid = np.random.uniform(low=LB, high=UB, size=n_grid)

    yy = interp1d_vec(x_grid, xp, yp, kind)
    if np.random.rand() < 0.5:
        yy2 = util.interp1d(x_grid, xp, yp, kind)
    else:
        yy2 = util.interp1d(x_grid, xp, yp, kind=kind)
    assert np.all(yy == yy2)


def test_interp1d_linear():
    N = np.random.randint(low=2, high=10)
    N_test = np.random.randint(low=0, high=10)

    # Get random points with dupes, but make sure end points are seperated
    # otherwise strict spacing adjustment will changes things.
    xp = np.random.randn(N)
    LB, UB = np.min(xp), np.max(xp)
    xp = np.random.choice(xp, size=N, replace=True)  # Some dupes
    xp.sort()
    xp[0] = LB
    xp[-1] = UB

    yp = np.random.randn(N)

    x_grid = np.random.uniform(low=LB, high=UB, size=N_test)

    assert not util.STRICT_SPACING  # Check default false
    util.STRICT_SPACING = True
    y_grid = util._interp1d(x_grid, xp, yp, kind="linear")
    util.STRICT_SPACING = False
    y_grid2 = util._interp1d(x_grid, xp, yp, kind="linear")
    assert np.allclose(y_grid, y_grid2)

    f = si.interp1d(xp, yp, kind="linear", assume_sorted=True)
    y_grid3 = f(x_grid)
    assert np.allclose(y_grid2, y_grid3)


def test_interp1d_prev():
    N = np.random.randint(low=2, high=10)
    N_test = np.random.randint(low=0, high=10)

    xp = np.random.randn(N)
    xp = np.sort(xp)
    yp = np.random.randn(N)
    dupes = np.random.randint(low=0, high=10, size=N)

    xp2 = []
    yp2 = []
    for ii in range(N):
        for _ in range(dupes[ii]):
            xp2.append(xp[ii])
            yp2.append(np.random.randn())
        xp2.append(xp[ii])
        yp2.append(yp[ii])
    xp2 = np.array(xp)
    yp2 = np.array(yp)

    # Use max since can't extrapolate w/o ival
    x_grid = np.maximum(np.random.randn(N_test), xp[0])

    y_grid = util._interp1d(x_grid, xp, yp, kind="previous")
    y_grid2 = util._interp1d(x_grid, xp2, yp2, kind="previous")
    assert np.all(y_grid == y_grid2)

    y_grid3 = util.eval_step_func(x_grid, xp, yp)
    assert np.all(y_grid == y_grid3)


def test_area():
    """This tests consistentcy with interp1d. Other tests are found in
    perf_curves_test."""
    N_GRID = 1000
    kind_list = ["linear", "previous"]
    x_grid = np.linspace(0, 1, N_GRID)

    kind = np.random.choice(kind_list)

    N = np.random.randint(2, 20)
    x = np.random.rand(N)
    x.sort()
    x[0] = x_grid[0]
    x[-1] = x_grid[-1]
    y = np.random.rand(N)

    auc, = util.area(x[None, :], y[None, :], kind)

    y_grid = util._interp1d(x_grid, x, y, kind)
    auc2, = util.area(x_grid[None, :], y_grid[None, :], kind)

    # Make sure interp1d and area are consistent with each other
    assert np.abs(auc - auc2) <= 10.0 / N_GRID


if __name__ == "__main__":
    np.random.seed(75675)

    for _ in range(1000):
        test_one_hot()
        test_normalize()
        test_epsilon_noise()
        test_unique_take_last()
        test_cummax_strict()
        test_eval_step_func()
        test_interp1d_vec()
        test_interp1d_linear()
        test_interp1d_prev()
        test_area()
    print("passed")
