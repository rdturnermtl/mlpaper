# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize
from statsmodels.distributions.empirical_distribution import StepFunction
import util


def test_one_hot():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    y = np.random.randint(low=0, high=n_labels, size=N)
    z0 = util.one_hot(y, n_labels)
    assert(z0.dtype.kind == 'b' and z0.shape == (N, n_labels))

    if N >= 1:
        enc = OneHotEncoder(n_values=n_labels, sparse=False, dtype=bool)
        z1 = enc.fit_transform(y[:, None])
        assert(np.all(z0 == z1))


def test_normalize():
    n_labels = np.random.randint(low=1, high=10)
    N = np.random.randint(low=0, high=10)

    log_pred_prob = np.random.randn(N, n_labels)
    z0 = util.normalize(log_pred_prob)
    assert(z0.shape == (N, n_labels))

    if N >= 1:
        z1 = np.log(normalize(np.exp(log_pred_prob), norm='l1', axis=1))
        assert(np.allclose(z0, z1))


def epsilon_noise_test():
    N = np.random.randint(low=0, high=10)
    x = np.random.randn(N)

    _, idx0 = np.unique(x, return_inverse=True)
    _, idx1 = np.unique(util.epsilon_noise(x), return_inverse=True)
    assert(np.all(idx0 == idx1))


def eval_step_func_test():
    N = np.random.randint(low=0, high=10)

    xp = np.random.randn(N)
    yp = np.random.randn(N)
    ival = np.random.randn()

    assume_sorted = np.random.rand() <= 0.5
    if assume_sorted:
        xp = np.sort(xp)

    y_grid = util.eval_step_func(xp, xp, yp, assume_sorted=assume_sorted)
    assert(np.allclose(y_grid, yp))

    N_test = np.random.randint(low=0, high=10)
    x_grid = np.concatenate((xp, np.random.randn(N_test)))

    y_grid = util.eval_step_func(x_grid, xp, yp, ival)

    SF = StepFunction(xp, yp, ival=ival, side='right', sorted=assume_sorted)
    y_grid2 = SF(x_grid)

    assert(np.allclose(y_grid, y_grid2))


def unique_take_last_test():
    N = np.random.randint(low=0, high=10)

    xp = np.sort(np.random.choice(np.random.randn(N + 1),
                                  size=N, replace=True))
    yp = np.random.randn(N)
    D = {xp[ii]: yp[ii] for ii in xrange(N)}

    xp2, yp2 = util.unique_take_last(xp, yp)
    assert(xp2.shape == yp2.shape)
    assert(np.all(np.unique(xp) == xp2))
    D2 = {xp2[ii]: yp2[ii] for ii in xrange(len(xp2))}
    assert(D == D2)

    xp3, yp3 = util.unique_take_last(xp)
    assert(np.all(xp2 == xp3))
    assert(yp3 is None)

    # Func basically same as using unique, but unique returns first occurance
    xp_rev = xp[::-1]
    _, idx = np.unique(xp_rev, return_index=True)
    xp4 = xp_rev[idx]
    yp4 = yp[::-1][idx]
    assert(np.all(xp2 == xp4))
    assert(np.all(yp2 == yp4))

np.random.seed(75675)

for _ in xrange(1000):
    test_one_hot()
    test_normalize()
    epsilon_noise_test()
    eval_step_func_test()
    unique_take_last_test()
print 'passed'
