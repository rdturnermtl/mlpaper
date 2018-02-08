# Ryan Turner (turnerry@iro.umontreal.ca)
# Modification of sklearn plot_compare_gpr_krr.py by
# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause
from __future__ import print_function, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.linear_model import BayesianRidge
import benchmark_tools.regression as btr
from benchmark_tools.regression import STD_REGR_LOSS
import benchmark_tools.sciprint as sp

rng = np.random.RandomState(0)

# TODO set general random seed too


def simple_data():
    X = 15 * rng.rand(100, 1)
    y = np.sin(X).ravel()
    y += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise
    return X, y

X_train, y_train = simple_data()
X_test, y_test = simple_data()

ridge1 = BayesianRidge()

gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    + WhiteKernel(1e-1)
gpr = GaussianProcessRegressor(kernel=gp_kernel)

regressors = \
    {'BLR': ridge1,
     'GPR': gpr,
     'iid': btr.JustNoise()}

full_tbl = btr.just_benchmark(X_train, y_train, X_test, y_test,
                              regressors, STD_REGR_LOSS, 'iid',
                              pairwise_CI=True)

print(sp.just_format_it(full_tbl, shift_mod=3, unit_dict={'NLL': 'nats'},
                        crap_limit_min={'NLL': 1}, EB_limit={'NLL': 1},
                        non_finite_fmt={sp.NAN_STR: 'N/A'}, use_tex=False))

print(sp.just_format_it(full_tbl, shift_mod=3, unit_dict={'NLL': 'nats'},
                        crap_limit_min={'NLL': 1}, EB_limit={'NLL': 1},
                        non_finite_fmt={sp.NAN_STR: 'N/A'}, use_tex=True))

# Predict using kernel ridge
X_plot = np.linspace(0, 20, 1000)[:, None]

# Predict using gaussian process regressor
y_gpr, std_gpr = gpr.predict(X_plot, return_std=True)

# Predict using gaussian process regressor
y_blr, std_blr = ridge1.predict(X_plot, return_std=True)

# Plot results
plt.figure(figsize=(10, 5))
lw = 2

plt.scatter(X_train, y_train, c='k', label='data')
plt.plot(X_plot, np.sin(X_plot), color='navy', lw=lw, label='True')

plt.plot(X_plot, y_blr, color='red', lw=lw, label='BLR')
plt.fill_between(X_plot[:, 0], y_blr - 2.0 * std_blr, y_blr + 2.0 * std_blr,
                 color='red', alpha=0.2)

plt.plot(X_plot, y_gpr, color='darkorange', lw=lw,
         label='GPR (%s)' % gpr.kernel_)
plt.fill_between(X_plot[:, 0], y_gpr - 2.0 * std_gpr, y_gpr + 2.0 * std_gpr,
                 color='darkorange', alpha=0.2)

plt.xlabel('data')
plt.ylabel('target')
plt.xlim(0, 20)
plt.ylim(-4, 4)
plt.legend(loc='best', scatterpoints=1, prop={'size': 8})
plt.grid()
plt.tight_layout(pad=0.0)
plt.show()
plt.savefig('regress.png', format='png', dpi=300, pad=0)
