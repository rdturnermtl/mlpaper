******************************
The ML Paper Package (mlpaper)
******************************

Easy benchmarking of machine learning models with sklearn interface with
statistical tests built-in.

Train, test, and evaluate models on multiple loss functions.
Full result tables with error bars and significance tests are a one-liner for sklearn compatible objects.
The design is documented in a workshop `paper <https://github.com/rdturnermtl/mlpaper/files/5009654/mlpaper_paper.pdf>`_ and `poster <https://github.com/rdturnermtl/mlpaper/files/5009653/mlpaper_poster.pdf>`_.

Installation
============

Only ``Python>=3.5`` is officially supported, but older versions of Python likely work as well.

The core package itself can be installed with:

.. code-block:: bash

   pip install mlpaper

To also get the dependencies for the demos in the README install with

.. code-block:: bash

   pip install mlpaper[demo]

See the `GitHub <https://github.com/rdturnermtl/mlpaper/>`_, `PyPI <https://pypi.org/project/mlpaper/>`_, and `Read the Docs <https://mlpaper.readthedocs.io/en/latest/>`_.

Executive summary
=================

* Classification uses `mlpaper.classification <https://mlpaper.readthedocs.io/en/latest/code.html#module-mlpaper.classification>`_
* Regression uses `mlpaper.regression <https://mlpaper.readthedocs.io/en/latest/code.html#module-mlpaper.regression>`_
* We use Bayes' decision rule to convert a predictive distribution to an *action* for each loss function
* Objects just support methods ``fit`` and ``predict_log_proba`` (sklearn interface)

Modular pieces:

* The "do-it-all" `just_benchmark <https://mlpaper.readthedocs.io/en/latest/code.html#mlpaper.classification.just_benchmark>`_ calls 3 modular routines
* `get_pred_log_prob <https://mlpaper.readthedocs.io/en/latest/code.html#mlpaper.classification.get_pred_log_prob>`_: predictive distributions on each test point and model
* `loss_table <https://mlpaper.readthedocs.io/en/latest/code.html#mlpaper.classification.loss_table>`_: the losses for each prediction
* `loss_summary_table <https://mlpaper.readthedocs.io/en/latest/code.html#mlpaper.mlpaper.loss_summary_table>`_: mean loss for each method and error bars/p-values

`Sciprint <https://mlpaper.readthedocs.io/en/latest/code.html#module-mlpaper.sciprint>`_:

* Publishable results: format a results dataframe for (LaTeX) publication
* Cleanly formatted: correct significant figures, shifting of exponent for compactness, and correct alignment of decimal points, units in headers

`Data splitter <https://mlpaper.readthedocs.io/en/latest/code.html#mlpaper.data_splitter.split_df>`_:

* Supports random, ordinal, or temporal splitting across features in pandas dataframes
* Jointly splitting across multiple features to test difficult generalization cases

Evaluation framework:

* Two metric types: *loss functions* and *curve summaries*
* Curve summaries: AUC for ROC, PR, and PRG
* Built-in *proper scoring rules*: log loss, Brier loss, spherical loss
* General loss matrices, and new metrics are easily added
* Non-probabilistic methods usable by pipelining a *calibrator*

Error bars and significance tests:

* Place confidence interval (CI) on mean loss of infinite test set from the same distribution
* Three options for CI in ``loss_summary_table``: t-test, bootstrap, and Bernstein bound
* The p-values are designed to match the error bars (via the 3 methods)

Error bars on curves:

* CI on raw curves (for plotting) and AUC (for tables) via bootstrap
* Vectorized bootstrap: reweight data points via multinomial distribution
* Avoids re-creating the data sets in memory (very slow)

Usage for classification problems
=================================

First, we consider the ``plot_classifier_comparison.py`` demo file. This extends
the standard sklearn `classifier
comparison <https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html>`__
but also demos the ease of `mlpaper` to create a performance
report.

The `mlpaper` package is meant to benchmark any model with any provided data set.
However, in this demo, we use the example of the three toy data sets and ten classifiers from the sklearn example:

.. figure:: https://user-images.githubusercontent.com/28273671/88328310-17f51d80-ccdd-11ea-8993-d833cb35c524.png
   :alt: sklearn

The `mlpaper` package can benchmark all of the of these methods and created a properly formatted LaTeX table (with error bars) in a few commands.
This generates a results table for copy-and-paste into a ML paper `.tex` file in a few commands.

Pandas tables with the performance results of all the methods can be
built by:

.. code:: python

    import mlpaper.classification as btc
    from mlpaper.classification import STD_BINARY_CURVES, STD_CLASS_LOSS

    performance_df, performance_curves_dict = btc.just_benchmark(
        X_train,
        y_train,
        X_test,
        y_test,
        2,
        classifiers,
        STD_CLASS_LOSS,
        STD_BINARY_CURVES,
        ref_method,
    )

This benchmarks all the models in classifiers on the data (``X_train``,
``y_train``, ``X_test``, ``y_test``) for 2-class classification. It uses
the loss function described in the dictionaries ``STD_CLASS_LOSS``, and
the curves (e.g., ROC, PR) in ``STD_BINARY_CURVES``. The ``ref_method``
defines the model that is the reference to compare against for assessing
statistically significant performance gains.

The `sciprint` module formats these tables for scientific presentation.
The performance dictionaries can be converted to cleanly formatted
tables: correct significant figures, shifting of exponent for
compactness, thresholding huge/small (crap limit) results, and correct
alignment of decimal points, units in headers, etc. Here we use:

.. code:: python

    import mlpaper.sciprint as sp

    print(
        sp.just_format_it(
            performance_df,
            shift_mod=3,
            unit_dict={"NLL": "nats"},
            crap_limit_min={"AUPRG": -1},
            EB_limit={"AUPRG": -1},
            non_finite_fmt={sp.NAN_STR: "N/A"},
            use_tex=False,
        )
    )

to export the results in plain text, or for LaTeX we use:

.. code:: python

    import mlpaper.sciprint as sp

    print(
        sp.just_format_it(
            performance_df,
            shift_mod=3,
            unit_dict={"NLL": "nats"},
            crap_limit_min={"AUPRG": -1},
            EB_limit={"AUPRG": -1},
            non_finite_fmt={sp.NAN_STR: "{--}"},
            use_tex=True,
        )
    )

Output
------

Dataset 0 Raw Results (Moons)
"""""""""""""""""""""""""""""

Here we show the input to ``just_format_it`` (``print(performance_df.to_string())``):

::

    metric                Brier                               NLL                            sphere                         zero_one                           AUC                       AP                    AUPRG
    stat                   mean     error             p      mean     error             p      mean     error             p     mean     error         p      mean     error    p      mean     error    p      mean     error    p
    method
    AdaBoost           0.415492  0.138707  1.386332e-10  0.368357  0.079299  2.946082e-10  0.363273  0.147183  7.040699e-11    0.075  0.085310  0.000008  0.949875  0.095655  0.0  0.933245  0.154225  0.0  0.904640  0.227702  0.0
    Decision Tree      0.177778  0.242857  5.124429e-08  0.403857  0.701531  4.071101e-01  0.158944  0.218431  3.489955e-09    0.050  0.070590  0.000012  0.966165  0.071165  0.0  0.947368  0.123839  0.0  0.938596  0.154283  0.0
    Gaussian Process   0.265248  0.160014  3.628068e-11  0.273804  0.104741  9.779350e-10  0.216574  0.154083  2.912358e-12    0.025  0.050567  0.000001  0.952381  0.105834  0.0  0.897840  0.224560  0.0  0.920814  0.198315  0.0
    Linear SVM         0.334650  0.248373  3.153531e-06  0.282571  0.170047  1.720037e-05  0.311622  0.239091  8.783367e-07    0.125  0.107116  0.000116  0.949875  0.075188  0.0  0.951728  0.095365  0.0  0.887049  0.222059  0.0
    Naive Bayes        0.339865  0.248629  3.457673e-06  0.282526  0.178926  3.465523e-05  0.313773  0.233882  5.719445e-07    0.125  0.107116  0.000116  0.957393  0.072682  0.0  0.957084  0.098593  0.0  0.897823  0.186842  0.0
    Nearest Neighbors  0.177778  0.205603  1.064302e-09  0.416345  0.696712  4.240499e-01  0.148434  0.175058  8.504074e-12    0.025  0.050567  0.000001  0.968672  0.073935  0.0  0.944444  0.111111  0.0  0.934985  0.162257  0.0
    Neural Net         0.324146  0.222908  3.134170e-07  0.278736  0.145830  1.091201e-06  0.297476  0.216746  8.206739e-08    0.125  0.107116  0.000116  0.959900  0.072432  0.0  0.961052  0.080379  0.0  0.915010  0.204456  0.0
    QDA                0.338089  0.262604  8.712525e-06  0.285470  0.206876  2.761767e-04  0.313055  0.243018  1.225787e-06    0.150  0.115652  0.000530  0.949875  0.077694  0.0  0.950718  0.098284  0.0  0.885171  0.192649  0.0
    RBF SVM            0.146465  0.189716  5.131397e-11  0.173264  0.167918  2.510477e-07  0.120762  0.167803  9.753115e-13    0.025  0.050567  0.000001  0.957393  0.119010  0.0  0.925618  0.183161  0.0  0.920814  0.211212  0.0
    Random Forest      0.305017  0.221354  1.639340e-07  0.264840  0.149891  9.905010e-07  0.273350  0.211773  2.624395e-08    0.075  0.085310  0.000008  0.966165  0.068922  0.0  0.975701  0.057849  0.0  0.956003  0.141548  0.0
    iid                1.004444  0.021566           NaN  0.695370  0.010787           NaN  1.005362  0.026018           NaN    0.525  0.161742       NaN  0.500000  0.000000  NaN  0.525000  0.150000  NaN  0.000000  0.000000  NaN

Dataset 0 (Moons)
"""""""""""""""""

Here we show the output of ``just_format_it``:

::

                              AP        p        AUC        p    AUPRG        p      Brier        p NLL (nats)        p     sphere        p   zero one        p
    AdaBoost           0.93(16)   <0.0001  0.950(96)  <0.0001  0.90464  <0.0001  0.42(14)   <0.0001  0.368(80)  <0.0001  0.36(15)   <0.0001  0.075(86)  <0.0001
    Decision Tree      0.95(13)   <0.0001  0.966(72)  <0.0001  0.93860  <0.0001  0.18(25)   <0.0001  0.40(71)    0.4072  0.16(22)   <0.0001  0.050(71)  <0.0001
    Gaussian Process   0.90(23)   <0.0001  0.95(11)   <0.0001  0.92081  <0.0001  0.27(17)   <0.0001  0.27(11)   <0.0001  0.22(16)   <0.0001  0.025(51)  <0.0001
    Linear SVM         0.952(96)  <0.0001  0.950(76)  <0.0001  0.88705  <0.0001  0.33(25)   <0.0001  0.28(18)   <0.0001  0.31(24)   <0.0001  0.13(11)    0.0002
    Naive Bayes        0.957(99)  <0.0001  0.957(73)  <0.0001  0.89782  <0.0001  0.34(25)   <0.0001  0.28(18)   <0.0001  0.31(24)   <0.0001  0.13(11)    0.0002
    Nearest Neighbors  0.94(12)   <0.0001  0.969(74)  <0.0001  0.93498  <0.0001  0.18(21)   <0.0001  0.42(70)    0.4241  0.15(18)   <0.0001  0.025(51)  <0.0001
    Neural Net         0.961(81)  <0.0001  0.960(73)  <0.0001  0.91501  <0.0001  0.32(23)   <0.0001  0.28(15)   <0.0001  0.30(22)   <0.0001  0.13(11)    0.0002
    QDA                0.951(99)  <0.0001  0.950(78)  <0.0001  0.88517  <0.0001  0.34(27)   <0.0001  0.29(21)    0.0003  0.31(25)   <0.0001  0.15(12)    0.0006
    RBF SVM            0.93(19)   <0.0001  0.96(12)   <0.0001  0.92081  <0.0001  0.15(19)   <0.0001  0.17(17)   <0.0001  0.12(17)   <0.0001  0.025(51)  <0.0001
    Random Forest      0.976(58)  <0.0001  0.966(69)  <0.0001  0.95600  <0.0001  0.31(23)   <0.0001  0.26(15)   <0.0001  0.27(22)   <0.0001  0.075(86)  <0.0001
    iid                0.53(15)       N/A  0.5(0)         N/A  0(0)         N/A  1.004(22)      N/A  0.695(11)      N/A  1.005(27)      N/A  0.53(17)       N/A

Dataset 0 (Moons) in LaTeX
""""""""""""""""""""""""""

Here we show the output of ``just_format_it`` with ``use_tex=True``:

::

    \begin{tabular}{|l|Sr|Sr|Sr|Sr|Sr|Sr|Sr|}
    \toprule
    {} &                      {AP} &      {p} &      {AUC} &      {p} &  {AUPRG} &      {p} &    {Brier} &      {p} & {NLL (nats)} &      {p} &   {sphere} &      {p} & {zero one} &      {p} \\
    \midrule
    AdaBoost          &  0.93(16)  &  <0.0001 &  0.950(96) &  <0.0001 &  0.90464 &  <0.0001 &  0.42(14)  &  <0.0001 &    0.368(80) &  <0.0001 &  0.36(15)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    Decision Tree     &  0.95(13)  &  <0.0001 &  0.966(72) &  <0.0001 &  0.93860 &  <0.0001 &  0.18(25)  &  <0.0001 &    0.40(71)  &   0.4072 &  0.16(22)  &  <0.0001 &  0.050(71) &  <0.0001 \\
    Gaussian Process  &  0.90(23)  &  <0.0001 &  0.95(11)  &  <0.0001 &  0.92081 &  <0.0001 &  0.27(17)  &  <0.0001 &    0.27(11)  &  <0.0001 &  0.22(16)  &  <0.0001 &  0.025(51) &  <0.0001 \\
    Linear SVM        &  0.952(96) &  <0.0001 &  0.950(76) &  <0.0001 &  0.88705 &  <0.0001 &  0.33(25)  &  <0.0001 &    0.28(18)  &  <0.0001 &  0.31(24)  &  <0.0001 &  0.13(11)  &   0.0002 \\
    Naive Bayes       &  0.957(99) &  <0.0001 &  0.957(73) &  <0.0001 &  0.89782 &  <0.0001 &  0.34(25)  &  <0.0001 &    0.28(18)  &  <0.0001 &  0.31(24)  &  <0.0001 &  0.13(11)  &   0.0002 \\
    Nearest Neighbors &  0.94(12)  &  <0.0001 &  0.969(74) &  <0.0001 &  0.93498 &  <0.0001 &  0.18(21)  &  <0.0001 &    0.42(70)  &   0.4241 &  0.15(18)  &  <0.0001 &  0.025(51) &  <0.0001 \\
    Neural Net        &  0.961(81) &  <0.0001 &  0.960(73) &  <0.0001 &  0.91501 &  <0.0001 &  0.32(23)  &  <0.0001 &    0.28(15)  &  <0.0001 &  0.30(22)  &  <0.0001 &  0.13(11)  &   0.0002 \\
    QDA               &  0.951(99) &  <0.0001 &  0.950(78) &  <0.0001 &  0.88517 &  <0.0001 &  0.34(27)  &  <0.0001 &    0.29(21)  &   0.0003 &  0.31(25)  &  <0.0001 &  0.15(12)  &   0.0006 \\
    RBF SVM           &  0.93(19)  &  <0.0001 &  0.96(12)  &  <0.0001 &  0.92081 &  <0.0001 &  0.15(19)  &  <0.0001 &    0.17(17)  &  <0.0001 &  0.12(17)  &  <0.0001 &  0.025(51) &  <0.0001 \\
    Random Forest     &  0.976(58) &  <0.0001 &  0.966(69) &  <0.0001 &  0.95600 &  <0.0001 &  0.31(23)  &  <0.0001 &    0.26(15)  &  <0.0001 &  0.27(22)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    iid               &  0.53(15)  &     {--} &  0.5(0)    &     {--} &  0(0)    &     {--} &  1.004(22) &     {--} &    0.695(11) &     {--} &  1.005(27) &     {--} &  0.53(17)  &     {--} \\
    \bottomrule
    \end{tabular}

Dataset 1 Raw Results (Circles)
"""""""""""""""""""""""""""""""

::

    metric                Brier                               NLL                            sphere                         zero_one                               AUC                         AP                      AUPRG
    stat                   mean     error             p      mean     error             p      mean     error             p     mean     error             p      mean     error      p      mean     error      p      mean     error      p
    method
    AdaBoost           0.772573  0.095313  2.033552e-07  0.576206  0.049498  1.935422e-07  0.734630  0.110164  2.279943e-07    0.175  0.123067  3.886877e-06  0.885417  0.117417  0.000  0.938284  0.095521  0.000  0.760908  0.492188  0.004
    Decision Tree      0.799998  0.518223  3.008083e-01  2.763103  1.789881  2.691681e-02  0.682842  0.442331  7.918040e-02    0.200  0.129556  2.738574e-04  0.802083  0.143964  0.000  0.863636  0.163636  0.000  0.763158  0.266426  0.000
    Gaussian Process   0.390730  0.221014  1.309465e-07  0.327736  0.134797  2.622545e-07  0.361218  0.224875  6.001903e-08    0.100  0.097167  2.365995e-07  0.963542  0.066106  0.000  0.977432  0.047043  0.000  0.930490  0.217950  0.000
    Linear SVM         1.022831  0.032154  7.027710e-02  0.704573  0.016091  7.017962e-02  1.027522  0.038764  7.042062e-02    0.600  0.158673  1.000000e+00  0.513021  0.203687  0.942  0.531643  0.175163  0.194  0.197563  0.390902  0.344
    Naive Bayes        0.644184  0.192038  3.242921e-07  0.478220  0.110889  2.871541e-07  0.630224  0.206960  4.057918e-07    0.300  0.148425  2.101106e-04  0.997396  0.013396  0.000  0.998264  0.008681  0.000  0.995747  0.030182  0.000
    Nearest Neighbors  0.300000  0.152301  5.949906e-11  0.234446  0.100982  4.246213e-11  0.276718  0.158441  1.125534e-10    0.075  0.085310  5.310307e-07  0.966146  0.049479  0.000  0.996377  0.012940  0.000  0.990702  0.051036  0.000
    Neural Net         0.699274  0.138407  2.892746e-09  0.532132  0.073755  3.119226e-09  0.664108  0.155756  3.187473e-09    0.275  0.144621  9.983420e-05  0.992188  0.025155  0.000  0.995192  0.019231  0.000  0.987240  0.055882  0.000
    QDA                0.629840  0.182293  4.465387e-08  0.473008  0.104901  4.571531e-08  0.612127  0.196927  5.707883e-08    0.275  0.144621  9.983420e-05  0.997396  0.013021  0.000  0.998264  0.010029  0.000  0.995747  0.026592  0.000
    RBF SVM            0.387512  0.207708  3.157955e-08  0.331539  0.128314  9.742683e-08  0.356649  0.210642  1.440976e-08    0.125  0.107116  6.271107e-07  0.966146  0.059580  0.000  0.979187  0.045865  0.000  0.936801  0.196317  0.000
    Random Forest      0.657978  0.206179  3.062032e-05  0.479941  0.119849  2.282042e-05  0.650341  0.222052  3.599606e-05    0.350  0.154486  8.725736e-04  0.945312  0.081904  0.000  0.970699  0.055514  0.000  0.905713  0.269476  0.000
    iid                1.071111  0.084626           NaN  0.728942  0.042566           NaN  1.084992  0.101256           NaN    0.600  0.158673           NaN  0.500000  0.000000    NaN  0.600000  0.175000    NaN  0.000000  0.000000    NaN

Dataset 1 (Circles)
"""""""""""""""""""

::

                               AP        p        AUC        p      AUPRG        p      Brier        p NLL (nats)        p     sphere        p   zero one        p
    AdaBoost           0.938(96)   <0.0001  0.89(12)   <0.0001  0.76091     0.0041  0.773(96)  <0.0001  0.576(50)  <0.0001  0.73(12)   <0.0001  0.17(13)   <0.0001
    Decision Tree      0.86(17)    <0.0001  0.80(15)   <0.0001  0.76316    <0.0001  0.80(52)    0.3009  2.8(18)     0.0270  0.68(45)    0.0792  0.20(13)    0.0003
    Gaussian Process   0.977(48)   <0.0001  0.964(67)  <0.0001  0.93049    <0.0001  0.39(23)   <0.0001  0.33(14)   <0.0001  0.36(23)   <0.0001  0.100(98)  <0.0001
    Linear SVM         0.53(18)     0.1941  0.51(21)    0.9420  0.19756     0.3440  1.023(33)   0.0703  0.705(17)   0.0702  1.028(39)   0.0705  0.60(16)    1.0000
    Naive Bayes        0.9983(87)  <0.0001  0.997(14)  <0.0001  0.996(31)  <0.0001  0.64(20)   <0.0001  0.48(12)   <0.0001  0.63(21)   <0.0001  0.30(15)    0.0003
    Nearest Neighbors  0.996(13)   <0.0001  0.966(50)  <0.0001  0.991(52)  <0.0001  0.30(16)   <0.0001  0.23(11)   <0.0001  0.28(16)   <0.0001  0.075(86)  <0.0001
    Neural Net         0.995(20)   <0.0001  0.992(26)  <0.0001  0.987(56)  <0.0001  0.70(14)   <0.0001  0.532(74)  <0.0001  0.66(16)   <0.0001  0.28(15)   <0.0001
    QDA                0.998(11)   <0.0001  0.997(14)  <0.0001  0.996(27)  <0.0001  0.63(19)   <0.0001  0.47(11)   <0.0001  0.61(20)   <0.0001  0.28(15)   <0.0001
    RBF SVM            0.979(46)   <0.0001  0.966(60)  <0.0001  0.93680    <0.0001  0.39(21)   <0.0001  0.33(13)   <0.0001  0.36(22)   <0.0001  0.13(11)   <0.0001
    Random Forest      0.971(56)   <0.0001  0.945(82)  <0.0001  0.90571    <0.0001  0.66(21)   <0.0001  0.48(12)   <0.0001  0.65(23)   <0.0001  0.35(16)    0.0009
    iid                0.60(18)        N/A  0.5(0)         N/A  0(0)           N/A  1.071(85)      N/A  0.729(43)      N/A  1.08(11)       N/A  0.60(16)       N/A

Dataset 1 (Circles) in LaTeX
""""""""""""""""""""""""""""

::

    \begin{tabular}{|l|Sr|Sr|Sr|Sr|Sr|Sr|Sr|}
    \toprule
    {} &                       {AP} &      {p} &      {AUC} &      {p} &    {AUPRG} &      {p} &    {Brier} &      {p} & {NLL (nats)} &      {p} &   {sphere} &      {p} & {zero one} &      {p} \\
    \midrule
    AdaBoost          &  0.938(96)  &  <0.0001 &  0.89(12)  &  <0.0001 &  0.76091   &   0.0041 &  0.773(96) &  <0.0001 &    0.576(50) &  <0.0001 &  0.73(12)  &  <0.0001 &  0.17(13)  &  <0.0001 \\
    Decision Tree     &  0.86(17)   &  <0.0001 &  0.80(15)  &  <0.0001 &  0.76316   &  <0.0001 &  0.80(52)  &   0.3009 &    2.8(18)   &   0.0270 &  0.68(45)  &   0.0792 &  0.20(13)  &   0.0003 \\
    Gaussian Process  &  0.977(48)  &  <0.0001 &  0.964(67) &  <0.0001 &  0.93049   &  <0.0001 &  0.39(23)  &  <0.0001 &    0.33(14)  &  <0.0001 &  0.36(23)  &  <0.0001 &  0.100(98) &  <0.0001 \\
    Linear SVM        &  0.53(18)   &   0.1941 &  0.51(21)  &   0.9420 &  0.19756   &   0.3440 &  1.023(33) &   0.0703 &    0.705(17) &   0.0702 &  1.028(39) &   0.0705 &  0.60(16)  &   1.0000 \\
    Naive Bayes       &  0.9983(87) &  <0.0001 &  0.997(14) &  <0.0001 &  0.996(31) &  <0.0001 &  0.64(20)  &  <0.0001 &    0.48(12)  &  <0.0001 &  0.63(21)  &  <0.0001 &  0.30(15)  &   0.0003 \\
    Nearest Neighbors &  0.996(13)  &  <0.0001 &  0.966(50) &  <0.0001 &  0.991(52) &  <0.0001 &  0.30(16)  &  <0.0001 &    0.23(11)  &  <0.0001 &  0.28(16)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    Neural Net        &  0.995(20)  &  <0.0001 &  0.992(26) &  <0.0001 &  0.987(56) &  <0.0001 &  0.70(14)  &  <0.0001 &    0.532(74) &  <0.0001 &  0.66(16)  &  <0.0001 &  0.28(15)  &  <0.0001 \\
    QDA               &  0.998(11)  &  <0.0001 &  0.997(14) &  <0.0001 &  0.996(27) &  <0.0001 &  0.63(19)  &  <0.0001 &    0.47(11)  &  <0.0001 &  0.61(20)  &  <0.0001 &  0.28(15)  &  <0.0001 \\
    RBF SVM           &  0.979(46)  &  <0.0001 &  0.966(60) &  <0.0001 &  0.93680   &  <0.0001 &  0.39(21)  &  <0.0001 &    0.33(13)  &  <0.0001 &  0.36(22)  &  <0.0001 &  0.13(11)  &  <0.0001 \\
    Random Forest     &  0.971(56)  &  <0.0001 &  0.945(82) &  <0.0001 &  0.90571   &  <0.0001 &  0.66(21)  &  <0.0001 &    0.48(12)  &  <0.0001 &  0.65(23)  &  <0.0001 &  0.35(16)  &   0.0009 \\
    iid               &  0.60(18)   &     {--} &  0.5(0)    &     {--} &  0(0)      &     {--} &  1.071(85) &     {--} &    0.729(43) &     {--} &  1.08(11)  &     {--} &  0.60(16)  &     {--} \\
    \bottomrule
    \end{tabular}

Dataset 2 Raw Results (Linear)
""""""""""""""""""""""""""""""

::

    metric                Brier                               NLL                            sphere                         zero_one                               AUC                       AP                    AUPRG
    stat                   mean     error             p      mean     error             p      mean     error             p     mean     error             p      mean     error    p      mean     error    p      mean     error    p
    method
    AdaBoost           0.214533  0.216136  2.523354e-09  0.266751  0.284832  3.316058e-03  0.181731  0.192985  5.067723e-11    0.050  0.070590  2.365995e-07  0.960859  0.084919  0.0  0.984375  0.046444  0.0  0.962739  0.152133  0.0
    Decision Tree      0.200000  0.282360  5.539287e-07  0.690777  0.975239  9.813826e-01  0.170711  0.241010  8.377727e-09    0.050  0.070590  2.365995e-07  0.954545  0.073593  0.0  1.000000  0.000000  0.0  1.000000  0.000000  0.0
    Gaussian Process   0.248299  0.233660  5.571488e-08  0.231293  0.167469  1.166786e-06  0.226209  0.221771  1.002195e-08    0.075  0.085310  3.288484e-06  0.977273  0.048884  0.0  0.983970  0.036602  0.0  0.967939  0.113686  0.0
    Linear SVM         0.195653  0.169766  1.953849e-12  0.171331  0.106189  8.714501e-13  0.182363  0.173447  2.092714e-12    0.075  0.085310  6.271107e-07  0.992424  0.025391  0.0  0.993883  0.020471  0.0  0.989313  0.046518  0.0
    Naive Bayes        0.182688  0.199860  1.436482e-10  0.153294  0.146642  2.446338e-09  0.169801  0.189483  2.112408e-11    0.050  0.070590  2.365995e-07  0.989899  0.025705  0.0  0.992154  0.029191  0.0  0.985926  0.053426  0.0
    Nearest Neighbors  0.288888  0.292454  8.819375e-06  0.758788  0.972439  9.062639e-01  0.253939  0.255113  3.272489e-07    0.075  0.085310  3.288484e-06  0.945707  0.079545  0.0  0.991736  0.030951  0.0  0.985062  0.062596  0.0
    Neural Net         0.241892  0.180491  6.591102e-11  0.225558  0.116770  2.636651e-10  0.213904  0.178405  1.739092e-11    0.050  0.070590  2.365995e-07  0.979798  0.041179  0.0  0.985330  0.040191  0.0  0.971326  0.097755  0.0
    QDA                0.212993  0.231863  1.247745e-08  0.229875  0.279135  1.326240e-03  0.194385  0.210940  6.717171e-10    0.075  0.085310  6.271107e-07  0.974747  0.062467  0.0  0.984199  0.046699  0.0  0.965601  0.119770  0.0
    RBF SVM            0.214270  0.250165  6.537310e-08  0.217172  0.210803  2.886575e-05  0.185181  0.225345  2.477126e-09    0.050  0.070590  2.365995e-07  0.969697  0.060865  0.0  0.980435  0.051863  0.0  0.957777  0.153369  0.0
    Random Forest      0.234000  0.239004  3.497739e-08  0.462160  0.698397  4.890795e-01  0.205669  0.216480  1.355248e-09    0.075  0.085310  6.271107e-07  0.972222  0.063131  0.0  0.993883  0.017963  0.0  0.989313  0.050657  0.0
    iid                1.017778  0.042969           NaN  0.702051  0.021516           NaN  1.021406  0.051753           NaN    0.550  0.161133           NaN  0.500000  0.000000  NaN  0.550000  0.150000  NaN  0.000000  0.000000  NaN

Dataset 2 (Linear)
""""""""""""""""""

::

                              AP        p        AUC        p      AUPRG        p      Brier        p NLL (nats)        p     sphere        p   zero one        p
    AdaBoost           0.984(47)  <0.0001  0.961(85)  <0.0001  0.96274    <0.0001  0.21(22)   <0.0001  0.27(29)    0.0034  0.18(20)   <0.0001  0.050(71)  <0.0001
    Decision Tree      1(0)       <0.0001  0.955(74)  <0.0001  1(0)       <0.0001  0.20(29)   <0.0001  0.69(98)    0.9814  0.17(25)   <0.0001  0.050(71)  <0.0001
    Gaussian Process   0.984(37)  <0.0001  0.977(49)  <0.0001  0.96794    <0.0001  0.25(24)   <0.0001  0.23(17)   <0.0001  0.23(23)   <0.0001  0.075(86)  <0.0001
    Linear SVM         0.994(21)  <0.0001  0.992(26)  <0.0001  0.989(47)  <0.0001  0.20(17)   <0.0001  0.17(11)   <0.0001  0.18(18)   <0.0001  0.075(86)  <0.0001
    Naive Bayes        0.992(30)  <0.0001  0.990(26)  <0.0001  0.986(54)  <0.0001  0.18(20)   <0.0001  0.15(15)   <0.0001  0.17(19)   <0.0001  0.050(71)  <0.0001
    Nearest Neighbors  0.992(31)  <0.0001  0.946(80)  <0.0001  0.985(63)  <0.0001  0.29(30)   <0.0001  0.76(98)    0.9063  0.25(26)   <0.0001  0.075(86)  <0.0001
    Neural Net         0.985(41)  <0.0001  0.980(42)  <0.0001  0.971(98)  <0.0001  0.24(19)   <0.0001  0.23(12)   <0.0001  0.21(18)   <0.0001  0.050(71)  <0.0001
    QDA                0.984(47)  <0.0001  0.975(63)  <0.0001  0.96560    <0.0001  0.21(24)   <0.0001  0.23(28)    0.0014  0.19(22)   <0.0001  0.075(86)  <0.0001
    RBF SVM            0.980(52)  <0.0001  0.970(61)  <0.0001  0.95778    <0.0001  0.21(26)   <0.0001  0.22(22)   <0.0001  0.19(23)   <0.0001  0.050(71)  <0.0001
    Random Forest      0.994(18)  <0.0001  0.972(64)  <0.0001  0.989(51)  <0.0001  0.23(24)   <0.0001  0.46(70)    0.4891  0.21(22)   <0.0001  0.075(86)  <0.0001
    iid                0.55(15)       N/A  0.5(0)         N/A  0(0)           N/A  1.018(43)      N/A  0.702(22)      N/A  1.021(52)      N/A  0.55(17)       N/A

Dataset 2 (Linear) in LaTeX
"""""""""""""""""""""""""""

::

    \begin{tabular}{|l|Sr|Sr|Sr|Sr|Sr|Sr|Sr|}
    \toprule
    {} &                      {AP} &      {p} &      {AUC} &      {p} &    {AUPRG} &      {p} &    {Brier} &      {p} & {NLL (nats)} &      {p} &   {sphere} &      {p} & {zero one} &      {p} \\
    \midrule
    AdaBoost          &  0.984(47) &  <0.0001 &  0.961(85) &  <0.0001 &  0.96274   &  <0.0001 &  0.21(22)  &  <0.0001 &    0.27(29)  &   0.0034 &  0.18(20)  &  <0.0001 &  0.050(71) &  <0.0001 \\
    Decision Tree     &  1(0)      &  <0.0001 &  0.955(74) &  <0.0001 &  1(0)      &  <0.0001 &  0.20(29)  &  <0.0001 &    0.69(98)  &   0.9814 &  0.17(25)  &  <0.0001 &  0.050(71) &  <0.0001 \\
    Gaussian Process  &  0.984(37) &  <0.0001 &  0.977(49) &  <0.0001 &  0.96794   &  <0.0001 &  0.25(24)  &  <0.0001 &    0.23(17)  &  <0.0001 &  0.23(23)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    Linear SVM        &  0.994(21) &  <0.0001 &  0.992(26) &  <0.0001 &  0.989(47) &  <0.0001 &  0.20(17)  &  <0.0001 &    0.17(11)  &  <0.0001 &  0.18(18)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    Naive Bayes       &  0.992(30) &  <0.0001 &  0.990(26) &  <0.0001 &  0.986(54) &  <0.0001 &  0.18(20)  &  <0.0001 &    0.15(15)  &  <0.0001 &  0.17(19)  &  <0.0001 &  0.050(71) &  <0.0001 \\
    Nearest Neighbors &  0.992(31) &  <0.0001 &  0.946(80) &  <0.0001 &  0.985(63) &  <0.0001 &  0.29(30)  &  <0.0001 &    0.76(98)  &   0.9063 &  0.25(26)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    Neural Net        &  0.985(41) &  <0.0001 &  0.980(42) &  <0.0001 &  0.971(98) &  <0.0001 &  0.24(19)  &  <0.0001 &    0.23(12)  &  <0.0001 &  0.21(18)  &  <0.0001 &  0.050(71) &  <0.0001 \\
    QDA               &  0.984(47) &  <0.0001 &  0.975(63) &  <0.0001 &  0.96560   &  <0.0001 &  0.21(24)  &  <0.0001 &    0.23(28)  &   0.0014 &  0.19(22)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    RBF SVM           &  0.980(52) &  <0.0001 &  0.970(61) &  <0.0001 &  0.95778   &  <0.0001 &  0.21(26)  &  <0.0001 &    0.22(22)  &  <0.0001 &  0.19(23)  &  <0.0001 &  0.050(71) &  <0.0001 \\
    Random Forest     &  0.994(18) &  <0.0001 &  0.972(64) &  <0.0001 &  0.989(51) &  <0.0001 &  0.23(24)  &  <0.0001 &    0.46(70)  &   0.4891 &  0.21(22)  &  <0.0001 &  0.075(86) &  <0.0001 \\
    iid               &  0.55(15)  &     {--} &  0.5(0)    &     {--} &  0(0)      &     {--} &  1.018(43) &     {--} &    0.702(22) &     {--} &  1.021(52) &     {--} &  0.55(17)  &     {--} \\
    \bottomrule
    \end{tabular}

ROC curves
""""""""""

The `just_benchmark` routines also produces ROC curves with error bars from bootstrap analysis, which have been vectorized for speed:

.. figure:: https://user-images.githubusercontent.com/28273671/88328302-13306980-ccdd-11ea-8862-2fd3e92239b3.png
   :alt: ROC

Precision-recall curves
"""""""""""""""""""""""

.. figure:: https://user-images.githubusercontent.com/28273671/88328286-0f9ce280-ccdd-11ea-815e-f3f0ce86d669.png
   :alt: PR

Precision-recall-gain curves
""""""""""""""""""""""""""""

.. figure:: https://user-images.githubusercontent.com/28273671/88328305-1592c380-ccdd-11ea-8906-79142178322f.png
   :alt: PRG

Usage for regression problems
=============================

The `mlpaper` package can also be applied to a regression problem with:

.. code:: python

    import mlpaper.regression as btr

    full_tbl = btr.just_benchmark(X_train, y_train, X_test, y_test, regressors, STD_REGR_LOSS, "iid", pairwise_CI=True)

Here we have used ``pairwise_CI=True`` which makes the confidence
intervals based on the uncertainty of the loss *difference* to the
reference method rather than a confidence interval on the actual loss.

Output
------

By extending the sklearn `regression
demo <https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py>`__
we can make simple formatted tables:

::

                 MAE       p          MSE        p   NLL (nats)        p
    BLR  0.96933(30)  0.0979  1.39881(67)   0.0665  1.58842(57)   0.9828
    GPR  0.75(13)     0.0009  0.75(28)     <0.0001  1.27(12)     <0.0001
    iid  0.96908         N/A  1.3982           N/A  1.5884           N/A

or in LaTeX:

::

    \begin{tabular}{|l|Sr|Sr|Sr|}
    \toprule
    {}  &        {MAE} &     {p} &        {MSE} &      {p} & {NLL (nats)} &      {p} \\
    \midrule
    BLR &  0.96933(30) &  0.0979 &  1.39881(67) &   0.0665 &  1.58842(57) &   0.9828 \\
    GPR &  0.75(13)    &  0.0009 &  0.75(28)    &  <0.0001 &  1.27(12)    &  <0.0001 \\
    iid &  0.96908     &     N/A &  1.3982      &      N/A &  1.5884      &      N/A \\
    \bottomrule
    \end{tabular}

.. figure:: https://user-images.githubusercontent.com/28273671/88328364-2c391a80-ccdd-11ea-8367-2e53427c184d.png
   :alt: regression demo

Contributing
============

The following instructions have been tested with Python 3.7.4 on Mac OS (10.14.6).

Install in editable mode
------------------------

First, define the variables for the paths we will use:

.. code-block:: bash

   GIT=/path/to/where/you/put/repos
   ENVS=/path/to/where/you/put/virtualenvs

Then clone the repo in your git directory ``$GIT``:

.. code-block:: bash

   cd $GIT
   git clone https://github.com/rdturnermtl/mlpaper.git

Inside your virtual environments folder ``$ENVS``, make the environment:

.. code-block:: bash

   cd $ENVS
   virtualenv mlpaper --python=python3.7
   source $ENVS/mlpaper/bin/activate

Now we can install the pip dependencies. Move back into your git directory and run

.. code-block:: bash

   cd $GIT/mlpaper
   pip install -r requirements/base.txt
   pip install -e .  # Install the package itself

Contributor tools
-----------------

First, we need to setup some needed tools:

.. code-block:: bash

   cd $ENVS
   virtualenv mlpaper_tools --python=python3.7
   source $ENVS/mlpaper_tools/bin/activate
   pip install -r $GIT/mlpaper/requirements/tools.txt

To install the pre-commit hooks for contributing run (in the ``mlpaper_tools`` environment):

.. code-block:: bash

   cd $GIT/mlpaper
   pre-commit install

To rebuild the requirements, we can run:

.. code-block:: bash

   cd $GIT/mlpaper

   # Check if there any discrepancies in the .in files
   pipreqs mlpaper/ --diff requirements/base.in
   pipreqs tests/ --diff requirements/test.in
   pipreqs demos/ --diff requirements/demo.in
   pipreqs docs/ --diff requirements/docs.in

   # Regenerate the .txt files from .in files
   pip-compile-multi --no-upgrade

Generating the documentation
----------------------------

First setup the environment for building with ``Sphinx``:

.. code-block:: bash

   cd $ENVS
   virtualenv mlpaper_docs --python=python3.7
   source $ENVS/mlpaper_docs/bin/activate
   pip install -r $GIT/mlpaper/requirements/docs.txt

Then we can do the build:

.. code-block:: bash

   cd $GIT/mlpaper/docs
   make all
   open _build/html/index.html

Documentation will be available in all formats in ``Makefile``. Use ``make html`` to only generate the HTML documentation.

Running the tests
-----------------

The tests for this package can be run with:

.. code-block:: bash

   cd $GIT/mlpaper
   ./local_test.sh

The script creates an environment using the requirements found in ``requirements/test.txt``.
A code coverage report will also be produced in ``$GIT/mlpaper/htmlcov/index.html``.

Deployment
----------

The wheel (tar ball) for deployment as a pip installable package can be built using the script:

.. code-block:: bash

   cd $GIT/mlpaper/
   ./build_wheel.sh

Links
=====

The `source <https://github.com/rdturnermtl/mlpaper/>`_ is hosted on GitHub.

The `documentation <https://mlpaper.readthedocs.io/en/latest/>`_ is hosted at Read the Docs.

Installable from `PyPI <https://pypi.org/project/mlpaper/>`_.

License
=======

This project is licensed under the Apache 2 License - see the LICENSE file for details.
