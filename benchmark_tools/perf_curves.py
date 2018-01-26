# Ryan Turner (turnerry@iro.umontreal.ca)
from __future__ import print_function, absolute_import
import numpy as np

EPSILON = 1e-10  # Size of pseudo-point to add to true/false positive count.

# Interpolation kinds used here
LINEAR = 'linear'
PREV = 'previous'

# ============================================================================
# Create general binary count curves
# ============================================================================


def _add_pseudo_points(fps, tps):
    """Add pseudo-points that make ROC and PR analysis give sensible results in
    corner case there are no true positive or no false positives.

    Parameters
    ----------
    fps : ndarray, shape (n_thresholds, n_boot)
        A count of false positives, at index i being the number of negative
        samples assigned a ``score >= thresholds[i]``. The total number of
        negative samples is equal to ``fps[-1]`` (thus true negatives are given
        by ``fps[-1] - fps``).
    tps : ndarray, shape (n_thresholds, n_boot)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a ``score >= thresholds[i]``. The total
        number of positive samples is equal to ``tps[-1]`` (thus false
        negatives are given by ``tps[-1] - tps``).

    Returns
    -------
    fps : ndarray, shape (n_thresholds, n_boot)
        If in corner case, `fps` after adding pseudo-points
    tps : ndarray, shape (n_thresholds, n_boot)
        If in corner case, `fps` after adding pseudo-points
    """
    assert(fps.shape == tps.shape)
    assert(fps.size > 0)  # Otherwise -1 index doesn't work

    fps_fix = (fps[-1, :] == 0)
    tps_fix = (tps[-1, :] == 0)

    if np.any(fps_fix) or np.any(tps_fix):
        fps, tps = fps.astype(float), tps.astype(float)
        fps[:, fps_fix] = EPSILON * tps[:, fps_fix]
        tps[:, tps_fix] = EPSILON * fps[:, tps_fix]
    return fps, tps


def _binary_clf_curve(y_true, y_score, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Based on `sklearn.metrics.ranking.binary_clf_curve` except that it supports
    a matrix a different sample weights `sample_weight`. It computes
    `binary_clf_curve` indenpedently for each column of `sample_weight` in a
    vectorized way. This is useful when doing a fast boot strap analysis. It is
    also more robust to corner cases such as when only a single class is
    present in `y_true`.

    Parameters
    ----------
    y_true : ndarray of type bool, shape (n_samples,)
        True targets of binary classification. Cannot be empty.
    y_score : ndarray, shape (n_samples,)
        Estimated probabilities or decision function. Must be finite.
    sample_weight : None or ndarray of shape (n_samples, n_boot)
        Sample weights. If `None`, all weights are one.

    Returns
    -------
    fps : ndarray, shape (n_thresholds, n_boot)
        A count of false positives, at index i being the number of negative
        samples assigned a ``score >= thresholds[i]``. The total number of
        negative samples is equal to ``fps[-1]`` (thus true negatives are given
        by ``fps[-1] - fps``).
    tps : ndarray, shape (n_thresholds, n_boot)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a ``score >= thresholds[i]``. The total
        number of positive samples is equal to ``tps[-1]`` (thus false
        negatives are given by ``tps[-1] - tps``).
    thresholds : ndarray, shape (n_thresholds, n_boot)
        Decreasing score values.
    """
    assert(y_true.ndim == 1 and y_true.dtype.kind == 'b')
    assert(y_score.shape == y_true.shape and np.all(np.isfinite(y_score)))
    assert(y_true.size >= 1), 'y_true.size {}'.format(y_true.size)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score, y_true = y_score[desc_score_indices], y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    if sample_weight is None:
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        assert(fps[-1] == np.sum(~y_true) and tps[-1] == np.sum(y_true))
        tps, fps = tps[:, None], fps[:, None]  # Make output 2D in either case
    else:
        assert(sample_weight.ndim == 2)
        assert(sample_weight.shape[0] == y_true.shape[0])
        assert(sample_weight.shape[1] >= 1)  # Might work at 0 anyway
        assert(np.all(np.isfinite(sample_weight)))
        # Negative weight makes no sense, 0 can violate assumps. of other funcs
        assert(np.all(sample_weight > 0))

        weight = sample_weight[desc_score_indices, :]
        tps = np.cumsum(y_true[:, None] * weight, axis=0)[threshold_idxs, :]
        fps = np.cumsum(weight, axis=0)[threshold_idxs, :] - tps
        assert(np.allclose(fps[-1, :], np.sum(weight[~y_true], axis=0)))
        assert(np.allclose(tps[-1, :], np.sum(weight[y_true], axis=0)))

    # Now put in the (0, 0) coord (y_score >= np.inf)
    zero_vec = np.zeros((1, fps.shape[1]), dtype=fps.dtype)
    fps, tps = np.r_[zero_vec, fps], np.r_[zero_vec, tps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]

    # Clean up corner case
    fps, tps = _add_pseudo_points(fps, tps)
    assert(np.all(fps[-1, :] > 0) and np.all(tps[-1, :] > 0))
    assert(fps.dtype == tps.dtype)

    # Remove any decreases due to numerics
    fps = np.maximum.accumulate(fps, axis=0)
    assert(np.all((np.diff(fps, axis=0) >= 0.0) &
                  (np.diff(tps, axis=0) >= 0.0)))

    return fps, tps, thresholds

# ============================================================================
# Convert general binary count curves to ROC, PR, PRG
# ============================================================================


def roc_curve(y_true, y_score, sample_weight=None):
    """Compute ROC curve with optional sample weight matrix.

    Based on `sklearn.metrics.ranking.roc_curve` except that it supports a
    matrix a different sample weights `sample_weight`. It computes
    the results indenpedently for each column of `sample_weight` in a
    vectorized way. This is useful when doing a fast boot strap analysis. It is
    also more robust to corner cases such as when only a single class is
    present in `y_true`.

    Parameters
    ----------
    y_true : ndarray of type bool, shape (n_samples,)
        True targets of binary classification. Cannot be empty.
    y_score : ndarray, shape (n_samples,)
        Estimated probabilities or decision function. Must be finite.
    sample_weight : None or ndarray of shape (n_samples, n_boot)
        Sample weights. If `None`, all weights are one.

    Returns
    -------
    fpr : ndarray, shape (n_thresholds, n_boot)
        The false positive rates. Each column is computed indepently by each
        column in `sample_weight`.
    tpr : ndarray, shape (n_thresholds, n_boot)
        The false positive rates. Each column is computed indepently by each
        column in `sample_weight`.
    thresholds : ndarray, shape (n_thresholds, n_boot)
        Decreasing score values.
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score,
                                             sample_weight=sample_weight)
    fpr = np.true_divide(fps, fps[-1:, :])
    tpr = np.true_divide(tps, tps[-1:, :])
    return (fpr, tpr, LINEAR), thresholds


def recall_precision_curve(y_true, y_score, sample_weight=None):
    """Compute recall precision curve with optional sample weight matrix. This
    has intentionally been named recall-precision rather than the traditional
    precision-recall.

    Based on `sklearn.metrics.ranking.precision_recall_curve` except that it
    supports a matrix a different sample weights `sample_weight`. The name
    order has been switched to `recall_precision_curve` to be consistent with
    `roc_curve` because recall is typically placed on the x-axis. It computes
    the results indenpedently for each column of `sample_weight` in a
    vectorized way. This is useful when doing a fast boot strap analysis. It is
    also more robust to corner cases such as when only a single class is
    present in `y_true`.

    Parameters
    ----------
    y_true : ndarray of type bool, shape (n_samples,)
        True targets of binary classification. Cannot be empty.
    y_score : ndarray, shape (n_samples,)
        Estimated probabilities or decision function. Must be finite.
    sample_weight : None or ndarray of shape (n_samples, n_boot)
        Sample weights. If `None`, all weights are one.

    Returns
    -------
    recall : ndarray, shape (n_thresholds, n_boot)
        The recall. Each column is computed indepently by each column in
        `sample_weight`.
    precision : ndarray, shape (n_thresholds, n_boot)
        The precision. Each column is computed indepently by each column in
        `sample_weight`.
    thresholds : ndarray, shape (n_thresholds, n_boot)
        Decreasing score values.
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score,
                                             sample_weight=sample_weight)
    recall = np.true_divide(tps, tps[-1:, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.true_divide(tps, tps + fps)
    precision[0, :] = precision[1, :]
    assert(np.all(0.0 <= precision) and np.all(precision <= 1.0))
    return (recall, precision, PREV), thresholds


def prg_curve(y_true, y_score, sample_weight=None):
    """Compute precision recall gain curve with optional sample weight matrix.
    Similar to `recall_precision_curve`.

    Parameters
    ----------
    y_true : ndarray of type bool, shape (n_samples,)
        True targets of binary classification. Cannot be empty.
    y_score : ndarray, shape (n_samples,)
        Estimated probabilities or decision function. Must be finite.
    sample_weight : None or ndarray of shape (n_samples, n_boot)
        Sample weights. If `None`, all weights are one.

    Returns
    -------
    recall_gain : ndarray, shape (n_thresholds, n_boot)
        The recall_gain. Each column is computed indepently by each column in
        `sample_weight`.
    prec_gain : ndarray, shape (n_thresholds, n_boot)
        The precision gain. Each column is computed indepently by each column
        in `sample_weight`.
    thresholds : ndarray, shape (n_thresholds, n_boot)
        Decreasing score values.
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score,
                                             sample_weight=sample_weight)
    n_neg, n_pos = fps[-1:, :], tps[-1:, :]
    fns = n_pos - tps

    den = n_neg * tps
    with np.errstate(divide='ignore', invalid='ignore'):
        rec_gain = 1.0 - np.true_divide(n_pos * fns, den)
        prec_gain = 1.0 - np.true_divide(n_pos * fps, den)
    # interpolate backward just like in PR curve
    prec_gain[0, :] = prec_gain[1, :]

    # Bring forward most recent negative point as point at 0
    with np.errstate(invalid='ignore'):
        assert(not np.any(np.diff(rec_gain, axis=0) < 0.0))
    rec_gain = np.maximum(0.0, rec_gain)
    assert(np.all(rec_gain <= 1.0))
    assert(np.all((rec_gain == 0.0) | (prec_gain <= 1.0)))
    return (rec_gain, prec_gain, PREV), thresholds
