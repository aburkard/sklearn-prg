from inspect import signature

import numpy as np
from sklearn.metrics import auc, make_scorer, precision_recall_curve, roc_curve


def precision_recall_gain_curve(y_true, y_scores):
    """Return precision gain and decreasing recall gain for binary labels.

    Labels may be 0/1 or -1/1, with 1 as the positive class. The curve
    includes the interpolated recall-gain-zero crossing. Precision gain
    may be negative; recall gain spans [0, 1]. One-class inputs return
    zeros, since the gain transformation is undefined in that case.
    """
    prevalence = np.mean(np.asarray(y_true) == 1)

    if prevalence == 0 or prevalence == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return np.zeros_like(precision), np.zeros_like(recall)

    fpr, tpr, _ = roc_curve(y_true, y_scores, drop_intermediate=False)
    start = np.searchsorted(tpr, prevalence)

    # RG=0 occurs at TPR=prevalence; interpolate before transforming.
    if tpr[start] > prevalence:
        crossing = np.interp(prevalence, tpr[start - 1:start + 1], fpr[start - 1:start + 1])
        fpr = np.r_[crossing, fpr[start:]]
        tpr = np.r_[prevalence, tpr[start:]]
    else:
        fpr, tpr = fpr[start:], tpr[start:]

    precision_gain = 1 - fpr / tpr
    recall_gain = (tpr - prevalence) / ((1 - prevalence) * tpr)
    return precision_gain[::-1], recall_gain[::-1]


def average_precision_recall_gain(y_true, y_scores):
    """Return the signed trapezoidal area over recall gain in [0, 1]."""
    pg, rg = precision_recall_gain_curve(y_true, y_scores)
    return auc(rg, pg)


if "response_method" in signature(make_scorer).parameters:
    average_precision_recall_gain_scorer = make_scorer(
        average_precision_recall_gain, response_method="predict_proba"
    )
else:
    average_precision_recall_gain_scorer = make_scorer(
        average_precision_recall_gain, needs_proba=True
    )
