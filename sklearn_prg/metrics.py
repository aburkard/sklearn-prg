import numpy as np
from sklearn.metrics import auc, precision_recall_curve, make_scorer


def precision_recall_gain_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prevalence = np.mean(y_true)

    # Explicitly handle all-negative (π=0) and all-positive (π=1) cases
    if prevalence == 0 or prevalence == 1:
        precision_gain = np.zeros_like(precision)
        recall_gain = np.zeros_like(recall)
        return precision_gain, recall_gain

    with np.errstate(divide='ignore', invalid='ignore'):
        precision_gain = (precision - prevalence) / ((1 - prevalence) * precision)
        recall_gain = (recall - prevalence) / ((1 - prevalence) * recall)

    # Replace NaNs and infinities explicitly
    precision_gain = np.nan_to_num(precision_gain, nan=0.0, posinf=0.0, neginf=0.0)
    recall_gain = np.nan_to_num(recall_gain, nan=0.0, posinf=0.0, neginf=0.0)

    # Negative gains clipped at 0
    precision_gain = np.clip(precision_gain, 0, None)
    recall_gain = np.clip(recall_gain, 0, None)

    # CRITICAL FIX: explicitly set recall_gain=1 when recall=1
    recall_gain[recall == 1] = 1.0

    return precision_gain, recall_gain


def average_precision_recall_gain(y_true, y_scores):
    pg, rg = precision_recall_gain_curve(y_true, y_scores)
    sort_idx = np.argsort(rg)
    rg_sorted, pg_sorted = rg[sort_idx], pg[sort_idx]
    return auc(rg_sorted, pg_sorted)


average_precision_recall_gain_scorer = make_scorer(average_precision_recall_gain, needs_proba=True)
