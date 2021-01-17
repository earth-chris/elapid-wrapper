"""
"""
import numpy as _np
from sklearn import metrics as _metrics


# a series of metrics for binary classification that accept categorical or continuous data
def binary_classes(y_true, y_pred):
    """"""
    n_ind = y_true == 0
    p_ind = y_true == 1

    # calculate the number of positive and negative samples
    n_sum = float(n_ind.sum())
    p_sum = float(p_ind.sum())

    tp = y_pred[p_ind].sum() / p_sum
    fn = 1 - tp

    fp = y_pred[n_ind].sum() / n_sum
    tn = 1 - fp

    return [tp, tn, fp, fn]


def accuracy(y_true, y_pred):
    """"""
    tp, tn, fp, fn = binary_classes(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def specificity(y_true, y_pred):
    """"""
    tp, tn, fp, fn = binary_classes(y_true, y_pred)
    return tn / (tn + fp)


def precision(y_true, y_pred):
    """"""
    tp, tn, fp, fn = binary_classes(y_true, y_pred)
    return tp / (tp + fp)


def recall(y_true, y_pred):
    """"""
    tp, tn, fp, fn = binary_classes(y_true, y_pred)
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """"""
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec))


def auc(y_true, y_pred):
    """"""
    return _metrics.roc_auc_score(y_true, y_pred)
