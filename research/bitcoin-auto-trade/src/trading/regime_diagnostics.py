"""Shared descriptive regime diagnostics; none of these values train the model."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def regime_spells(series):
    """Keep gaps as boundaries and censor the first/last run of each valid block."""
    values = series.to_numpy()
    rows = []
    start = None
    for i in range(len(values) + 1):
        same = i < len(values) and values[i] >= 0 and start is not None and values[i] == values[start]
        if start is not None and not same:
            rows.append(dict(regime=int(values[start]), days=i-start,
                start=series.index[start], end=series.index[i-1],
                left_censored=start == 0 or values[start-1] < 0,
                right_censored=i == len(values) or values[i] < 0))
            start = None
        if i < len(values) and values[i] >= 0 and start is None:
            start = i
    return pd.DataFrame(rows, columns=["regime", "days", "start", "end", "left_censored", "right_censored"])


def comparison_metrics(labels, reference, values, mask):
    valid = mask & labels.ge(0) & reference.ge(0) & values.notna()
    if not valid.any():
        return dict(n=0, agreement=None, balanced_recall=None, majority_agreement=None, spearman=None)
    predicted, target = labels[valid], reference[valid]
    rho = predicted.corr(values[valid], method="spearman") if predicted.nunique() > 1 and values[valid].nunique() > 1 else None
    # Explicit recall across all 3 categories; undefined if a category is absent.
    matrix = confusion_matrix(target, predicted, labels=[0, 1, 2])
    recalls = np.divide(matrix.diagonal(), matrix.sum(axis=1), out=np.full(3, np.nan), where=matrix.sum(axis=1)>0)
    return dict(n=int(valid.sum()), agreement=float((predicted==target).mean()),
                balanced_recall=float(recalls.mean()) if np.isfinite(recalls).all() else None,
                majority_agreement=float(target.value_counts(normalize=True).max()),
                spearman=float(rho) if rho is not None and np.isfinite(rho) else None)
