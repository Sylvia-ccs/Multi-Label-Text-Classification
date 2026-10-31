import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, hamming_loss,
    roc_auc_score, label_ranking_loss, coverage_error
)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    y_pred = (probabilities >= threshold).astype(int)

    result = {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    try:
        result["macro_auc"] = float(roc_auc_score(y_true, probabilities, average="macro"))
    except ValueError:
        result["macro_auc"] = float("nan")
    try:
        result["ranking_loss"] = float(label_ranking_loss(y_true, probabilities))
        result["coverage"] = float(coverage_error(y_true, probabilities) - 1.0)
    except ValueError:
        result["ranking_loss"] = float("nan")
        result["coverage"] = float("nan")
    return result
