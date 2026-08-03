"""Shared scoring for every attack.

Following Carlini et al. (arXiv:2112.03570), a privacy attack is judged by what it
achieves at a low, fixed false positive rate rather than on average. An attack can look
respectable on accuracy while identifying nobody confidently, and confident identification
of a few subjects is what actually breaks an anonymity promise.
"""

import numpy as np

FPR_POINTS: tuple[float, ...] = (0.01, 0.001)


def tpr_at_fpr(y_binary: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """Highest TPR achievable while holding the false positive rate at or below `fpr`."""
    y_binary = np.asarray(y_binary)
    scores = np.asarray(scores, dtype=float)
    negatives = scores[y_binary == 0]
    positives = scores[y_binary == 1]
    if negatives.size == 0 or positives.size == 0:
        return 0.0
    threshold = np.quantile(negatives, 1.0 - fpr)
    return float((positives > threshold).mean())


def roc_points(y_binary: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays sorted by ascending fpr, for log-log plotting."""
    y_binary = np.asarray(y_binary)
    scores = np.asarray(scores, dtype=float)
    positives = int((y_binary == 1).sum())
    negatives = int((y_binary == 0).sum())
    # Clamping these to 1 would return an all-zero axis for an absent class, which reads
    # as a real curve. A ROC over one class is meaningless; say so rather than plot it.
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"roc_points() needs both classes present, got {positives} positive "
            f"and {negatives} negative"
        )
    order = np.argsort(-scores)
    labels = y_binary[order]
    tpr = np.cumsum(labels == 1) / positives
    fpr = np.cumsum(labels == 0) / negatives
    return fpr, tpr


def per_class_tpr(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray,
                  fpr: float) -> dict[str, float]:
    """One-vs-rest TPR at fixed `fpr`, per class, keyed by class label."""
    y_true = np.asarray(y_true)
    return {
        str(cls): tpr_at_fpr((y_true == cls).astype(int), proba[:, i], fpr)
        for i, cls in enumerate(classes)
    }
