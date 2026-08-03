"""Cheap separability check with a shuffled-label control.

Splits by task so no task appears in both train and test: without that, a model can
memorise task identity and it looks like tenant signal.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from magma_frontier.eval.metrics import FPR_POINTS as _FPR_POINTS
from magma_frontier.eval.metrics import tpr_at_fpr as _tpr_at_fpr

# Session duration reflects serving-endpoint latency at collection time, not
# organizational behaviour. Excluding these gives the structure-only regime.
TIMING_FEATURES = ("duration_s", "steps_per_minute", "has_duration")


@dataclass(frozen=True, slots=True)
class SmokeResult:
    accuracy: float
    chance: float
    majority_baseline: float
    tpr_at_fpr: dict[float, float]
    null_tpr_at_fpr: dict[float, float]
    shuffled_accuracy: float
    per_tenant_recall: dict[str, float]
    n: int
    n_tenants: int


def _tpr_curve(y_test, proba, classes) -> dict[float, float]:
    """One-vs-rest TPR at each fixed low FPR, averaged over tenants."""
    return {
        fpr: float(np.mean([
            _tpr_at_fpr((y_test == cls).astype(int), proba[:, i], fpr)
            for i, cls in enumerate(classes)
        ]))
        for fpr in _FPR_POINTS
    }


def _fit_score(X, y, groups, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    scaler = StandardScaler().fit(X[train_idx])
    model = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    model.fit(scaler.transform(X[train_idx]), y[train_idx])
    proba = model.predict_proba(scaler.transform(X[test_idx]))
    predictions = model.classes_[proba.argmax(axis=1)]
    return y[test_idx], predictions, proba, model.classes_


def run_smoke(fs, seed: int = 0) -> SmokeResult:
    """Measure task-matched tenant separability, plus a shuffled-label control."""
    y = np.asarray(fs.tenant_ids)
    tenants = np.unique(y)
    if tenants.size < 2:
        raise ValueError(f"need at least two tenants, got {tenants.size}")

    X = fs.X
    groups = np.asarray(fs.task_ids)

    y_test, predictions, proba, classes = _fit_score(X, y, groups, seed)
    accuracy = float((predictions == y_test).mean())

    # A classifier that always names the most common tenant in the test fold. Chance
    # assumes balance; this does not, so it is the baseline the result must beat.
    _, class_counts = np.unique(y_test, return_counts=True)
    majority_baseline = float(class_counts.max()) / float(y_test.size)

    tpr_at_fpr = _tpr_curve(y_test, proba, classes)

    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    shuffled_test, shuffled_pred, shuffled_proba, shuffled_classes = _fit_score(
        X, y_shuffled, groups, seed
    )
    shuffled_accuracy = float((shuffled_pred == shuffled_test).mean())
    # The measured null for the low-FPR metrics: identical computation, labels broken.
    null_tpr_at_fpr = _tpr_curve(shuffled_test, shuffled_proba, shuffled_classes)

    per_tenant_recall = {
        str(cls): float((predictions[y_test == cls] == cls).mean())
        for cls in np.unique(y_test)
    }

    return SmokeResult(
        accuracy=accuracy,
        chance=1.0 / tenants.size,
        majority_baseline=majority_baseline,
        tpr_at_fpr=tpr_at_fpr,
        null_tpr_at_fpr=null_tpr_at_fpr,
        shuffled_accuracy=shuffled_accuracy,
        per_tenant_recall=per_tenant_recall,
        n=int(X.shape[0]),
        n_tenants=int(tenants.size),
    )


def run_smoke_seeds(fs, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> list[SmokeResult]:
    """Run the gate across several seeds so results carry a spread, not a point estimate."""
    return [run_smoke(fs, seed=s) for s in seeds]
