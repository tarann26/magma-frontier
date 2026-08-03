"""Per-tenant value by leave-one-tenant-out downstream utility.

Coverage is a proxy for how much a contributor adds. This is the direct measurement:
train a task-success predictor on the pool, remove each tenant from the TRAINING set
only, and see how much held-out accuracy drops. The test set is fixed across every run
so the deltas are comparable to each other.

The size control is not optional. Removing any tenant shrinks the training set and a
smaller training set hurts accuracy by itself, so each tenant's delta is paired with a
size-matched random removal drawn from the rest of the pool. The adjusted delta is the
difference. Without it every tenant looks valuable in proportion to its row count, which
measures nothing about the data.

A skill check travels with the result. `majority_baseline` is the test fold's
constant-predictor accuracy and `baseline_auc` is the model's discrimination. If the
baseline accuracy does not beat the majority baseline, or the AUC sits at 0.5, then every
per-tenant delta is a difference between two unskilled models and the axis measures
nothing — a fact that must be visible in the returned object rather than inferred later.
"""

import zlib
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

# The control is averaged over this many size-matched draws. A single draw carries noise
# comparable to the entire between-tenant spread, and because the draw is seeded per
# tenant name it does not average out on its own — it would rank contributors partly by a
# hash of their name.
CONTROL_DRAWS = 10


@dataclass(frozen=True, slots=True)
class UtilityResult:
    baseline_accuracy: float
    majority_baseline: float
    baseline_auc: float
    per_tenant_delta: dict[str, float]
    per_tenant_adjusted: dict[str, float]
    n_train: int
    n_test: int


def _fit_predict(Z, y, train_mask, test_idx, seed):
    model = HistGradientBoostingClassifier(random_state=seed, max_iter=150)
    model.fit(Z[train_mask], y[train_mask])
    predictions = model.predict(Z[test_idx])
    scores = model.predict_proba(Z[test_idx])[:, 1]
    return float((predictions == y[test_idx]).mean()), scores


def _accuracy(Z, y, train_mask, test_idx, seed) -> float:
    return _fit_predict(Z, y, train_mask, test_idx, seed)[0]


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC. 0.5 means the model carries no discriminative information."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    return float((ranks[y_true].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def loto_utility(Z: np.ndarray, tenant_ids: np.ndarray, task_ids: np.ndarray,
                 outcomes, seed: int = 0) -> UtilityResult:
    """Measure each tenant's contribution to held-out task-success prediction."""
    Z = np.asarray(Z, dtype=np.float64)
    tenant_ids = np.asarray(tenant_ids)
    task_ids = np.asarray(task_ids)

    labelled = np.array([o is not None for o in outcomes], dtype=bool)
    if not labelled.any():
        raise ValueError("loto_utility() needs at least one labelled session")

    Z, tenant_ids, task_ids = Z[labelled], tenant_ids[labelled], task_ids[labelled]
    y = np.array([bool(o) for o, keep in zip(outcomes, labelled) if keep])

    tenants = np.unique(tenant_ids)
    if tenants.size < 2:
        raise ValueError(f"loto_utility() needs at least two tenants, got {tenants.size}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(Z, y, task_ids))
    train_mask = np.zeros(Z.shape[0], dtype=bool)
    train_mask[train_idx] = True

    baseline, baseline_scores = _fit_predict(Z, y, train_mask, test_idx, seed)
    test_labels = y[test_idx]
    majority = float(max(test_labels.mean(), 1.0 - test_labels.mean()))
    auc = _auc(test_labels, baseline_scores)

    delta: dict[str, float] = {}
    adjusted: dict[str, float] = {}
    for tenant in tenants:
        drop = train_mask & (tenant_ids == tenant)
        without = train_mask & ~drop
        if drop.sum() == 0:
            # No training rows: removing this tenant changes nothing, and 0.0 is exact.
            delta[str(tenant)] = 0.0
            adjusted[str(tenant)] = 0.0
            continue
        if without.sum() == 0:
            # This tenant IS the training set. That is the most valuable a contributor
            # can be, not the least, so it must not share the 0.0 sentinel above.
            delta[str(tenant)] = float("nan")
            adjusted[str(tenant)] = float("nan")
            continue
        delta[str(tenant)] = baseline - _accuracy(Z, y, without, test_idx, seed)

        # Size-matched control: remove the same number of training rows at random from
        # everyone ELSE, so the comparison isolates what this tenant's data adds beyond
        # what an equally large slice of other people's data would have added.
        #
        # Drawing from the whole training pool would be wrong: a tenant holding a large
        # share would draw most of its own rows as its own control, the control delta
        # would converge on the real delta, and the adjustment would collapse toward zero
        # for exactly the dominant contributors this measurement exists to characterise.
        pool = np.flatnonzero(without)
        if pool.size < int(drop.sum()):
            # Not enough other people's rows to match the removal. Rather than fall back
            # to a control that includes this tenant, say the adjustment is unavailable.
            adjusted[str(tenant)] = float("nan")
            continue
        # Seeded per tenant so a tenant's control draw depends only on its own name, not
        # on which other tenants happened to be processed before it in this run.
        tenant_rng = np.random.default_rng([seed, zlib.crc32(str(tenant).encode())])
        control_deltas = []
        for _ in range(CONTROL_DRAWS):
            victims = tenant_rng.choice(pool, size=int(drop.sum()), replace=False)
            control_mask = train_mask.copy()
            control_mask[victims] = False
            control_deltas.append(baseline - _accuracy(Z, y, control_mask, test_idx, seed))
        adjusted[str(tenant)] = delta[str(tenant)] - float(np.mean(control_deltas))

    return UtilityResult(
        baseline_accuracy=baseline,
        majority_baseline=majority,
        baseline_auc=auc,
        per_tenant_delta=delta,
        per_tenant_adjusted=adjusted,
        n_train=int(train_mask.sum()),
        n_test=int(test_idx.size),
    )
