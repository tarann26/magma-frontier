import numpy as np
import pytest

from magma_frontier.features.extract import FeatureSet
from magma_frontier.smoke import _tpr_at_fpr, run_smoke, run_smoke_seeds


def _separable(n_per_tenant=200, n_tenants=3, noise=0.15, seed=0):
    """Tenants sit at different feature centroids; tasks are shared across tenants."""
    rng = np.random.default_rng(seed)
    rows, tenants, tasks = [], [], []
    for t in range(n_tenants):
        centre = np.zeros(6)
        centre[t % 6] = 3.0
        for i in range(n_per_tenant):
            rows.append(centre + rng.normal(0, noise, 6))
            tenants.append(f"tenant{t}")
            tasks.append(f"task{i % 10}")
    return FeatureSet(
        X=np.array(rows), feature_names=tuple(f"f{i}" for i in range(6)),
        tenant_ids=tuple(tenants), task_ids=tuple(tasks),
        session_ids=tuple(f"s{i}" for i in range(len(rows))),
        outcomes=tuple(True for _ in rows),
        ngram_vocabulary=(),
    )


def _noise_only(n_per_tenant=200, n_tenants=3, seed=0):
    """No tenant signal at all: every tenant drawn from one distribution."""
    rng = np.random.default_rng(seed)
    rows, tenants, tasks = [], [], []
    for t in range(n_tenants):
        for i in range(n_per_tenant):
            rows.append(rng.normal(0, 1, 6))
            tenants.append(f"tenant{t}")
            tasks.append(f"task{i % 10}")
    return FeatureSet(
        X=np.array(rows), feature_names=tuple(f"f{i}" for i in range(6)),
        tenant_ids=tuple(tenants), task_ids=tuple(tasks),
        session_ids=tuple(f"s{i}" for i in range(len(rows))),
        outcomes=tuple(True for _ in rows),
        ngram_vocabulary=(),
    )


def test_detects_real_separability():
    result = run_smoke(_separable(), seed=0)
    assert result.accuracy > 0.85
    assert result.chance == pytest.approx(1 / 3, abs=0.02)


def test_reports_chance_on_noise():
    result = run_smoke(_noise_only(), seed=0)
    assert result.accuracy < result.chance + 0.10


def test_shuffled_labels_collapse_to_chance():
    """The canary. Signal under shuffled labels means pipeline leakage.

    The bound is stated relative to chance rather than as a loose absolute, because
    this assertion is the only thing standing between a leaking pipeline and a
    result nobody can trust.
    """
    result = run_smoke(_separable(), seed=0)
    assert result.shuffled_accuracy < result.chance + 0.10


def test_reports_tpr_at_low_fpr():
    result = run_smoke(_separable(), seed=0)
    assert set(result.tpr_at_fpr) == {0.01, 0.001}
    assert all(0.0 <= v <= 1.0 for v in result.tpr_at_fpr.values())


def test_tpr_at_fpr_is_one_when_classes_are_perfectly_separated():
    labels = np.concatenate([np.zeros(1000, dtype=int), np.ones(100, dtype=int)])
    scores = np.concatenate([np.zeros(1000), np.ones(100)])
    assert _tpr_at_fpr(labels, scores, 0.01) == pytest.approx(1.0)


def test_tpr_at_fpr_is_near_zero_when_scores_carry_no_signal():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(1000, dtype=int), np.ones(1000, dtype=int)])
    scores = rng.normal(size=2000)
    assert _tpr_at_fpr(labels, scores, 0.01) < 0.05


def test_tpr_at_fpr_returns_zero_when_a_class_is_absent():
    labels = np.zeros(10, dtype=int)
    scores = np.arange(10, dtype=float)
    assert _tpr_at_fpr(labels, scores, 0.01) == 0.0


def test_is_deterministic_under_seed():
    a = run_smoke(_separable(), seed=7)
    b = run_smoke(_separable(), seed=7)
    assert a.accuracy == b.accuracy
    assert a.shuffled_accuracy == b.shuffled_accuracy


def test_rejects_single_tenant():
    fs = _separable(n_tenants=1)
    with pytest.raises(ValueError, match="at least two tenants"):
        run_smoke(fs)


def test_reports_measured_null_at_the_same_operating_points():
    result = run_smoke(_separable(), seed=0)
    assert set(result.null_tpr_at_fpr) == set(result.tpr_at_fpr)
    for fpr, null_tpr in result.null_tpr_at_fpr.items():
        assert null_tpr < result.tpr_at_fpr[fpr]


def test_per_tenant_recall_covers_every_tenant_in_the_test_fold():
    result = run_smoke(_separable(), seed=0)
    assert len(result.per_tenant_recall) >= 2
    assert all(0.0 <= v <= 1.0 for v in result.per_tenant_recall.values())


def test_majority_baseline_is_reported_and_near_chance_when_balanced():
    result = run_smoke(_separable(), seed=0)
    assert result.majority_baseline >= result.chance
    assert result.majority_baseline < result.chance + 0.10


def test_run_smoke_seeds_uses_a_different_split_per_seed():
    """The old version's `n == n` disjunct was unconditionally true and could not fail."""
    results = run_smoke_seeds(_separable(n_per_tenant=60), seeds=(0, 1, 2))
    assert len(results) == 3
    shuffled = [r.shuffled_accuracy for r in results]
    assert len(set(shuffled)) > 1, "identical shuffled accuracy means the split never varied"
