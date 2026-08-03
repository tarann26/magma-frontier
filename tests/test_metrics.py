import numpy as np
import pytest

from magma_frontier.eval.metrics import FPR_POINTS, per_class_tpr, roc_points, tpr_at_fpr


def test_tpr_is_one_under_perfect_separation():
    labels = np.concatenate([np.zeros(1000, dtype=int), np.ones(100, dtype=int)])
    scores = np.concatenate([np.zeros(1000), np.ones(100)])
    assert tpr_at_fpr(labels, scores, 0.01) == pytest.approx(1.0)


def test_tpr_is_near_zero_without_signal():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(1000, dtype=int), np.ones(1000, dtype=int)])
    assert tpr_at_fpr(labels, rng.normal(size=2000), 0.01) < 0.05


def test_tpr_returns_zero_when_a_class_is_absent():
    assert tpr_at_fpr(np.zeros(10, dtype=int), np.arange(10, dtype=float), 0.01) == 0.0


def test_roc_points_are_monotone_and_bounded():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    scores = np.concatenate([rng.normal(0, 1, 500), rng.normal(2, 1, 500)])
    fpr, tpr = roc_points(labels, scores)
    assert len(fpr) == len(tpr)
    assert np.all(np.diff(fpr) >= 0)
    assert fpr.min() >= 0.0 and fpr.max() <= 1.0
    assert tpr.min() >= 0.0 and tpr.max() <= 1.0


def test_per_class_tpr_covers_every_class():
    y = np.array(["a", "a", "b", "b"])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    result = per_class_tpr(y, proba, np.array(["a", "b"]), 0.5)
    assert set(result) == {"a", "b"}
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_per_class_tpr_binds_each_class_to_its_own_column():
    """Asymmetric by construction: swapping class-to-column would invert the two scores,
    and a symmetric fixture would let that swap pass unnoticed."""
    y = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    proba = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.7], [0.6, 0.8],
        [0.1, 0.9], [0.2, 0.8], [0.3, 0.3], [0.4, 0.2],
    ])
    result = per_class_tpr(y, proba, np.array(["a", "b"]), 0.5)
    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(0.5)


def test_roc_points_rejects_a_single_class():
    with pytest.raises(ValueError, match="both classes present"):
        roc_points(np.zeros(10, dtype=int), np.arange(10, dtype=float))


def test_fpr_points_are_the_documented_operating_points():
    assert FPR_POINTS == (0.01, 0.001)
