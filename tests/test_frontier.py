import numpy as np
import pytest

from magma_frontier.frontier import FrontierResult, compute_frontier, join_tenants


def _aligned(n=22, seed=0):
    rng = np.random.default_rng(seed)
    tenants = [f"t{i}" for i in range(n)]
    base = rng.normal(size=n)
    coverage = {t: float(1.0 + v) for t, v in zip(tenants, base)}
    utility = {t: float(0.01 * v) for t, v in zip(tenants, base)}
    exposure = {t: float(0.2 + 0.05 * v) for t, v in zip(tenants, base)}
    return coverage, utility, exposure


def _unrelated(n=22, seed=0):
    rng = np.random.default_rng(seed)
    tenants = [f"t{i}" for i in range(n)]
    coverage = {t: float(v) for t, v in zip(tenants, rng.normal(size=n))}
    utility = {t: float(v) for t, v in zip(tenants, rng.normal(size=n))}
    exposure = {t: float(v) for t, v in zip(tenants, rng.normal(size=n))}
    return coverage, utility, exposure


def test_detects_a_monotone_relationship():
    result = compute_frontier(*_aligned(), seed=0, n_permutations=2000)
    assert result.rho_coverage > 0.9
    assert result.p_coverage < 0.01


def test_reports_no_relationship_when_there_is_none():
    result = compute_frontier(*_unrelated(), seed=0, n_permutations=2000)
    assert abs(result.rho_coverage) < 0.6
    assert result.p_coverage > 0.05


def test_covers_both_value_axes():
    result = compute_frontier(*_aligned(), seed=0, n_permutations=2000)
    assert result.rho_utility > 0.9
    assert result.p_utility < 0.01


def test_vectors_are_aligned_to_the_tenant_order():
    coverage, utility, exposure = _aligned(n=5)
    result = compute_frontier(coverage, utility, exposure, seed=0, n_permutations=500)
    for i, tenant in enumerate(result.tenants):
        assert result.coverage_lift[i] == pytest.approx(coverage[tenant])
        assert result.exposure[i] == pytest.approx(exposure[tenant])


def test_p_value_is_bounded_and_never_zero():
    """A permutation p-value of exactly 0 overstates certainty at any finite count."""
    result = compute_frontier(*_aligned(), seed=0, n_permutations=1000)
    assert 0.0 < result.p_coverage <= 1.0


def test_is_deterministic_under_seed():
    a = compute_frontier(*_aligned(), seed=7, n_permutations=1000)
    b = compute_frontier(*_aligned(), seed=7, n_permutations=1000)
    assert a.p_coverage == b.p_coverage


def test_rejects_mismatched_tenant_sets():
    coverage, utility, exposure = _aligned(n=5)
    del exposure["t0"]
    with pytest.raises(ValueError, match="same tenants"):
        compute_frontier(coverage, utility, exposure, seed=0, n_permutations=100)


def test_compute_frontier_rejects_non_finite_input():
    """NaN does not raise on its own: np.unique groups NaNs, _ranks gives them the top
    rank, and a plausible-looking rho comes back meaning nothing. The guard must fire."""
    coverage, utility, exposure = _aligned(n=6)
    utility["t0"] = float("nan")
    with pytest.raises(ValueError, match="non-finite utility"):
        compute_frontier(coverage, utility, exposure, seed=0, n_permutations=100)


def test_join_drops_and_reports_a_non_finite_utility():
    shared, dropped = join_tenants(
        {"a": 1.0, "b": 1.0}, {"a": 0.1, "b": float("nan")}, {"a": 0.2, "b": 0.2},
        {"a", "b"},
    )
    assert shared == {"a"}
    assert dropped == ["b"]


def test_join_reports_a_tenant_missing_from_the_utility_axis():
    """Every session unlabelled means the tenant is absent as a key, not NaN-valued."""
    shared, dropped = join_tenants(
        {"a": 1.0, "b": 1.0}, {"a": 0.1}, {"a": 0.2, "b": 0.2}, {"a", "b"},
    )
    assert shared == {"a"}
    assert dropped == ["b"]


def test_join_reports_a_tenant_missing_from_the_exposure_axis():
    """Never reaching a test fold in any seed also removes the tenant silently."""
    shared, dropped = join_tenants(
        {"a": 1.0, "b": 1.0}, {"a": 0.1, "b": 0.1}, {"a": 0.2}, {"a", "b"},
    )
    assert shared == {"a"}
    assert dropped == ["b"]


def test_rejects_too_few_tenants():
    coverage, utility, exposure = _aligned(n=2)
    with pytest.raises(ValueError, match="at least 4"):
        compute_frontier(coverage, utility, exposure, seed=0, n_permutations=100)


def test_negative_association_is_detected():
    """The published headline is a NEGATIVE correlation, and no existing fixture pins
    that sign — a monotone-increasing-only implementation would pass the rest."""
    coverage, utility, exposure = _aligned(n=12)
    flipped = {t: -v for t, v in exposure.items()}
    result = compute_frontier(coverage, utility, flipped, seed=0, n_permutations=2000)
    assert result.rho_coverage < -0.9
    assert result.p_coverage < 0.01
