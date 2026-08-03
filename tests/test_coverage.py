import numpy as np
import pytest

from magma_frontier.value.coverage import CoverageResult, greedy_coverage


def _three_clusters(n_per=40, seed=0):
    """t0 is spread out and covers only itself; t1 and t2 are tight and interchangeable.

    Built from DIRECTIONS, not positions. The objective is cosine similarity, which
    ignores magnitude, so a Euclidean-separated fixture can be cosine-identical: points
    at [6,0] and [0.3,0] lie on the same ray and have similarity 1.0.

    t0 spreads widely over its own region, so covering it needs many representatives.
    t1 and t2 sit on top of each other in a tight region, so one representative covers
    both. Coverage value should therefore favour t0.
    """
    rng = np.random.default_rng(seed)
    rows, tenants = [], []
    for _ in range(n_per):
        v = np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.35, 3)
        rows.append(v / np.linalg.norm(v))
        tenants.append("t0")
    for name in ("t1", "t2"):
        for _ in range(n_per):
            v = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.25, 3)
            rows.append(v / np.linalg.norm(v))
            tenants.append(name)
    return np.array(rows), np.array(tenants)


def test_selects_exactly_k():
    Z, tenants = _three_clusters()
    result = greedy_coverage(Z, tenants, k=20, seed=0)
    assert result.k == 20
    assert result.selected_idx.shape == (20,)
    assert len(set(result.selected_idx.tolist())) == 20


def test_gains_are_non_increasing_under_exhaustive_search():
    """Diminishing returns is a property of the objective, and it is only visible when
    every candidate is evaluated. Stochastic greedy samples a different subset each
    round, so its realised gains can rise; that is sampling noise, not a broken
    objective. A tiny epsilon forces the sample to cover the whole pool."""
    Z, tenants = _three_clusters()
    result = greedy_coverage(Z, tenants, k=20, epsilon=1e-9, seed=0)
    assert np.all(np.diff(result.gains) <= 1e-9)


def test_gains_are_non_negative():
    """Monotonicity: a pick can never reduce coverage."""
    Z, tenants = _three_clusters()
    result = greedy_coverage(Z, tenants, k=20, seed=0)
    assert np.all(result.gains >= 0.0)


def test_shares_sum_to_one():
    Z, tenants = _three_clusters()
    result = greedy_coverage(Z, tenants, k=30, seed=0)
    assert sum(result.per_tenant_share.values()) == pytest.approx(1.0)
    assert set(result.per_tenant_share) == {"t0", "t1", "t2"}


def test_isolated_tenant_earns_lift_above_one():
    """t0 is far from everyone; it must be over-represented relative to its pool share."""
    Z, tenants = _three_clusters()
    result = greedy_coverage(Z, tenants, k=30, seed=0)
    assert result.per_tenant_lift["t0"] > 1.0


def test_redundant_tenants_earn_lift_below_the_isolated_one():
    """t1 and t2 sit on top of each other, so each covers ground the other already covers."""
    Z, tenants = _three_clusters()
    result = greedy_coverage(Z, tenants, k=30, seed=0)
    assert result.per_tenant_lift["t1"] < result.per_tenant_lift["t0"]
    assert result.per_tenant_lift["t2"] < result.per_tenant_lift["t0"]


def test_duplicate_padding_does_not_inflate_lift():
    """The incentive property: near-duplicates add volume but almost no coverage.

    Padding is applied to t0, the tenant with no redundant peer. Padding a tenant that
    HAS an interchangeable peer is a different and weaker case: extra rows make it more
    likely to win the picks for their shared region, which partly offsets the pool-share
    dilution. That is a real limitation of the metric, recorded in the result document,
    not something this test should paper over.
    """
    Z, tenants = _three_clusters(n_per=40)
    padded_Z = np.vstack([Z, Z[tenants == "t0"] + 1e-6])
    padded_tenants = np.concatenate([tenants, np.repeat("t0", (tenants == "t0").sum())])
    plain = greedy_coverage(Z, tenants, k=30, seed=0)
    padded = greedy_coverage(padded_Z, padded_tenants, k=30, seed=0)
    assert padded.per_tenant_lift["t0"] < plain.per_tenant_lift["t0"]


def test_is_deterministic_under_seed():
    Z, tenants = _three_clusters()
    a = greedy_coverage(Z, tenants, k=20, seed=3)
    b = greedy_coverage(Z, tenants, k=20, seed=3)
    assert np.array_equal(a.selected_idx, b.selected_idx)


def test_seed_changes_the_selection():
    """Stochastic greedy samples candidates; a dead seed would give identical picks."""
    Z, tenants = _three_clusters(n_per=100)
    picks = {tuple(greedy_coverage(Z, tenants, k=20, seed=s).selected_idx.tolist())
             for s in range(5)}
    assert len(picks) > 1


def test_rejects_k_larger_than_the_pool():
    Z, tenants = _three_clusters(n_per=5)
    with pytest.raises(ValueError, match="k must be"):
        greedy_coverage(Z, tenants, k=100, seed=0)
