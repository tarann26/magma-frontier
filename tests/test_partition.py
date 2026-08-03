import numpy as np
import pytest

from magma_frontier.attacks.partition import PartitionResult, partition_attack


def _separable(n_per_tenant=50, n_tenants=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    rows, tenants = [], []
    for t in range(n_tenants):
        centre = np.zeros(6)
        centre[t % 6] = 4.0
        for _ in range(n_per_tenant):
            rows.append(centre + rng.normal(0, noise, 6))
            tenants.append(f"tenant{t}")
    return np.array(rows), np.array(tenants)


def _noise_only(n_per_tenant=50, n_tenants=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n_per_tenant * n_tenants, 6))
    tenants = np.array([f"tenant{i // n_per_tenant}" for i in range(len(rows))])
    return rows, tenants


def test_recovers_well_separated_tenants():
    Z, tenants = _separable()
    result = partition_attack(Z, tenants, seed=0)
    assert result.ari > 0.9
    assert result.n_clusters == 3


def test_ari_is_near_zero_without_structure():
    """ARI corrects for chance, so an unstructured pool must score near zero, not near 1/k."""
    Z, tenants = _noise_only()
    result = partition_attack(Z, tenants, seed=0)
    assert result.ari < 0.05


def test_reports_per_tenant_purity():
    Z, tenants = _separable()
    result = partition_attack(Z, tenants, seed=0)
    assert set(result.per_tenant_purity) == {"tenant0", "tenant1", "tenant2"}
    assert all(0.0 <= v <= 1.0 for v in result.per_tenant_purity.values())
    assert min(result.per_tenant_purity.values()) > 0.9


def test_purity_is_low_for_unstructured_pool():
    Z, tenants = _noise_only()
    result = partition_attack(Z, tenants, seed=0)
    assert max(result.per_tenant_purity.values()) < 0.8


def test_cluster_count_defaults_to_tenant_count():
    Z, tenants = _separable(n_tenants=4)
    assert partition_attack(Z, tenants, seed=0).n_clusters == 4


def test_cluster_count_can_be_overridden():
    Z, tenants = _separable()
    assert partition_attack(Z, tenants, n_clusters=7, seed=0).n_clusters == 7


def test_reports_largest_cluster_share():
    Z, tenants = _separable()
    result = partition_attack(Z, tenants, seed=0)
    assert 0.0 < result.largest_cluster_share <= 1.0


def test_is_deterministic_under_seed():
    Z, tenants = _separable()
    a = partition_attack(Z, tenants, seed=5)
    b = partition_attack(Z, tenants, seed=5)
    assert a.ari == b.ari
    assert a.per_tenant_purity == b.per_tenant_purity


def test_seed_actually_reaches_the_clustering():
    """Determinism alone cannot prove the seed is wired up.

    On a well-separated fixture KMeans converges to the same optimum from any
    initialization, so `test_is_deterministic_under_seed` passes even when `seed` never
    reaches KMeans at all. An unstructured pool has many near-equivalent local optima, so
    if the seed is genuinely propagating, different seeds must land on different
    partitions.
    """
    Z, tenants = _noise_only(n_per_tenant=60, n_tenants=4)
    aris = {partition_attack(Z, tenants, seed=s).ari for s in range(6)}
    assert len(aris) > 1, "every seed gave the same result; seed is not reaching KMeans"


def test_rejects_single_tenant():
    Z, _ = _separable(n_tenants=1)
    with pytest.raises(ValueError, match="at least two tenants"):
        partition_attack(Z, np.array(["only"] * len(Z)), seed=0)
