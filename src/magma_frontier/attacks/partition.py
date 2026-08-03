"""The unsupervised partition attack: can a buyer split a pooled release by contributor?

This is the threat model with real commercial consequence. The adversary holds a purchased
pool and no labels. If clustering recovers the contributor structure, they can work out
which suppliers they want and go to them directly, which breaks the supplier's anonymity
and the marketplace's position at the same time.

Adjusted Rand Index is primary because it corrects for chance agreement: a random
clustering scores about zero no matter how many clusters it produces, so the score cannot
be inflated by picking a convenient k. Per-tenant purity accompanies it because averages
hide the distribution, and Phase 1A showed the distribution is where the finding lives.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


@dataclass(frozen=True, slots=True)
class PartitionResult:
    ari: float
    ami: float
    n_clusters: int
    per_tenant_purity: dict[str, float]
    largest_cluster_share: float


def partition_attack(Z: np.ndarray, tenant_ids: np.ndarray,
                     n_clusters: int | None = None, seed: int = 0) -> PartitionResult:
    """Cluster the pool with no labels and score the recovered partition against truth."""
    tenant_ids = np.asarray(tenant_ids)
    tenants = np.unique(tenant_ids)
    if tenants.size < 2:
        raise ValueError(f"partition attack needs at least two tenants, got {tenants.size}")

    k = n_clusters if n_clusters is not None else int(tenants.size)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)

    # Purity per tenant: the largest share of that tenant's rows landing in one cluster.
    # A tenant whose rows scatter evenly is hidden; one that concentrates is exposed.
    purity: dict[str, float] = {}
    for tenant in tenants:
        assigned = labels[tenant_ids == tenant]
        counts = Counter(assigned)
        purity[str(tenant)] = float(counts.most_common(1)[0][1] / assigned.size)

    cluster_counts = Counter(labels)
    largest = cluster_counts.most_common(1)[0][1] / labels.size

    return PartitionResult(
        ari=float(adjusted_rand_score(tenant_ids, labels)),
        ami=float(adjusted_mutual_info_score(tenant_ids, labels)),
        n_clusters=k,
        per_tenant_purity=purity,
        largest_cluster_share=float(largest),
    )
