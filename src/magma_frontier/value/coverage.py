"""Per-tenant coverage value by stochastic-greedy facility location.

The question: if a buyer could keep only k traces and wanted them to represent the pool
as well as possible, how many would come from each tenant? A tenant supplying more than
its population share contributes distinct coverage; one supplying less is redundant with
the rest of the pool.

Facility location is monotone submodular, so greedy carries the Nemhauser-Wolsey-Fisher
(1978) 1 - 1/e guarantee. Plain greedy needs O(n*k) objective evaluations, too slow at
this scale, so this uses stochastic greedy (Mirzasoleiman et al. 2015): sample
(n/k)*ln(1/epsilon) candidates per round for a 1 - 1/e - epsilon guarantee.

The guarantee bounds COVERAGE, not value and not fairness. Coverage is a proxy for how
much a contributor adds to a pool, chosen because exact game-theoretic attribution is
intractable. Do not describe it as an optimal payout.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CoverageResult:
    selected_idx: np.ndarray
    gains: np.ndarray
    per_tenant_share: dict[str, float]
    per_tenant_lift: dict[str, float]
    k: int


def greedy_coverage(Z: np.ndarray, tenant_ids: np.ndarray, k: int = 500,
                    epsilon: float = 0.01, seed: int = 0) -> CoverageResult:
    """Select k representatives maximising cosine-similarity coverage of the pool."""
    Z = np.asarray(Z, dtype=np.float64)
    tenant_ids = np.asarray(tenant_ids)
    n = Z.shape[0]
    if not 0 < k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")

    # Cosine similarity: L2-normalise once, then sim(i, j) is a dot product.
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Zn = Z / norms

    rng = np.random.default_rng(seed)
    sample_size = min(n, max(1, int(math.ceil((n / k) * math.log(1.0 / epsilon)))))

    # best[i] is point i's similarity to its closest selected representative so far.
    # Initialised to -1.0, the floor of cosine similarity, rather than -inf: with -inf
    # every first-round marginal is +inf and the opening pick is arbitrary instead of
    # being the point that covers the pool best.
    best = np.full(n, -1.0)
    selected: list[int] = []
    gains: list[float] = []
    available = np.ones(n, dtype=bool)

    for _ in range(k):
        pool = np.flatnonzero(available)
        candidates = rng.choice(pool, size=min(sample_size, pool.size), replace=False)
        # (c, n) similarities from each candidate to every point.
        sims = Zn[candidates] @ Zn.T
        marginal = np.maximum(sims - best, 0.0).sum(axis=1)
        winner = int(candidates[int(np.argmax(marginal))])
        gains.append(float(marginal.max()))
        best = np.maximum(best, Zn[winner] @ Zn.T)
        selected.append(winner)
        available[winner] = False

    selected_arr = np.array(selected, dtype=int)
    chosen = tenant_ids[selected_arr]
    tenants = np.unique(tenant_ids)

    share = {str(t): float((chosen == t).sum()) / float(k) for t in tenants}
    pool_share = {str(t): float((tenant_ids == t).sum()) / float(n) for t in tenants}
    lift = {t: (share[t] / pool_share[t] if pool_share[t] > 0 else 0.0) for t in share}

    return CoverageResult(
        selected_idx=selected_arr,
        gains=np.array(gains, dtype=np.float64),
        per_tenant_share=share,
        per_tenant_lift=lift,
        k=k,
    )
