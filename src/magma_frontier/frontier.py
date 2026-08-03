# src/magma_frontier/frontier.py
"""Join per-tenant value against per-tenant exposure and test for association.

The open question this answers: at CONTRIBUTOR level, are the tenants whose data is worth
most to a buyer the tenants most exposed by contributing it? The correlation is
established at record level (Wen, Backes & Zhang, NDSS 2025: 10.2x and 27.9x higher TPR
at 1% FPR for high-importance records) and contested at contributor level (El Mestari et
al., SECRYPT 2025: no significant correlation across federated-learning clients).

Spearman rather than Pearson: the question is monotonic co-movement, not linearity, and
n=22 with unequal scales makes rank correlation the honest choice. A permutation test
rather than the analytic p-value: at n=22 the asymptotic approximation is unreliable,
while permuting exposure against fixed value vectors gives an exact null.
"""

import math
from dataclasses import dataclass

import numpy as np

MIN_TENANTS = 4


@dataclass(frozen=True, slots=True)
class FrontierResult:
    tenants: tuple[str, ...]
    coverage_lift: np.ndarray
    utility_adjusted: np.ndarray
    exposure: np.ndarray
    rho_coverage: float
    p_coverage: float
    rho_utility: float
    p_utility: float
    n: int


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def _permutation_p(value: np.ndarray, exposure: np.ndarray, rho: float,
                   n_permutations: int, rng) -> float:
    """Two-sided p-value. The +1 numerator and denominator keep it strictly positive:
    a p of exactly 0 would overstate certainty at any finite permutation count."""
    extreme = 0
    for _ in range(n_permutations):
        if abs(_spearman(value, rng.permutation(exposure))) >= abs(rho):
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)


def compute_frontier(coverage: dict[str, float], utility: dict[str, float],
                     exposure: dict[str, float], seed: int = 0,
                     n_permutations: int = 10000) -> FrontierResult:
    """Correlate each value axis against exposure, per tenant."""
    if not (set(coverage) == set(utility) == set(exposure)):
        raise ValueError("coverage, utility and exposure must cover the same tenants")
    tenants = tuple(sorted(coverage))
    if len(tenants) < MIN_TENANTS:
        raise ValueError(f"need at least {MIN_TENANTS} tenants, got {len(tenants)}")

    cov = np.array([coverage[t] for t in tenants], dtype=np.float64)
    uti = np.array([utility[t] for t in tenants], dtype=np.float64)
    exp = np.array([exposure[t] for t in tenants], dtype=np.float64)

    # A NaN would not raise here. np.unique groups all NaNs together, _ranks assigns
    # them the maximum rank, and the function returns a plausible-looking rho that means
    # nothing. Callers must filter first; this is the backstop that says so out loud.
    for name, values in (("coverage", cov), ("utility", uti), ("exposure", exp)):
        if not np.isfinite(values).all():
            bad = [tenants[i] for i in np.flatnonzero(~np.isfinite(values))]
            raise ValueError(f"non-finite {name} for tenants: {bad}")

    rho_cov = _spearman(cov, exp)
    rho_uti = _spearman(uti, exp)

    rng = np.random.default_rng(seed)
    p_cov = _permutation_p(cov, exp, rho_cov, n_permutations, rng)
    p_uti = _permutation_p(uti, exp, rho_uti, n_permutations, rng)

    return FrontierResult(
        tenants=tenants,
        coverage_lift=cov,
        utility_adjusted=uti,
        exposure=exp,
        rho_coverage=rho_cov,
        p_coverage=p_cov,
        rho_utility=rho_uti,
        p_utility=p_uti,
        n=len(tenants),
    )


def join_tenants(coverage: dict[str, float], utility: dict[str, float],
                 exposure: dict[str, float],
                 all_tenants: set[str]) -> tuple[set[str], list[str]]:
    """Return the tenants usable on all three axes, and every one that is not.

    A tenant can fall out three ways: a non-finite utility adjustment, or absence as a
    key from the utility axis (every session unlabelled) or from the exposure axis (never
    reached a test fold). All three must be reported, otherwise the join silently returns
    a smaller n with no explanation of which contributors vanished or why.
    """
    usable = {t for t, v in utility.items() if math.isfinite(v)}
    shared = usable & set(exposure) & set(coverage)
    return shared, sorted(all_tenants - shared)


@dataclass(frozen=True, slots=True)
class FrontierRun:
    frontier: FrontierResult
    coverage_k: int
    utility_baseline: float
    utility_majority: float
    utility_auc: float
    exposure_seeds: int
    dropped_tenants: tuple[str, ...]


def run_frontier(fs, n_components: int = 128, k: int = 500,
                 exposure_seeds: int = 5, seed: int = 0) -> FrontierRun:
    """Build both value axes and the exposure axis, then correlate them."""
    from sklearn.model_selection import GroupShuffleSplit

    from magma_frontier.embed.representation import fit_transform
    from magma_frontier.smoke import run_smoke_seeds
    from magma_frontier.value.coverage import greedy_coverage
    from magma_frontier.value.utility import loto_utility

    groups = np.asarray(fs.task_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, _ = next(splitter.split(fs.X, np.asarray(fs.tenant_ids), groups))
    rep = fit_transform(fs, train_idx, n_components=n_components, seed=seed)

    cov = greedy_coverage(rep.Z, np.asarray(fs.tenant_ids), k=k, seed=seed)
    uti = loto_utility(rep.Z, np.asarray(fs.tenant_ids), groups, fs.outcomes, seed=seed)

    # Exposure is the supervised attack's per-tenant recall, averaged over seeds. The
    # partition attack cannot supply it: Phase 1B-1 measured ARI 0.020, so its purity
    # numbers are dominated by cluster size imbalance rather than by recovery.
    # Offset by `seed` so a different --seed genuinely varies the risk axis. Without the
    # offset every run reused smoke seeds 0..4 and produced a byte-identical exposure
    # vector, so repeated runs were not replications.
    results = run_smoke_seeds(
        fs, seeds=tuple(seed * exposure_seeds + i for i in range(exposure_seeds))
    )
    totals: dict[str, list[float]] = {}
    for result in results:
        for tenant, recall in result.per_tenant_recall.items():
            totals.setdefault(tenant, []).append(recall)
    exposure = {t: float(np.mean(v)) for t, v in totals.items()}

    all_tenants = {str(t) for t in np.unique(fs.tenant_ids)}
    shared, dropped = join_tenants(
        cov.per_tenant_lift, uti.per_tenant_adjusted, exposure, all_tenants
    )
    frontier = compute_frontier(
        {t: cov.per_tenant_lift[t] for t in shared},
        {t: uti.per_tenant_adjusted[t] for t in shared},
        {t: exposure[t] for t in shared},
        seed=seed,
    )
    return FrontierRun(
        frontier=frontier,
        coverage_k=cov.k,
        utility_baseline=uti.baseline_accuracy,
        utility_majority=uti.majority_baseline,
        utility_auc=uti.baseline_auc,
        exposure_seeds=exposure_seeds,
        dropped_tenants=tuple(dropped),
    )
