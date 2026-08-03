"""Reproduce every number in the Phase 1B-2 frontier result document.

Two stages, so the expensive work happens once and every later question is cheap:

    uv run python -u scripts/frontier_analysis.py cache     # ~60 min, once
    uv run python -u scripts/frontier_analysis.py report    # ~10 min, repeatable

`cache` parses the 1.9 GB corpus, fits the representation, runs the five supervised-attack
seeds that give the exposure axis, and runs the utility estimator, then writes all of it to
`out/frontier_cache.npz` as plain arrays. `report` reads that cache and does the cheap
analysis: coverage lift over many seeds, the rank correlations, the permutation tests, the
bootstrap interval and the per-tenant table.

Always run with `python -u`. Without it stdout is buffered when redirected to a file and a
long run gives no progress signal at all.

Thread caps are set at import time so this takes a bounded slice of the machine rather
than every idle core.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from magma_frontier.corpus import build_corpus  # noqa: E402
from magma_frontier.embed.representation import fit_transform  # noqa: E402
from magma_frontier.frontier import _spearman  # noqa: E402
from magma_frontier.smoke import TIMING_FEATURES, run_smoke_seeds  # noqa: E402
from magma_frontier.value.coverage import greedy_coverage  # noqa: E402
from magma_frontier.value.utility import loto_utility  # noqa: E402

CORPUS = Path("data/toolathlon")
CACHE = Path("out/frontier_cache.npz")
COVERAGE_SEEDS = 10
EXPOSURE_SEEDS = 5
N_COMPONENTS = 128
K = 500
PERMUTATIONS = 20000


def _log(message: str, start: float) -> None:
    print(f"[{time.time() - start:6.0f}s] {message}", flush=True)


def build_cache() -> None:
    start = time.time()
    _log("parsing corpus", start)
    features, report_ = build_corpus(CORPUS, depth=1, exclude=TIMING_FEATURES)
    tenant_ids = np.asarray(features.tenant_ids)
    task_ids = np.asarray(features.task_ids)
    tenants = sorted({str(t) for t in tenant_ids})
    _log(f"parsed {report_.parsed}/{report_.total}, {features.X.shape[0]} sessions, "
         f"{len(tenants)} tenants", start)

    _log("fitting representation", start)
    train_idx, _ = next(
        GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        .split(features.X, tenant_ids, task_ids)
    )
    rep = fit_transform(features, train_idx, n_components=N_COMPONENTS, seed=0)

    _log(f"exposure: {EXPOSURE_SEEDS} supervised-attack seeds (the slow part)", start)
    recalls: dict[str, list[float]] = {}
    for smoke in run_smoke_seeds(features, seeds=tuple(range(EXPOSURE_SEEDS))):
        for tenant, recall in smoke.per_tenant_recall.items():
            recalls.setdefault(tenant, []).append(recall)
    _log("exposure done", start)

    _log("utility estimator (243 model fits)", start)
    utility = loto_utility(rep.Z, tenant_ids, task_ids, features.outcomes, seed=0)
    _log(f"utility done: accuracy {utility.baseline_accuracy:.4f} vs majority "
         f"{utility.majority_baseline:.4f}, AUC {utility.baseline_auc:.4f}", start)

    # Per-tenant scalars derived from the corpus, cached so `report` needs no corpus.
    Zn = rep.Z / np.maximum(np.linalg.norm(rep.Z, axis=1, keepdims=True), 1e-12)
    homogeneity, success, counts = [], [], []
    for tenant in tenants:
        rows = Zn[tenant_ids == tenant]
        centroid = rows.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        homogeneity.append(float((rows @ centroid).mean()))
        outcomes = [o for o, t in zip(features.outcomes, tenant_ids)
                    if str(t) == tenant and o is not None]
        success.append(float(np.mean(outcomes)) if outcomes else np.nan)
        counts.append(int((tenant_ids == tenant).sum()))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        Z=rep.Z,
        tenant_ids=tenant_ids,
        tenants=np.array(tenants),
        exposure=np.array([float(np.mean(recalls[t])) for t in tenants]),
        exposure_seed_counts=np.array([len(recalls[t]) for t in tenants]),
        utility_adjusted=np.array([utility.per_tenant_adjusted[t] for t in tenants]),
        utility_skill=np.array([utility.baseline_accuracy, utility.majority_baseline,
                                utility.baseline_auc]),
        homogeneity=np.array(homogeneity),
        success=np.array(success),
        row_counts=np.array(counts),
        parsed=np.array([report_.parsed, report_.total, report_.skipped]),
    )
    _log(f"cache written to {CACHE}", start)


def permutation_p(value: np.ndarray, exposure: np.ndarray, rho: float, rng) -> float:
    extreme = sum(
        abs(_spearman(value, rng.permutation(exposure))) >= abs(rho)
        for _ in range(PERMUTATIONS)
    )
    return (extreme + 1) / (PERMUTATIONS + 1)


def bootstrap_ci(value: np.ndarray, exposure: np.ndarray, rng, draws: int = 5000):
    """Resample TENANTS, not rows: the tenant is the unit of analysis."""
    rhos = []
    for _ in range(draws):
        idx = rng.integers(0, value.size, size=value.size)
        if np.unique(value[idx]).size >= 3:
            rhos.append(_spearman(value[idx], exposure[idx]))
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def report() -> None:
    start = time.time()
    if not CACHE.exists():
        sys.exit(f"no cache at {CACHE}; run `{sys.argv[0]} cache` first")
    data = np.load(CACHE)
    tenants = [str(t) for t in data["tenants"]]
    exposure = data["exposure"]
    parsed, total, skipped = data["parsed"]
    accuracy, majority, auc = data["utility_skill"]

    print(f"corpus: {parsed}/{total} records parsed, {skipped} skipped, "
          f"{len(tenants)} tenants")
    print(f"exposure seeds per tenant: min {data['exposure_seed_counts'].min()}, "
          f"max {data['exposure_seed_counts'].max()}\n")

    print("utility model skill check")
    print(f"  accuracy {accuracy:.4f}   majority baseline {majority:.4f}   AUC {auc:.4f}")
    skilled = accuracy > majority and auc >= 0.55
    print(f"  VERDICT: {'usable' if skilled else 'NO SKILL - this axis measures nothing'}")
    print(f"  utility delta vs exposure: rho "
          f"{_spearman(data['utility_adjusted'], exposure):+.4f} "
          f"(recorded to document the failure, not as a finding)\n")

    _log(f"coverage lift over {COVERAGE_SEEDS} seeds", start)
    Z, tenant_ids = data["Z"], data["tenant_ids"]
    lifts = np.array([
        [greedy_coverage(Z, tenant_ids, k=K, seed=s).per_tenant_lift[t] for t in tenants]
        for s in range(COVERAGE_SEEDS)
    ])
    mean_lift = lifts.mean(axis=0)
    per_seed = [_spearman(lifts[s], exposure) for s in range(COVERAGE_SEEDS)]
    pairwise = [_spearman(lifts[i], lifts[j])
                for i in range(COVERAGE_SEEDS) for j in range(i + 1, COVERAGE_SEEDS)]

    rng = np.random.default_rng(0)
    rho = _spearman(mean_lift, exposure)
    p = permutation_p(mean_lift, exposure, rho, rng)
    lo, hi = bootstrap_ci(mean_lift, exposure, rng)

    print(f"\ncoverage lift ({COVERAGE_SEEDS}-seed mean) vs exposure")
    print(f"  rho {rho:+.4f}   permutation p {p:.5f}   bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  per-seed rho: {min(per_seed):+.3f} to {max(per_seed):+.3f}, "
          f"negative in {sum(r < 0 for r in per_seed)}/{COVERAGE_SEEDS}")
    print(f"  lift test-retest reliability (mean pairwise rho): {np.mean(pairwise):+.3f}\n")

    print("exploratory, post hoc, not pre-registered")
    for label, vector in (("homogeneity", data["homogeneity"]),
                          ("task success rate", data["success"]),
                          ("row count", data["row_counts"].astype(float))):
        print(f"  {label:<18} vs exposure  rho {_spearman(vector, exposure):+.3f}"
              f"   vs mean lift  rho {_spearman(vector, mean_lift):+.3f}")

    print(f"\n{'tenant':<26}{'lift':>8}{'exposure':>10}{'homog':>8}{'success':>9}{'rows':>7}")
    for i in np.argsort(-exposure):
        print(f"{tenants[i]:<26}{mean_lift[i]:>8.3f}{exposure[i]:>10.3f}"
              f"{data['homogeneity'][i]:>8.3f}{data['success'][i]:>9.3f}"
              f"{data['row_counts'][i]:>7}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command == "cache":
        build_cache()
    elif command == "report":
        report()
    else:
        sys.exit("usage: frontier_analysis.py [cache|report]")
