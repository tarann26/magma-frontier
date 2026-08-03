"""Command line entry points: download, build, smoke."""

import argparse
from pathlib import Path

import numpy as np

from magma_frontier.corpus import build_corpus
from magma_frontier.download import download_corpus
from magma_frontier.frontier import run_frontier
from magma_frontier.partition_run import run_partition
from magma_frontier.smoke import TIMING_FEATURES, run_smoke, run_smoke_seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="magma_frontier")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="fetch the gated Toolathlon corpus")
    dl.add_argument("--dest", type=Path, default=Path("data/toolathlon"))
    dl.add_argument("--files", nargs="*", default=None)

    sm = sub.add_parser("smoke", help="build features and run the separability check")
    sm.add_argument("--data", type=Path, default=Path("data/toolathlon"))
    sm.add_argument("--depth", type=int, default=1)
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--exclude-timing", action="store_true",
                    help="drop duration_s, steps_per_minute and has_duration")
    sm.add_argument("--seeds", type=int, default=1,
                    help="number of seeds to run (0..N-1); reports mean and range")

    pt = sub.add_parser("partition", help="unsupervised partition attack on a pooled corpus")
    pt.add_argument("--data", type=Path, default=Path("data/toolathlon"))
    pt.add_argument("--depth", type=int, default=1)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--components", type=int, default=128)
    pt.add_argument("--exclude-timing", action="store_true",
                    help="drop duration_s, steps_per_minute and has_duration")

    fr = sub.add_parser("frontier", help="per-tenant value against per-tenant exposure")
    fr.add_argument("--data", type=Path, default=Path("data/toolathlon"))
    fr.add_argument("--depth", type=int, default=1)
    fr.add_argument("--seed", type=int, default=0)
    fr.add_argument("--components", type=int, default=128)
    fr.add_argument("--k", type=int, default=500)
    fr.add_argument("--exposure-seeds", type=int, default=5)
    fr.add_argument("--exclude-timing", action="store_true",
                    help="drop duration_s, steps_per_minute and has_duration")

    args = parser.parse_args(argv)

    if args.command == "download":
        paths = download_corpus(args.dest, args.files)
        print(f"downloaded {len(paths)} files to {args.dest}")
        return 0

    if args.command == "partition":
        exclude = TIMING_FEATURES if args.exclude_timing else ()
        features, report = build_corpus(args.data, depth=args.depth, exclude=exclude)
        print(f"parsed {report.parsed}/{report.total} records "
              f"({report.skipped} skipped: {report.reasons or 'none'})")
        run = run_partition(features, n_components=args.components, seed=args.seed)
        print(f"sessions: {run.n}  tenants: {run.n_tenants}")
        print(f"representation: {run.n_components} components, "
              f"{run.explained_variance:.3f} explained variance")
        print("confound audit:")
        print(f"  tenant predicted from task alone: "
              f"{run.confound.tenant_from_task_accuracy:.3f} "
              f"(majority baseline {run.confound.task_chance:.3f}, "
              f"median {run.confound.median_sessions_per_task:.0f} sessions/task)")
        print(f"  task-set jaccard: min {run.confound.min_task_jaccard:.3f}, "
              f"mean {run.confound.mean_task_jaccard:.3f}")
        print("partition attack:")
        print(f"  ARI: {run.partition.ari:.3f}   AMI: {run.partition.ami:.3f}")
        print(f"  ARI on held-out tasks only: {run.ari_heldout:.3f} "
              f"({run.n_heldout} sessions)")
        print(f"  clusters: {run.partition.n_clusters}   "
              f"largest cluster share: {run.partition.largest_cluster_share:.3f}")
        print("  per-tenant purity:")
        for tenant, value in sorted(run.partition.per_tenant_purity.items(),
                                    key=lambda kv: -kv[1]):
            print(f"    {tenant}: {value:.3f}")
        return 0

    if args.command == "frontier":
        exclude = TIMING_FEATURES if args.exclude_timing else ()
        features, report = build_corpus(args.data, depth=args.depth, exclude=exclude)
        print(f"parsed {report.parsed}/{report.total} records "
              f"({report.skipped} skipped: {report.reasons or 'none'})")
        run = run_frontier(features, n_components=args.components, k=args.k,
                           exposure_seeds=args.exposure_seeds, seed=args.seed)
        f = run.frontier
        print(f"tenants: {f.n}   coreset k: {run.coverage_k}")
        print(f"utility model: accuracy {run.utility_baseline:.3f}  "
              f"majority baseline {run.utility_majority:.3f}  "
              f"AUC {run.utility_auc:.3f}")
        if run.utility_baseline <= run.utility_majority or run.utility_auc < 0.55:
            print("  WARNING: the utility model has no skill on held-out tasks. "
                  "Its per-tenant deltas are differences between unskilled models "
                  "and the utility axis measures nothing.")
        if run.dropped_tenants:
            print(f"dropped (utility adjustment unavailable): "
                  f"{', '.join(run.dropped_tenants)}")
        print(f"coverage lift vs exposure:  rho {f.rho_coverage:+.3f}  p {f.p_coverage:.4f}")
        print(f"utility delta vs exposure:  rho {f.rho_utility:+.3f}  p {f.p_utility:.4f}")
        print(f"{'tenant':<26}{'cov lift':>10}{'util adj':>10}{'exposure':>10}")
        order = np.argsort(-f.exposure)
        for i in order:
            print(f"{f.tenants[i]:<26}{f.coverage_lift[i]:>10.3f}"
                  f"{f.utility_adjusted[i]:>+10.4f}{f.exposure[i]:>10.3f}")
        return 0

    exclude = TIMING_FEATURES if args.exclude_timing else ()
    features, report = build_corpus(args.data, depth=args.depth, exclude=exclude)
    print(f"parsed {report.parsed}/{report.total} records "
          f"({report.skipped} skipped: {report.reasons or 'none'})")
    print(f"features: {features.X.shape[0]} sessions x {features.X.shape[1]} columns")
    print(f"regime: {'structure-only (timing excluded)' if exclude else 'all features'}")

    if args.seeds > 1:
        _report_seeds(run_smoke_seeds(features, seeds=range(args.seeds)))
    else:
        _report_single(run_smoke(features, seed=args.seed))
    return 0


def _spread(values: list[float]) -> str:
    """Mean followed by the observed min-max range."""
    return f"{sum(values) / len(values):.3f} [{min(values):.3f}-{max(values):.3f}]"


def _report_single(result) -> None:
    print(f"tenants: {result.n_tenants}  chance: {result.chance:.3f}  "
          f"majority baseline: {result.majority_baseline:.3f}")
    print(f"accuracy: {result.accuracy:.3f}")
    print(f"shuffled-label control: {result.shuffled_accuracy:.3f}")
    for fpr, tpr in sorted(result.tpr_at_fpr.items(), reverse=True):
        print(f"TPR @ {fpr:.1%} FPR: {tpr:.3f}  "
              f"(measured null: {result.null_tpr_at_fpr[fpr]:.3f})")
    print("per-tenant recall:")
    for tenant, recall in sorted(result.per_tenant_recall.items(),
                                 key=lambda kv: kv[1], reverse=True):
        print(f"  {tenant}: {recall:.3f}")


def _report_seeds(results: list) -> None:
    first = results[0]
    print(f"seeds: {len(results)}  tenants: {first.n_tenants}  "
          f"chance: {first.chance:.3f}")
    print(f"majority baseline: {_spread([r.majority_baseline for r in results])}")
    print(f"accuracy: {_spread([r.accuracy for r in results])}")
    print(f"shuffled-label control: {_spread([r.shuffled_accuracy for r in results])}")
    for fpr in sorted(first.tpr_at_fpr, reverse=True):
        print(f"TPR @ {fpr:.1%} FPR: {_spread([r.tpr_at_fpr[fpr] for r in results])}  "
              f"(measured null: {_spread([r.null_tpr_at_fpr[fpr] for r in results])})")

    per_tenant: dict[str, list[float]] = {}
    for result in results:
        for tenant, recall in result.per_tenant_recall.items():
            per_tenant.setdefault(tenant, []).append(recall)
    print("per-tenant recall (mean over seeds in which the tenant reached the test fold):")
    for tenant, recalls in sorted(per_tenant.items(),
                                  key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True):
        print(f"  {tenant}: {_spread(recalls)}  (n_seeds={len(recalls)})")


if __name__ == "__main__":
    raise SystemExit(main())
