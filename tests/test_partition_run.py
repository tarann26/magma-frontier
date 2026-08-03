import numpy as np
import pytest

from magma_frontier.features.extract import FeatureSet
from magma_frontier.partition_run import PartitionRun, run_partition


def _fs(n_per_tenant=40, n_tenants=3, seed=0):
    rng = np.random.default_rng(seed)
    rows, tenants, tasks = [], [], []
    for t in range(n_tenants):
        centre = np.zeros(8)
        centre[t] = 5.0
        for i in range(n_per_tenant):
            rows.append(np.abs(centre + rng.normal(0, 0.3, 8)))
            tenants.append(f"t{t}")
            tasks.append(f"k{i % 10}")
    names = ("n_steps", "error_rate") + tuple(f"uni:tool{i}" for i in range(6))
    return FeatureSet(
        X=np.array(rows), feature_names=names,
        tenant_ids=tuple(tenants), task_ids=tuple(tasks),
        session_ids=tuple(f"s{i}" for i in range(len(rows))),
        outcomes=tuple(True for _ in rows),
        ngram_vocabulary=tuple(n for n in names if n.startswith("uni:")),
    )


def test_returns_both_confound_and_partition_results():
    run = run_partition(_fs(), n_components=4, seed=0)
    assert run.n_tenants == 3
    assert run.n == 120
    assert run.partition.ari > 0.5
    assert run.confound.min_task_jaccard == pytest.approx(1.0)


def test_reports_explained_variance_of_the_representation():
    run = run_partition(_fs(), n_components=4, seed=0)
    assert 0.0 < run.explained_variance <= 1.0
    assert run.n_components == 4


def test_reports_held_out_ari_alongside_the_whole_pool():
    """The whole-pool score covers rows the SVD basis was fitted on; this is the check."""
    run = run_partition(_fs(), n_components=4, seed=0)
    assert 0 < run.n_heldout < run.n
    assert -1.0 <= run.ari_heldout <= 1.0


def test_is_deterministic_under_seed():
    a = run_partition(_fs(), n_components=4, seed=2)
    b = run_partition(_fs(), n_components=4, seed=2)
    assert a.partition.ari == b.partition.ari
    assert a.ari_heldout == b.ari_heldout
