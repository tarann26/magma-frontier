# tests/test_confound.py
import numpy as np
import pytest

from magma_frontier.eval.confound import ConfoundReport, audit
from magma_frontier.features.extract import FeatureSet


def _fs(tenant_ids, task_ids):
    n = len(tenant_ids)
    return FeatureSet(
        X=np.zeros((n, 3)),
        feature_names=("n_steps", "error_rate", "uni:a"),
        tenant_ids=tuple(tenant_ids),
        task_ids=tuple(task_ids),
        session_ids=tuple(f"s{i}" for i in range(n)),
        outcomes=tuple(True for _ in range(n)),
        ngram_vocabulary=("uni:a",),
    )


def test_balanced_design_is_not_confounded():
    """Every tenant attempts every task: task identity says nothing about tenant."""
    tenants = [f"t{i % 3}" for i in range(60)]
    tasks = [f"k{i % 10}" for i in range(60)]
    report = audit(_fs(tenants, tasks))
    assert report.tenant_from_task_accuracy < report.task_chance + 0.15
    assert report.min_task_jaccard == pytest.approx(1.0)


def test_disjoint_task_sets_are_flagged():
    """Each tenant attempts its own tasks: task identity fully determines tenant."""
    tenants = [f"t{i // 20}" for i in range(60)]
    tasks = [f"k{i}" for i in range(60)]
    report = audit(_fs(tenants, tasks))
    assert report.tenant_from_task_accuracy > 0.9
    assert report.min_task_jaccard == pytest.approx(0.0)


def test_partial_overlap_lands_between():
    tenants = [f"t{i % 2}" for i in range(60)]
    tasks = [f"k{i % 10}" if i % 2 == 0 else f"k{5 + i % 10}" for i in range(60)]
    report = audit(_fs(tenants, tasks))
    assert 0.0 < report.min_task_jaccard < 1.0


def test_accuracy_is_row_weighted_over_a_genuine_majority_split():
    """Uneven task sizes with real (non-tied, non-singleton) majorities.

    Row-weighted argmax gives 0.7 here; a per-task unweighted mean would give 0.8125
    and a min-instead-of-max rule 0.5, so this fixture separates the correct rule from
    the plausible wrong ones. Balanced or singleton fixtures cannot.
    """
    tenants = ["t0"] * 5 + ["t1"] * 3 + ["t1"] * 2
    tasks = ["k0"] * 8 + ["k1"] * 2
    report = audit(_fs(tenants, tasks))
    assert report.tenant_from_task_accuracy == pytest.approx(0.7)
    assert report.task_chance == pytest.approx(0.5)
    assert report.median_sessions_per_task == pytest.approx(5.0)


def test_reports_median_sessions_per_task():
    tenants = [f"t{i % 3}" for i in range(60)]
    tasks = [f"k{i % 10}" for i in range(60)]
    report = audit(_fs(tenants, tasks))
    assert report.median_sessions_per_task == pytest.approx(6.0)


def test_reports_per_tenant_task_counts():
    tenants = ["t0"] * 10 + ["t1"] * 5
    tasks = [f"k{i % 4}" for i in range(15)]
    report = audit(_fs(tenants, tasks))
    assert report.tenant_task_counts["t0"] == 4
    assert report.tenant_task_counts["t1"] >= 1


def test_task_chance_is_the_majority_tenant_share():
    tenants = ["t0"] * 40 + ["t1"] * 20
    tasks = [f"k{i % 10}" for i in range(60)]
    report = audit(_fs(tenants, tasks))
    assert report.task_chance == pytest.approx(40 / 60)


def test_rejects_single_tenant():
    with pytest.raises(ValueError, match="at least two tenants"):
        audit(_fs(["t0"] * 10, [f"k{i}" for i in range(10)]))
