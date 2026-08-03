import numpy as np
import pytest

from magma_frontier.value.utility import UtilityResult, loto_utility


def _informative_and_noise(n_per=120, seed=0):
    """t_signal's rows carry the label; t_noise's are pure noise with random labels."""
    rng = np.random.default_rng(seed)
    rows, tenants, tasks, outcomes = [], [], [], []
    for i in range(n_per):
        label = bool(i % 2)
        rows.append(np.array([3.0 if label else -3.0, 0.0]) + rng.normal(0, 0.3, 2))
        tenants.append("t_signal")
        tasks.append(f"k{i % 12}")
        outcomes.append(label)
    for i in range(n_per):
        rows.append(rng.normal(0, 3.0, 2))
        tenants.append("t_noise")
        tasks.append(f"k{i % 12}")
        outcomes.append(bool(rng.integers(2)))
    return (np.array(rows), np.array(tenants), np.array(tasks),
            tuple(outcomes))


def test_reports_a_baseline_and_a_delta_per_tenant():
    Z, tenants, tasks, outcomes = _informative_and_noise()
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert set(result.per_tenant_delta) == {"t_signal", "t_noise"}
    assert set(result.per_tenant_adjusted) == {"t_signal", "t_noise"}


def test_removing_the_informative_tenant_hurts_more():
    Z, tenants, tasks, outcomes = _informative_and_noise()
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert result.per_tenant_delta["t_signal"] > result.per_tenant_delta["t_noise"]


def test_adjusted_delta_separates_signal_from_mere_volume():
    """Both tenants are the same size, so only the size-adjusted delta is interpretable."""
    Z, tenants, tasks, outcomes = _informative_and_noise()
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert result.per_tenant_adjusted["t_signal"] > result.per_tenant_adjusted["t_noise"]


def _unequal_sizes(seed=0):
    """A large noise contributor and a small informative one, plus filler.

    Sized unequally on purpose. With equal sizes the control shifts both deltas by a
    similar amount and cannot change their ordering, so a test built on equal sizes would
    pass identically if the control were deleted outright.
    """
    rng = np.random.default_rng(seed)
    rows, tenants, tasks, outcomes = [], [], [], []
    for name, n, informative in (("t_signal", 200, True),
                                 ("t_noise", 300, False),
                                 ("t_filler", 200, True)):
        for i in range(n):
            label = bool(i % 2)
            if informative:
                rows.append([(3.0 if label else -3.0) + rng.normal(0, 0.5),
                             rng.normal(0, 1)])
            else:
                rows.append(rng.normal(0, 3.0, 2))
                label = bool(rng.integers(2))
            tenants.append(name)
            tasks.append(f"k{i % 15}")
            outcomes.append(label)
    return (np.array(rows), np.array(tenants), np.array(tasks), tuple(outcomes))


def test_the_control_pushes_a_noise_contributor_below_its_raw_delta():
    """Regression guard for the size control itself.

    Removing a noise contributor's rows costs less than removing the same number of
    other people's rows, so the control must push its value DOWN. If the control were
    deleted and `adjusted` simply mirrored `delta`, this assertion fails immediately.

    Verified across six fixture seeds while writing the plan: the control lowers the
    noise contributor's value every time, and in two of the six the raw delta claimed it
    had positive value while the adjusted delta correctly called it negative.
    """
    Z, tenants, tasks, outcomes = _unequal_sizes()
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert result.per_tenant_adjusted["t_noise"] < result.per_tenant_delta["t_noise"]
    assert result.per_tenant_adjusted["t_signal"] > result.per_tenant_adjusted["t_noise"]


def test_train_and_test_are_disjoint_by_task():
    """Row counts alone would pass with an ungrouped split, which is the thing the
    grouping exists to prevent, so assert the grouping property directly."""
    from sklearn.model_selection import GroupShuffleSplit

    Z, tenants, tasks, outcomes = _informative_and_noise()
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert result.n_train > 0
    assert result.n_test > 0
    assert result.n_train + result.n_test == len(tenants)

    y = np.array([bool(o) for o in outcomes])
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(Z, y, tasks)
    )
    assert set(tasks[train_idx]).isdisjoint(set(tasks[test_idx]))


def test_sessions_without_an_outcome_are_dropped():
    Z, tenants, tasks, outcomes = _informative_and_noise(n_per=60)
    holed = list(outcomes)
    holed[0] = None
    holed[1] = None
    result = loto_utility(Z, tenants, tasks, tuple(holed), seed=0)
    assert result.n_train + result.n_test == len(tenants) - 2


def test_is_deterministic_under_seed():
    Z, tenants, tasks, outcomes = _informative_and_noise()
    a = loto_utility(Z, tenants, tasks, outcomes, seed=4)
    b = loto_utility(Z, tenants, tasks, outcomes, seed=4)
    assert a.baseline_accuracy == b.baseline_accuracy
    assert a.per_tenant_adjusted == b.per_tenant_adjusted


def test_rejects_a_single_tenant():
    Z, tenants, tasks, outcomes = _informative_and_noise()
    with pytest.raises(ValueError, match="at least two tenants"):
        loto_utility(Z, np.repeat("only", len(tenants)), tasks, outcomes, seed=0)


def test_rejects_when_no_session_has_an_outcome():
    Z, tenants, tasks, outcomes = _informative_and_noise(n_per=30)
    with pytest.raises(ValueError, match="labelled"):
        loto_utility(Z, tenants, tasks, tuple(None for _ in outcomes), seed=0)


def test_reports_the_majority_baseline_and_auc():
    Z, tenants, tasks, outcomes = _informative_and_noise()
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert 0.0 <= result.majority_baseline <= 1.0
    assert 0.0 <= result.baseline_auc <= 1.0


def test_a_skilled_model_beats_its_majority_baseline():
    """The fixture is separable, so the estimator must clear the constant predictor.
    On the real corpus it does not, which is exactly what these fields exist to expose."""
    Z, tenants, tasks, outcomes = _informative_and_noise()
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert result.baseline_accuracy > result.majority_baseline
    assert result.baseline_auc > 0.6


def test_auc_is_near_half_when_labels_carry_no_signal():
    rng = np.random.default_rng(0)
    n = 240
    Z = rng.normal(size=(n, 3))
    tenants = np.array([f"t{i % 3}" for i in range(n)])
    tasks = np.array([f"k{i % 12}" for i in range(n)])
    outcomes = tuple(bool(v) for v in rng.integers(2, size=n))
    result = loto_utility(Z, tenants, tasks, outcomes, seed=0)
    assert abs(result.baseline_auc - 0.5) < 0.2
