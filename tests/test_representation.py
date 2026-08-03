import numpy as np
import pytest

from magma_frontier.embed.representation import Representation, fit_transform
from magma_frontier.features.extract import FeatureSet


def _fs(n=60, d=40, seed=0):
    rng = np.random.default_rng(seed)
    names = tuple(f"n_steps" if i == 0 else f"uni:tool{i}" for i in range(d))
    return FeatureSet(
        X=np.abs(rng.normal(size=(n, d))),
        feature_names=names,
        tenant_ids=tuple(f"t{i % 3}" for i in range(n)),
        task_ids=tuple(f"k{i % 10}" for i in range(n)),
        session_ids=tuple(f"s{i}" for i in range(n)),
        outcomes=tuple(True for _ in range(n)),
        ngram_vocabulary=tuple(x for x in names if x.startswith("uni:")),
    )


def test_returns_requested_component_count():
    fs = _fs()
    rep = fit_transform(fs, np.arange(40), n_components=8, seed=0)
    assert rep.Z.shape == (60, 8)
    assert rep.n_components == 8


def test_transforms_every_row_including_held_out():
    fs = _fs()
    rep = fit_transform(fs, np.arange(40), n_components=8, seed=0)
    assert np.isfinite(rep.Z).all()
    assert not np.allclose(rep.Z[40:], 0.0)


def test_fit_uses_training_rows_only():
    """Corrupting held-out rows must not change the basis the training rows map to."""
    fs = _fs()
    train = np.arange(40)
    clean = fit_transform(fs, train, n_components=8, seed=0)

    poisoned = FeatureSet(
        X=fs.X.copy(), feature_names=fs.feature_names, tenant_ids=fs.tenant_ids,
        task_ids=fs.task_ids, session_ids=fs.session_ids, outcomes=fs.outcomes,
        ngram_vocabulary=fs.ngram_vocabulary,
    )
    poisoned.X[40:] *= 1000.0
    dirty = fit_transform(poisoned, train, n_components=8, seed=0)

    assert np.allclose(np.abs(clean.Z[:40]), np.abs(dirty.Z[:40]))


def test_is_deterministic_under_seed():
    fs = _fs()
    a = fit_transform(fs, np.arange(40), n_components=8, seed=3)
    b = fit_transform(fs, np.arange(40), n_components=8, seed=3)
    assert np.allclose(a.Z, b.Z)


def test_reports_explained_variance():
    fs = _fs()
    rep = fit_transform(fs, np.arange(40), n_components=8, seed=0)
    assert 0.0 < rep.explained_variance <= 1.0


def test_caps_components_at_available_rank():
    fs = _fs(n=20, d=12)
    rep = fit_transform(fs, np.arange(10), n_components=128, seed=0)
    assert rep.n_components < 12
    assert rep.Z.shape[1] == rep.n_components


def test_rejects_empty_training_index():
    fs = _fs()
    with pytest.raises(ValueError, match="training rows"):
        fit_transform(fs, np.array([], dtype=int), n_components=8, seed=0)


def test_rejects_single_training_row():
    """One row gives zero variance, and sklearn returns NaN rather than raising."""
    fs = _fs()
    with pytest.raises(ValueError, match="at least 2 training rows"):
        fit_transform(fs, np.array([0], dtype=int), n_components=8, seed=0)


def test_explained_variance_is_never_nan():
    fs = _fs()
    rep = fit_transform(fs, np.arange(2), n_components=8, seed=0)
    assert np.isfinite(rep.explained_variance)
