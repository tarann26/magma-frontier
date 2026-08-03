import numpy as np
import pytest

from magma_frontier.adapters.toolathlon import parse_line
from magma_frontier.features.extract import FeatureSet, extract
from magma_frontier.schema import RawSession, Span
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "toolathlon_sample.jsonl"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="fixture holds gated corpus records; rebuild with scripts/build_fixture.py",
)


def _fixture_sessions():
    return [parse_line(l) for l in FIXTURE.read_text().splitlines() if l.strip()]


def _synthetic(tool_ids, tenant="t0", task="k0", outcome=True):
    spans = tuple(
        Span(tool_id=t, kind="tool", order=i, status="ok", arg_arity=1, arg_type_shape="str")
        for i, t in enumerate(tool_ids)
    )
    return RawSession(session_id=f"{tenant}::{task}", tenant_id=tenant, run_id="1",
                      task_id=task, outcome=outcome, duration_s=10, spans=spans)


def test_matrix_is_numeric_and_finite():
    fs = extract(_fixture_sessions())
    assert fs.X.dtype == np.float64
    assert np.isfinite(fs.X).all()
    assert fs.X.shape[0] == 2
    assert fs.X.shape[1] == len(fs.feature_names)


def test_matrix_holds_no_string_or_object_data():
    fs = extract(_fixture_sessions())
    assert fs.X.dtype.kind == "f"


def test_malformed_tool_id_cannot_reach_feature_names():
    """A tool id carrying prose or control characters must not become a column name."""
    prose = "google_\n<ctrl94>thought\nThe user wants to recall a product named Widget"
    fs = extract([_synthetic([prose, "github-x"])], depth=2)
    assert all("thought" not in name for name in fs.feature_names)
    assert all("\n" not in name for name in fs.feature_names)
    assert "uni:malformed" in fs.feature_names


def test_overlong_tool_id_is_collapsed():
    fs = extract([_synthetic(["a" * 200])], depth=2)
    assert "uni:malformed" in fs.feature_names
    assert all(len(name) < 100 for name in fs.feature_names)


def test_exclude_drops_named_features():
    fs = extract([_synthetic(["a-x", "b-y"])], exclude=("duration_s", "steps_per_minute"))
    assert "duration_s" not in fs.feature_names
    assert "steps_per_minute" not in fs.feature_names
    assert "n_steps" in fs.feature_names
    assert fs.X.shape[1] == len(fs.feature_names)


def test_exclude_does_not_shift_remaining_columns():
    plain = extract([_synthetic(["a-x", "b-y"])])
    trimmed = extract([_synthetic(["a-x", "b-y"])], exclude=("duration_s",))
    for name in trimmed.feature_names:
        assert trimmed.X[0, trimmed.feature_names.index(name)] == pytest.approx(
            plain.X[0, plain.feature_names.index(name)]
        )


def test_labels_travel_beside_features_not_inside():
    fs = extract(_fixture_sessions())
    assert len(fs.tenant_ids) == fs.X.shape[0]
    assert len(fs.task_ids) == fs.X.shape[0]
    assert all("tenant" not in name for name in fs.feature_names)


def test_step_count_feature():
    fs = extract([_synthetic(["filesystem-read_file"] * 5)])
    idx = fs.feature_names.index("n_steps")
    assert fs.X[0, idx] == 5.0


def test_repeat_run_feature_counts_consecutive_repeats():
    fs = extract([_synthetic(["a-x", "a-x", "a-x", "b-y"])])
    idx = fs.feature_names.index("max_repeat_run")
    assert fs.X[0, idx] == 3.0


def test_error_rate_feature():
    spans = (
        Span(tool_id="a-x", kind="tool", order=0, status="error", arg_arity=0, arg_type_shape=""),
        Span(tool_id="a-x", kind="tool", order=1, status="ok", arg_arity=0, arg_type_shape=""),
    )
    session = RawSession(session_id="s", tenant_id="t", run_id="1", task_id="k",
                         outcome=False, duration_s=1, spans=spans)
    fs = extract([session])
    idx = fs.feature_names.index("error_rate")
    assert fs.X[0, idx] == pytest.approx(0.5)


def test_orphan_rate_is_separate_from_error_rate():
    spans = (
        Span(tool_id="a-x", kind="tool", order=0, status="error", arg_arity=0, arg_type_shape=""),
        Span(tool_id="a-x", kind="tool", order=1, status="missing", arg_arity=0, arg_type_shape=""),
        Span(tool_id="a-x", kind="tool", order=2, status="ok", arg_arity=0, arg_type_shape=""),
        Span(tool_id="a-x", kind="tool", order=3, status="ok", arg_arity=0, arg_type_shape=""),
    )
    session = RawSession(session_id="s", tenant_id="t", run_id="1", task_id="k",
                         outcome=False, duration_s=1, spans=spans)
    fs = extract([session])
    assert fs.X[0, fs.feature_names.index("error_rate")] == pytest.approx(0.25)
    assert fs.X[0, fs.feature_names.index("orphan_rate")] == pytest.approx(0.25)


def test_steps_per_minute_is_a_real_rate_for_sub_minute_sessions():
    """Flooring the denominator at a minute would collapse this to n_steps."""
    session = _synthetic(["a-x"] * 5)
    session = RawSession(session_id=session.session_id, tenant_id=session.tenant_id,
                         run_id=session.run_id, task_id=session.task_id,
                         outcome=session.outcome, duration_s=6, spans=session.spans)
    fs = extract([session])
    assert fs.X[0, fs.feature_names.index("steps_per_minute")] == pytest.approx(50.0)


def test_missing_duration_is_distinguishable_from_zero_duration():
    spans = (Span(tool_id="a-x", kind="tool", order=0, status="ok",
                  arg_arity=0, arg_type_shape=""),)
    unknown = RawSession(session_id="s1", tenant_id="t", run_id="1", task_id="k",
                         outcome=True, duration_s=None, spans=spans)
    instant = RawSession(session_id="s2", tenant_id="t", run_id="1", task_id="k",
                         outcome=True, duration_s=0, spans=spans)
    fs = extract([unknown, instant])
    col = fs.feature_names.index("has_duration")
    assert fs.X[0, col] == 0.0
    assert fs.X[1, col] == 1.0


def test_zero_span_session_raises_a_clear_error():
    empty = RawSession(session_id="s", tenant_id="t", run_id="1", task_id="k",
                       outcome=None, duration_s=1, spans=())
    with pytest.raises(ValueError, match="no spans"):
        extract([empty])


def test_ngram_features_use_taxonomy_depth():
    shallow = extract([_synthetic(["github-a", "github-b"])], depth=1)
    assert "uni:github" in shallow.feature_names
    deep = extract([_synthetic(["github-a", "github-b"])], depth=2)
    assert "uni:github-a" in deep.feature_names


def test_vocabulary_is_shared_across_sessions():
    fs = extract([_synthetic(["github-a"], tenant="t0"),
                  _synthetic(["filesystem-b"], tenant="t1")], depth=2)
    assert "uni:github-a" in fs.feature_names
    assert "uni:filesystem-b" in fs.feature_names
    assert fs.X.shape[0] == 2


def test_excluded_fields_absent():
    fs = extract(_fixture_sessions())
    banned = ("token", "cost", "timestamp", "arg_value")
    for name in fs.feature_names:
        assert not any(b in name.lower() for b in banned)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one session"):
        extract([])


def test_safe_token_rejects_a_trailing_newline():
    """`$` matches before a trailing newline; fullmatch does not."""
    fs = extract([_synthetic(["github-x\n", "github-y"])], depth=2)
    assert all("\n" not in name for name in fs.feature_names)
    assert "uni:malformed" in fs.feature_names


def test_derived_vocabulary_is_exposed():
    fs = extract([_synthetic(["a-x", "b-y"])], depth=2)
    assert "uni:a-x" in fs.ngram_vocabulary
    assert "uni:b-y" in fs.ngram_vocabulary
    assert all(n in fs.feature_names for n in fs.ngram_vocabulary)


def test_supplied_vocabulary_is_used_verbatim():
    vocab = ("uni:a-x", "uni:never-seen")
    fs = extract([_synthetic(["a-x", "b-y"])], depth=2, vocabulary=vocab)
    assert fs.ngram_vocabulary == vocab
    assert "uni:b-y" not in fs.feature_names
    assert "uni:never-seen" in fs.feature_names


def test_supplied_vocabulary_zero_fills_unseen_columns():
    vocab = ("uni:a-x", "uni:never-seen")
    fs = extract([_synthetic(["a-x"])], depth=2, vocabulary=vocab)
    assert fs.X[0, fs.feature_names.index("uni:never-seen")] == 0.0
    assert fs.X[0, fs.feature_names.index("uni:a-x")] > 0.0


def test_supplied_vocabulary_still_honours_exclude():
    vocab = ("uni:a-x",)
    fs = extract([_synthetic(["a-x"])], depth=2, vocabulary=vocab, exclude=("duration_s",))
    assert "duration_s" not in fs.feature_names
    assert "uni:a-x" in fs.feature_names
