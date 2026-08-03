from dataclasses import FrozenInstanceError

import pytest

from magma_frontier.schema import RawSession, SchemaError, Span, validate_session


def _span(**over):
    base = dict(tool_id="filesystem-read_file", kind="tool", order=0,
                status="ok", arg_arity=1, arg_type_shape="str")
    base.update(over)
    return Span(**base)


def _session(**over):
    base = dict(session_id="gpt-5-mini_1::train-ticket-plan", tenant_id="gpt-5-mini",
                run_id="1", task_id="train-ticket-plan", outcome=True,
                duration_s=144, spans=(_span(),))
    base.update(over)
    return RawSession(**base)


def test_valid_session_passes():
    validate_session(_session())


def test_rejects_unknown_span_kind():
    with pytest.raises(SchemaError, match="kind"):
        validate_session(_session(spans=(_span(kind="wizard"),)))


def test_rejects_unknown_status():
    with pytest.raises(SchemaError, match="status"):
        validate_session(_session(spans=(_span(status="perhaps"),)))


def test_rejects_empty_spans():
    with pytest.raises(SchemaError, match="at least one span"):
        validate_session(_session(spans=()))


def test_rejects_negative_arity():
    with pytest.raises(SchemaError, match="arg_arity"):
        validate_session(_session(spans=(_span(arg_arity=-1),)))


def test_session_is_frozen():
    s = _session()
    with pytest.raises(FrozenInstanceError):
        s.tenant_id = "other"
