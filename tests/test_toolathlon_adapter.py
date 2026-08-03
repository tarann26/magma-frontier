import json
from pathlib import Path

import pytest

from magma_frontier.adapters.toolathlon import (
    AdapterError,
    classify_tool_result,
    parse_line,
    tenant_and_run,
)
from magma_frontier.schema import validate_session

FIXTURE = Path(__file__).parent / "fixtures" / "toolathlon_sample.jsonl"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="fixture holds gated corpus records; rebuild with scripts/build_fixture.py",
)


def _lines():
    return [l for l in FIXTURE.read_text().splitlines() if l.strip()]


def test_tenant_and_run_splits_trailing_run_suffix():
    assert tenant_and_run("gpt-5-mini_1") == ("gpt-5-mini", "1")
    assert tenant_and_run("claude-4.5-sonnet-0929_3") == ("claude-4.5-sonnet-0929", "3")
    assert tenant_and_run("deepseek-3.2-thinking_2") == ("deepseek-3.2-thinking", "2")


def test_tenant_and_run_rejects_missing_suffix():
    with pytest.raises(AdapterError, match="run suffix"):
        tenant_and_run("gpt-5-mini")


def test_classify_tool_result():
    assert classify_tool_result("Error: file not found") == "error"
    assert classify_tool_result("Traceback (most recent call last):") == "error"
    assert classify_tool_result("FAILED to connect") == "error"
    assert classify_tool_result("2025-10-18") == "ok"
    assert classify_tool_result(None) == "ok"


def test_parses_fixture_records():
    sessions = [parse_line(l) for l in _lines()]
    assert len(sessions) == 2
    for s in sessions:
        validate_session(s)
        assert s.tenant_id == "gpt-5-mini"
        assert s.run_id == "1"
        assert s.task_id
        assert s.session_id == f"gpt-5-mini_1::{s.task_id}"
        assert len(s.spans) >= 1
        assert isinstance(s.outcome, bool)
        assert s.duration_s is not None and s.duration_s >= 0


def test_spans_are_ordered_tool_calls_with_namespaced_ids():
    session = parse_line(_lines()[0])
    assert [sp.order for sp in session.spans] == list(range(len(session.spans)))
    assert all(sp.kind == "tool" for sp in session.spans)
    assert all("-" in sp.tool_id for sp in session.spans)


def test_arg_arity_and_type_shape_have_no_values():
    session = parse_line(_lines()[0])
    for sp in session.spans:
        assert sp.arg_arity >= 0
        # type shape is a sorted list of JSON type names only, never argument values
        for token in sp.arg_type_shape.split(","):
            assert token in {"", "str", "int", "float", "bool", "list", "dict", "null"}


def test_orphaned_tool_call_is_missing_not_ok():
    """A call with no result message never returned; it must not read as success."""
    import json as _json
    record = _json.loads(_lines()[0])
    messages = _json.loads(record["messages"])
    stripped = [m for m in messages if m.get("role") != "tool"]
    record["messages"] = _json.dumps(stripped)
    session = parse_line(_json.dumps(record))
    assert all(sp.status == "missing" for sp in session.spans)
    assert not any(sp.status == "ok" for sp in session.spans)


def test_rejects_null_required_field():
    """Runs that never completed carry `"messages": null`; they are skips, not crashes."""
    import json as _json
    record = _json.loads(_lines()[0])
    record["messages"] = None
    with pytest.raises(AdapterError, match="messages"):
        parse_line(_json.dumps(record))


def test_rejects_malformed_line():
    with pytest.raises(AdapterError):
        parse_line("{not json")
