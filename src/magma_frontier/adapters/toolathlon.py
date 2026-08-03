"""Adapter for hkust-nlp/Toolathlon-Trajectories JSONL.

Every nested field in the source is a JSON-encoded string, so each record needs a
second parse pass. Text never leaves this module except as tool identifiers, which
`features/` maps through a taxonomy.
"""

import json
import re
from datetime import datetime

from magma_frontier.schema import RawSession, Span

_RUN_SUFFIX = re.compile(r"^(?P<model>.+)_(?P<run>\d+)$")
_ERROR_MARKERS = ("error", "traceback", "failed", "fail:", "exception", "denied")
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


class AdapterError(ValueError):
    """Raised when a source record cannot be parsed into the canonical schema."""


def tenant_and_run(modelname_run: str) -> tuple[str, str]:
    """Split "gpt-5-mini_1" into ("gpt-5-mini", "1")."""
    match = _RUN_SUFFIX.match(modelname_run)
    if match is None:
        raise AdapterError(f"{modelname_run!r} has no trailing _<digits> run suffix")
    return match.group("model"), match.group("run")


def classify_tool_result(content: str | None) -> str:
    """Reduce a tool result to ok/error. Only the label survives, never the text."""
    if not content:
        return "ok"
    head = content[:200].lower()
    return "error" if any(marker in head for marker in _ERROR_MARKERS) else "ok"


def _arg_shape(arguments: str) -> tuple[int, str]:
    """Return (arity, type-shape) for a JSON argument blob. Values are discarded."""
    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return 0, ""
    if not isinstance(parsed, dict):
        return 0, ""
    names = {
        str: "str", bool: "bool", int: "int", float: "float",
        list: "list", dict: "dict", type(None): "null",
    }
    # bool before int matters: bool is a subclass of int in Python.
    shape = sorted(
        "bool" if isinstance(v, bool) else names.get(type(v), "null")
        for v in parsed.values()
    )
    return len(parsed), ",".join(shape)


def _duration_s(record: dict) -> int | None:
    start, end = record.get("initial_run_time"), record.get("completion_time")
    if not start or not end:
        return None
    try:
        delta = datetime.strptime(end, _TIME_FMT) - datetime.strptime(start, _TIME_FMT)
    except ValueError:
        return None
    return max(0, int(delta.total_seconds()))


def parse_line(raw: str) -> RawSession:
    """Parse one JSONL line into a RawSession."""
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"line is not valid JSON: {exc}") from exc

    # Presence is not enough: ~1.6% of corpus records carry `"messages": null`
    # for runs that never completed. Those must be counted as skips, not crash the
    # load, so require the field to actually be a string.
    for required in ("modelname_run", "task_name", "task_status", "messages"):
        if not isinstance(record.get(required), str):
            found = type(record.get(required)).__name__
            raise AdapterError(
                f"record field {required!r} is {found}, expected str"
            )

    tenant, run = tenant_and_run(record["modelname_run"])
    task_id = record["task_name"]

    try:
        status = json.loads(record["task_status"])
        messages = json.loads(record["messages"])
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{tenant}/{task_id}: nested JSON unparseable: {exc}") from exc

    outcome = status.get("evaluation")
    outcome = outcome if isinstance(outcome, bool) else None

    # Pair each assistant tool_call with the tool message that reports its result.
    # A call with no matching result never returned (truncated or crashed session)
    # and is recorded as "missing", not silently as a success.
    results: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            results[message["tool_call_id"]] = classify_tool_result(message.get("content"))

    spans: list[Span] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            arity, shape = _arg_shape(function.get("arguments") or "")
            spans.append(
                Span(
                    tool_id=name,
                    kind="tool",
                    order=len(spans),
                    status=results.get(call.get("id", ""), "missing"),
                    arg_arity=arity,
                    arg_type_shape=shape,
                )
            )

    if not spans:
        raise AdapterError(f"{tenant}/{task_id}: no tool calls in session")

    return RawSession(
        session_id=f"{record['modelname_run']}::{task_id}",
        tenant_id=tenant,
        run_id=run,
        task_id=task_id,
        outcome=outcome,
        duration_s=_duration_s(record),
        spans=tuple(spans),
    )
