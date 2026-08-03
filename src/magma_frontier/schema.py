"""Canonical trace types. Everything downstream of adapters speaks this."""

from dataclasses import dataclass

VALID_KINDS = frozenset({"llm", "tool", "agent"})
VALID_STATUSES = frozenset({"ok", "error", "missing"})


class SchemaError(ValueError):
    """Raised when a parsed session violates the canonical schema."""


@dataclass(frozen=True, slots=True)
class Span:
    tool_id: str
    kind: str
    order: int
    status: str
    arg_arity: int
    arg_type_shape: str


@dataclass(frozen=True, slots=True)
class RawSession:
    session_id: str
    tenant_id: str
    run_id: str
    task_id: str
    outcome: bool | None
    duration_s: int | None
    spans: tuple[Span, ...]


def validate_session(session: RawSession) -> None:
    """Raise SchemaError if the session is malformed. Returns None on success."""
    if not session.spans:
        raise SchemaError(f"{session.session_id}: must have at least one span")
    for span in session.spans:
        if span.kind not in VALID_KINDS:
            raise SchemaError(
                f"{session.session_id}: bad span kind {span.kind!r}, "
                f"expected one of {sorted(VALID_KINDS)}"
            )
        if span.status not in VALID_STATUSES:
            raise SchemaError(
                f"{session.session_id}: bad span status {span.status!r}, "
                f"expected one of {sorted(VALID_STATUSES)}"
            )
        if span.arg_arity < 0:
            raise SchemaError(
                f"{session.session_id}: arg_arity must be >= 0, got {span.arg_arity}"
            )
