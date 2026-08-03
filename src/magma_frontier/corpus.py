"""Corpus loading with explicit skip accounting."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from magma_frontier.adapters.toolathlon import AdapterError, parse_line
from magma_frontier.features.extract import FeatureSet, extract
from magma_frontier.schema import RawSession, SchemaError, validate_session


@dataclass(frozen=True, slots=True)
class SkipReport:
    total: int
    parsed: int
    skipped: int
    reasons: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "SkipReport") -> "SkipReport":
        reasons = Counter(self.reasons) + Counter(other.reasons)
        return SkipReport(
            total=self.total + other.total,
            parsed=self.parsed + other.parsed,
            skipped=self.skipped + other.skipped,
            reasons=dict(reasons),
        )


def load_sessions(path: Path) -> tuple[list[RawSession], SkipReport]:
    """Parse one JSONL file. Malformed records are counted, never silently dropped."""
    sessions: list[RawSession] = []
    reasons: Counter[str] = Counter()
    total = 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            session = parse_line(line)
            validate_session(session)
        except (AdapterError, SchemaError) as exc:
            reasons[type(exc).__name__] += 1
            continue
        sessions.append(session)
    return sessions, SkipReport(
        total=total,
        parsed=len(sessions),
        skipped=total - len(sessions),
        reasons=dict(reasons),
    )


def build_corpus(directory: Path, depth: int = 1,
                 exclude: Sequence[str] = ()) -> tuple[FeatureSet, SkipReport]:
    """Load every .jsonl in `directory` and extract features over a shared vocabulary.

    `exclude` names feature columns to drop; it is forwarded verbatim to `extract`.
    """
    paths = sorted(Path(directory).glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no .jsonl files under {directory}")

    all_sessions: list[RawSession] = []
    report = SkipReport(total=0, parsed=0, skipped=0, reasons={})
    for path in paths:
        sessions, file_report = load_sessions(path)
        all_sessions.extend(sessions)
        report = report.merge(file_report)

    if not all_sessions:
        raise ValueError(f"parsed 0 sessions from {len(paths)} files under {directory}")
    return extract(all_sessions, depth=depth, exclude=exclude), report
