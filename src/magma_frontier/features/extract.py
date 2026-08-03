"""Structural feature extraction. This module is the text boundary.

Nothing textual crosses into `FeatureSet.X`. Tool identifiers are used only to build
counts over a taxonomy vocabulary; argument values, message content, tool-call ids,
absolute timestamps, token counts and cost never appear.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from magma_frontier.features.taxonomy import taxonomy_path
from magma_frontier.schema import RawSession

_SHAPE_TOKENS = ("str", "int", "float", "bool", "list", "dict", "null")

# Tool ids that survive taxonomy generalization become feature-column names. A
# malformed id (models sometimes emit prose or control characters where a tool name
# belongs) would otherwise put raw text into feature_names and act as a one-hot
# tenant indicator. Anything not matching this shape collapses to a single token.
# `re.match` with a trailing `$` accepts one trailing newline, which would let a
# truncated model emission through with its newline intact. `fullmatch` does not.
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
_UNSAFE_TOKEN = "malformed"


def _safe(token: str) -> str:
    return token if _SAFE_TOKEN.fullmatch(token) else _UNSAFE_TOKEN


@dataclass(frozen=True, slots=True)
class FeatureSet:
    X: np.ndarray
    feature_names: tuple[str, ...]
    tenant_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    outcomes: tuple[bool | None, ...]
    ngram_vocabulary: tuple[str, ...]


def _scalar_features(session: RawSession) -> dict[str, float]:
    spans = session.spans
    n = len(spans)
    if n == 0:
        raise ValueError(f"{session.session_id}: session has no spans")
    statuses = [s.status for s in spans]
    arities = [s.arg_arity for s in spans]

    max_run, current_run = 1, 1
    for prev, cur in zip(spans, spans[1:]):
        current_run = current_run + 1 if cur.tool_id == prev.tool_id else 1
        max_run = max(max_run, current_run)

    distinct = len({s.tool_id for s in spans})
    shape_counts = Counter(
        token
        for s in spans
        for token in s.arg_type_shape.split(",")
        if token in _SHAPE_TOKENS
    )

    features = {
        "n_steps": float(n),
        "n_distinct_tools": float(distinct),
        "tool_reuse_ratio": float(n - distinct) / float(n),
        "max_repeat_run": float(max_run),
        "error_rate": float(statuses.count("error")) / float(n),
        "orphan_rate": float(statuses.count("missing")) / float(n),
        "error_then_retry": float(
            sum(
                1
                for a, b in zip(spans, spans[1:])
                if a.status == "error" and b.tool_id == a.tool_id
            )
        ),
        "mean_arity": float(np.mean(arities)),
        "max_arity": float(max(arities)),
        "zero_arity_ratio": float(arities.count(0)) / float(n),
        "duration_s": float(session.duration_s or 0),
        # 1.0 indicates the source carried both timestamps; 0.0 means duration is unknown.
        # Without this, a session with missing timestamps is indistinguishable from a
        # genuinely instantaneous one, since both store duration_s = 0.
        "has_duration": 0.0 if session.duration_s is None else 1.0,
        # Floor the denominator at one second, not one minute: flooring at a minute
        # collapses every sub-minute session to n_steps and destroys the rate signal.
        "steps_per_minute": 60.0 * float(n) / max(1.0, float(session.duration_s or 0)),
    }
    for token in _SHAPE_TOKENS:
        features[f"shape_{token}_ratio"] = float(shape_counts[token]) / float(n)
    return features


def _ngram_counts(session: RawSession, depth: int) -> Counter[str]:
    path = [_safe(taxonomy_path(s.tool_id, depth)) for s in session.spans]
    counts: Counter[str] = Counter()
    total = len(path)
    for token in path:
        counts[f"uni:{token}"] += 1
    for a, b in zip(path, path[1:]):
        counts[f"bi:{a}>{b}"] += 1
    for a, b, c in zip(path, path[1:], path[2:]):
        counts[f"tri:{a}>{b}>{c}"] += 1
    # Normalize to rates so session length does not dominate the n-gram block.
    return Counter({k: v / total for k, v in counts.items()})


def extract(sessions: Sequence[RawSession], depth: int = 1,
            exclude: Sequence[str] = (),
            vocabulary: Sequence[str] | None = None) -> FeatureSet:
    """Build a numeric FeatureSet from sessions.

    `depth` selects taxonomy generality. `exclude` drops named feature columns.
    `vocabulary` forces an exact n-gram column set: pass the vocabulary derived from a
    training fold so held-out rows cannot introduce columns the model never trained on.
    Unseen columns are zero-filled; columns present in the data but absent from the
    supplied vocabulary are discarded.
    """
    if not sessions:
        raise ValueError("extract() needs at least one session")

    scalar_rows = [_scalar_features(s) for s in sessions]
    ngram_rows = [_ngram_counts(s, depth) for s in sessions]

    scalar_names = tuple(sorted(scalar_rows[0]))
    if vocabulary is None:
        ngram_names = tuple(sorted({name for row in ngram_rows for name in row}))
    else:
        ngram_names = tuple(vocabulary)
    derived_vocabulary = ngram_names

    feature_names = scalar_names + ngram_names
    if exclude:
        dropped = set(exclude)
        scalar_names = tuple(n for n in scalar_names if n not in dropped)
        ngram_names = tuple(n for n in ngram_names if n not in dropped)
        feature_names = scalar_names + ngram_names

    X = np.zeros((len(sessions), len(feature_names)), dtype=np.float64)
    for i, (scalars, ngrams) in enumerate(zip(scalar_rows, ngram_rows)):
        for j, name in enumerate(scalar_names):
            X[i, j] = scalars[name]
        offset = len(scalar_names)
        for j, name in enumerate(ngram_names):
            X[i, offset + j] = ngrams.get(name, 0.0)

    return FeatureSet(
        X=X,
        feature_names=feature_names,
        tenant_ids=tuple(s.tenant_id for s in sessions),
        task_ids=tuple(s.task_id for s in sessions),
        session_ids=tuple(s.session_id for s in sessions),
        outcomes=tuple(s.outcome for s in sessions),
        ngram_vocabulary=derived_vocabulary,
    )
