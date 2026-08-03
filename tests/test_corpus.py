from pathlib import Path

import pytest

from magma_frontier.corpus import SkipReport, build_corpus, load_sessions

FIXTURE = Path(__file__).parent / "fixtures" / "toolathlon_sample.jsonl"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="fixture holds gated corpus records; rebuild with scripts/build_fixture.py",
)


def test_load_sessions_parses_fixture(tmp_path):
    target = tmp_path / "gpt-5-mini_1.jsonl"
    target.write_text(FIXTURE.read_text())
    sessions, report = load_sessions(target)
    assert len(sessions) == 2
    assert report.parsed == 2
    assert report.skipped == 0


def test_malformed_lines_are_counted_not_dropped(tmp_path):
    target = tmp_path / "gpt-5-mini_1.jsonl"
    target.write_text(FIXTURE.read_text() + "{not json\n")
    sessions, report = load_sessions(target)
    assert len(sessions) == 2
    assert report.total == 3
    assert report.skipped == 1
    assert sum(report.reasons.values()) == 1


def test_blank_lines_are_not_counted_as_records(tmp_path):
    target = tmp_path / "gpt-5-mini_1.jsonl"
    target.write_text(FIXTURE.read_text() + "\n\n")
    _, report = load_sessions(target)
    assert report.total == 2


def test_build_corpus_merges_files_and_reports(tmp_path):
    (tmp_path / "gpt-5-mini_1.jsonl").write_text(FIXTURE.read_text())
    (tmp_path / "gpt-5-mini_2.jsonl").write_text(FIXTURE.read_text())
    fs, report = build_corpus(tmp_path)
    assert fs.X.shape[0] == 4
    assert report.parsed == 4
    assert set(fs.tenant_ids) == {"gpt-5-mini"}


def test_build_corpus_is_deterministic(tmp_path):
    (tmp_path / "gpt-5-mini_1.jsonl").write_text(FIXTURE.read_text())
    (tmp_path / "gpt-5-mini_2.jsonl").write_text(FIXTURE.read_text())
    first, _ = build_corpus(tmp_path)
    second, _ = build_corpus(tmp_path)
    assert first.feature_names == second.feature_names
    assert first.session_ids == second.session_ids
    assert (first.X == second.X).all()


def test_build_corpus_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no .jsonl files"):
        build_corpus(tmp_path)
