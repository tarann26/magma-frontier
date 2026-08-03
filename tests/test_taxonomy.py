import pytest

from magma_frontier.features.taxonomy import category_of, namespace_of, taxonomy_path


def test_namespace_is_prefix_before_first_hyphen():
    assert namespace_of("filesystem-read_file") == "filesystem"
    assert namespace_of("rail_12306-get-current-date") == "rail_12306"
    assert namespace_of("google-cloud-bigquery_run_query") == "google"


def test_namespace_without_hyphen_is_whole_name():
    assert namespace_of("claim_done") == "claim_done"


def test_category_groups_namespaces():
    assert category_of("filesystem-read_file") == "storage"
    assert category_of("terminal-run_command") == "exec"
    assert category_of("local-python-execute") == "exec"
    assert category_of("github-get_file_contents") == "devtools"
    assert category_of("google_sheet-append") == "productivity"
    assert category_of("yahoo-finance-get_stock_price_by_date") == "web"


def test_unknown_namespace_falls_back_to_other():
    assert category_of("wizardry-cast_spell") == "other"


def test_taxonomy_path_depths():
    assert taxonomy_path("github-get_file_contents", 0) == "devtools"
    assert taxonomy_path("github-get_file_contents", 1) == "github"
    assert taxonomy_path("github-get_file_contents", 2) == "github-get_file_contents"


def test_taxonomy_path_rejects_bad_depth():
    with pytest.raises(ValueError, match="depth"):
        taxonomy_path("github-get_file_contents", 3)
