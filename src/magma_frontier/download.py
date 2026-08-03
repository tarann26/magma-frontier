"""Fetch the gated Toolathlon corpus. Requires HF login and accepted terms."""

from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "hkust-nlp/Toolathlon-Trajectories"


def download_corpus(dest: Path, files: Sequence[str] | None = None) -> list[Path]:
    """Download trajectory files into `dest`. Pass `files` to fetch a subset."""
    dest.mkdir(parents=True, exist_ok=True)
    available = [f for f in list_repo_files(REPO_ID, repo_type="dataset")
                 if f.endswith(".jsonl")]
    wanted = list(files) if files is not None else available
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"not in repo: {missing}")

    paths = []
    for name in wanted:
        local = hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=dest)
        paths.append(Path(local))
    return paths
