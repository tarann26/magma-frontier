"""Rebuild the adapter test fixture from the upstream corpus.

    uv run python scripts/build_fixture.py

The fixture is two real records from hkust-nlp/Toolathlon-Trajectories. That dataset is
gated, so the records are not redistributed with this repository — you need to accept its
terms on HuggingFace and be logged in. Tests that depend on the fixture skip without it.
"""

import json
import os
import urllib.request
from pathlib import Path

SOURCE = ("https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories"
          "/resolve/main/gpt-5-mini_1.jsonl")
TARGET = Path("tests/fixtures/toolathlon_sample.jsonl")
RECORDS = 2


def main() -> None:
    token_path = Path(os.path.expanduser("~/.cache/huggingface/token"))
    if not token_path.exists():
        raise SystemExit(
            "no HuggingFace token found. Run `hf auth login`, and accept the dataset "
            "terms at https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories"
        )
    token = token_path.read_text().strip()

    request = urllib.request.Request(
        SOURCE, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-4000000"}
    )
    body = urllib.request.urlopen(request).read().decode("utf-8", "replace")

    lines = []
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            break  # truncated tail of the ranged request
        lines.append(line)
        if len(lines) == RECORDS:
            break

    if len(lines) != RECORDS:
        raise SystemExit(f"expected {RECORDS} complete records, got {len(lines)}")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(lines) + "\n")
    print(f"wrote {RECORDS} records to {TARGET}")


if __name__ == "__main__":
    main()
