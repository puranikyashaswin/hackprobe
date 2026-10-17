import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_revisions_exist():
    rev = json.loads((ROOT / "env" / "revisions.json").read_text())
    assert len(rev["datasets"]["JH012/HackTrace-Data"]) == 40


def test_private_folder_is_ignored():
    assert "private/*" in (ROOT / ".gitignore").read_text()
