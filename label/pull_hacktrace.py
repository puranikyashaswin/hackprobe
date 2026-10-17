"""S2.1: download HackTrace at the pinned revision and check it against MANIFEST.json.

Checks size, sha256 and row count for every file, and the ACT positive count
for the two trajectory files. Reads only parquet metadata and the ACT column.
"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "JH012/HackTrace-Data"
REV = json.loads((ROOT / "env/revisions.json").read_text())["datasets"][REPO]
RAW = ROOT / "data/raw/hacktrace"
OUT = ROOT / "results/hacktrace/data/manifest_check.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    get = lambda name: Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REV, local_dir=RAW))
    manifest = json.loads(get("MANIFEST.json").read_text())
    report, ok = {"repo": REPO, "revision": REV, "files": {}}, True
    for name, want in manifest.items():
        path = get(name)
        got = {"bytes": path.stat().st_size, "sha256": sha256(path), "rows": pq.ParquetFile(path).metadata.num_rows}
        if "act_positives" in want:
            acts = pq.read_table(path, columns=["ACT"]).column("ACT").to_pylist()
            got["act_positives"] = sum(1 for v in acts if v)
        match = all(got[k] == want[k] for k in want)
        ok &= match
        report["files"][name] = {"want": want, "got": got, "match": match}
        print(f"{name}: {'OK' if match else 'MISMATCH'} rows={got['rows']}")
    report["pass"] = ok
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if ok else "FAIL", "->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
