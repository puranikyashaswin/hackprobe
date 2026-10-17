import json

from common.jsonl import append, done_keys, pending


def test_resume_skips_written_rows(tmp_path):
    out = tmp_path / "out.jsonl"
    rows = [{"id": i} for i in range(5)]
    for row in list(pending(rows, str(out)))[:3]:
        append(str(out), {"id": row["id"], "score": row["id"] * 2})
    assert done_keys(str(out)) == {0, 1, 2}
    assert [r["id"] for r in pending(rows, str(out))] == [3, 4]


def test_torn_last_line_is_ignored(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({"id": 0}) + "\n" + '{"id": 1, "sco')
    assert done_keys(str(out)) == {0}
