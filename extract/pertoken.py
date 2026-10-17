"""S3.2: per-token activations at 6 layers over the stored spans of turn t*.

Layers: 4 evenly spaced (8, 16, 24, 32) + the S3.3 inner-fold winners (19
read-only, 4 editable). Positions: the spans listed by S2.3 (per-token part),
plus the last prompt token of turn t* (site a2). For every positive the token
ids fed to the model must hold fsrt_prev_tok at p*-1 and fsrt_tok at p*
(asserted; a mismatch is a template or offset bug and stops S3.6).
Shards of 256 rows: one array per layer (rows concatenated) + an index.
Usage: python -m extract.pertoken {test|train} [--part k --nparts n]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from common.jsonl import append, done_keys
from common.render import render_turn
from extract.pooled import RAW, ROOT, load_model

LAYERS = [4, 8, 16, 19, 24, 32]
SPL = ROOT / "results/hacktrace/splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["test", "train", "C"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    rows = pq.read_table(SPL / {"test": "anchor_pertoken.parquet", "train": "anchor_pertoken_train.parquet", "C": "anchor_pertoken_C.parquet"}[a.split]).to_pylist()
    rows = [r for k, r in enumerate(rows) if k % a.nparts == a.part]
    c_names = set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
    if a.split == "train":
        assert not any(r["problem_name"] in c_names for r in rows), "row from set C"
    if a.split == "C":
        assert all(r["problem_name"] in c_names for r in rows), "row outside set C"
    want = {r["id"] for r in rows}
    msgs = {r["id"]: r["messages"] for r in pq.read_table(RAW / f"{'train' if a.split == 'C' else a.split}.parquet", columns=["id", "messages"]).to_pylist()
            if r["id"] in want}
    tok, model = load_model()
    out = ROOT / f"acts/qwen3-8b/hacktrace/pertoken_{a.split}" / (f"part_{a.part:02d}" if a.nparts > 1 else "")
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    done = done_keys(str(index), "key")
    todo = [r for r in rows if r["key"] not in done]
    shard = len({json.loads(l)["shard"] for l in open(index)}) if index.exists() else 0
    print(f"{a.split} part {a.part}: {len(todo)} rows to do", flush=True)
    for start in range(0, len(todo), 256):
        chunk = todo[start: start + 256]
        bufs, a2, ids_buf, meta, cursor = [[] for _ in LAYERS], [], [], [], 0
        for r in chunk:
            rr = render_turn(tok, msgs[r["id"]], r["t_star"])
            s0 = rr.assistant_start
            if r["label"] == 1:
                assert rr.ids[s0 + r["p_star"]] == r["fsrt_tok"] and rr.ids[s0 + r["p_star"] - 1] == r["fsrt_prev_tok"], \
                    f"FSRT token mismatch in {r['id']}"
            lo, hi = s0 + r["span_start"], min(s0 + r["span_end"], len(rr.ids))
            with torch.no_grad():
                hs = model.model(torch.tensor([rr.ids[:hi]], device="cuda"), output_hidden_states=True).hidden_states
            for j, layer in enumerate(LAYERS):
                bufs[j].append(hs[layer][0, lo:hi].half().cpu().numpy())
            a2.append(torch.stack([hs[l][0, s0 - 1] for l in LAYERS]).half().cpu().numpy())
            ids_buf.append(np.array(rr.ids[lo:hi], np.int32))
            meta.append({"key": r["key"], "id": r["id"], "start": cursor, "length": hi - lo, "span_start": r["span_start"]})
            cursor += hi - lo
            del hs
        d = out / f"shard_{shard:04d}"
        d.mkdir(exist_ok=True)
        for j, layer in enumerate(LAYERS):
            np.save(d / f"L{layer:02d}.npy", np.concatenate(bufs[j]))
        np.save(d / "a2.npy", np.stack(a2))
        np.save(d / "token_ids.npy", np.concatenate(ids_buf))
        with open(d / "rows.jsonl", "w") as f:
            for m in meta:
                f.write(json.dumps(m) + "\n")
        for k, m in enumerate(meta):
            append(str(index), {"key": m["key"], "shard": shard, "row": k})
        print(f"  shard {shard}: {start + len(chunk)}/{len(todo)}", flush=True)
        shard += 1


if __name__ == "__main__":
    main()
