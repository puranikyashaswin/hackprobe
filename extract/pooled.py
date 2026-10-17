"""S3.1: pooled activations of Qwen3-8B on the anchor rows.

For every row, the final assistant turn is rendered turn by turn (common/render.py)
and run through the model once. Two pooled readouts per layer (37 hidden states):
  meancode: mean over fenced-code tokens of the final turn (HackTrace's site)
  meangen:  mean over all assistant tokens of the final turn
Plus `lastprompt`: the last token of the turn-1 prompt, once per problem group
(identical for every row of a problem), for S3.5 only.

Written in 256-row shards with an index appended only after a shard is on disk,
so a restart redoes at most one shard. Train rows exclude set C (asserted).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.jsonl import append, done_keys
from common.render import prompt_text, render_turn, n_turns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace/trajectories"
SPLITS = ROOT / "results/hacktrace/splits"
MODEL = "Qwen/Qwen3-8B"
REV = json.loads((ROOT / "env/revisions.json").read_text())["models"][MODEL]
SHARD = 256


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa").eval()
    return tok, model


@torch.no_grad()
def states(model, ids):
    # the base model returns the same hidden states without computing vocabulary logits (saves GPU memory)
    out = model.model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
    return torch.stack(out.hidden_states)[:, 0]  # [37, L, 4096]


@torch.no_grad()
def pooled_means(model, ids, masks):
    """Mean of each of the 37 hidden states over each boolean mask (cuda tensors), pooled as the
    forward pass reaches each layer, so long rows never hold all layers at once.
    Gives the same numbers as states(model, ids)[:, m].float().mean(1) (checked by check_pooled_means)."""
    out = [[None] * 37 for _ in masks]

    def put(l, h):
        for j, m in enumerate(masks):
            out[j][l] = h[0, m].float().mean(0)
    hooks = [model.model.embed_tokens.register_forward_hook(lambda mod, i, o: put(0, o))]
    for l, block in enumerate(model.model.layers[:-1]):  # outputs of layers 0..34 = hidden_states[1..35]
        hooks.append(block.register_forward_hook(lambda mod, i, o, l=l: put(l + 1, o[0] if isinstance(o, tuple) else o)))
    try:
        last = model.model(torch.tensor([ids], device="cuda")).last_hidden_state  # hidden_states[36] (after the final norm)
    finally:
        for h in hooks:
            h.remove()
    put(36, last)
    return [torch.stack(o) for o in out]  # each [37, 4096] float32


def check_pooled_means(model, ids, masks):
    ref = states(model, ids)
    for m, got in zip(masks, pooled_means(model, ids, masks)):
        assert torch.equal(ref[:, m].float().mean(1), got), "pooled_means differs from states()"
    del ref
    torch.cuda.empty_cache()


def c_names():
    return set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))


def messages_for(ids_wanted, split):
    t = pq.read_table(RAW / f"{split}.parquet", columns=["id", "problem_name", "messages"])
    want = set(ids_wanted)
    return {r["id"]: r for r in t.to_pylist() if r["id"] in want}


def pooled(args):
    tok, model = load_model()
    out = ROOT / "acts/qwen3-8b/hacktrace/pooled"
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    rows = [("test", r["id"]) for r in pq.read_table(SPLITS / "anchor_test.parquet").to_pylist()] + \
           [("train", r["id"]) for r in pq.read_table(SPLITS / "anchor_train.parquet").to_pylist()]
    if args.limit:
        rows = rows[: args.limit]
    msgs = {**messages_for([i for s, i in rows if s == "test"], "test"),
            **messages_for([i for s, i in rows if s == "train"], "train")}
    banned = c_names()
    assert not any(msgs[i]["problem_name"] in banned for _, i in rows), "row from set C"

    done = done_keys(str(index))
    todo = [i for _, i in rows if i not in done]
    shard_no = len({json.loads(l)["shard"] for l in open(index)}) if index.exists() else 0
    print(f"{len(rows)} rows, {len(todo)} to do, starting at shard {shard_no}", flush=True)
    for start in range(0, len(todo), SHARD):
        chunk = todo[start: start + SHARD]
        code = np.zeros((len(chunk), 37, 4096), np.float16)
        gen = np.zeros_like(code)
        for k, rid in enumerate(chunk):
            m = msgs[rid]["messages"]
            r = render_turn(tok, m, n_turns(m))
            h = states(model, r.ids)
            cm = torch.tensor(r.code_mask, device="cuda")
            gm = torch.zeros_like(cm)
            gm[r.assistant_start:] = True
            if not cm.any():
                cm = gm
            code[k] = h[:, cm].float().mean(1).half().cpu().numpy()
            gen[k] = h[:, gm].float().mean(1).half().cpu().numpy()
        d = out / f"shard_{shard_no:04d}"
        d.mkdir(exist_ok=True)
        for layer in range(37):
            np.save(d / f"L{layer:02d}_meancode.npy", code[:, layer])
            np.save(d / f"L{layer:02d}_meangen.npy", gen[:, layer])
        for off, rid in enumerate(chunk):
            append(str(index), {"id": rid, "shard": shard_no, "offset": off})
        print(f"shard {shard_no}: {start + len(chunk)}/{len(todo)}", flush=True)
        shard_no += 1


def lastprompt(args):
    """One turn-1 prompt per (variant, tier, problem): test (all 315) and train minus C, non-conf."""
    tok, model = load_model()
    out = ROOT / "acts/qwen3-8b/hacktrace/lastprompt"
    out.mkdir(parents=True, exist_ok=True)
    banned = c_names()
    groups = {}
    for split in ("test", "train"):
        t = pq.read_table(RAW / f"{split}.parquet", columns=["variant", "tier", "problem_name", "messages"],
                          filters=[("problem_name", "not in", sorted(banned))] if split == "train" else None)
        for r in t.to_pylist():
            if "conf" in r["tier"]:
                continue
            key = (split, r["variant"], r["tier"], r["problem_name"])
            head = (r["messages"][0]["content"], r["messages"][1]["content"])
            assert groups.setdefault(key, head) == head, f"turn-1 prompt differs within {key}"
    keys = sorted(groups)
    vec = np.zeros((len(keys), 37, 4096), np.float16)
    for k, key in enumerate(keys):
        sys_msg, user = groups[key]
        ids = tok(prompt_text(tok, [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}], 1),
                  add_special_tokens=False)["input_ids"]
        vec[k] = states(model, ids)[:, -1].half().cpu().numpy()
    np.save(out / "lastprompt.npy", vec)
    with open(out / "groups.jsonl", "w") as f:
        for key in keys:
            f.write(json.dumps(dict(zip(["split", "variant", "tier", "problem_name"], key))) + "\n")
    print(f"{len(keys)} prompt groups", flush=True)


def check(args):
    """Repeatability: re-run 20 rows and compare with the stored shard."""
    tok, model = load_model()
    out = ROOT / "acts/qwen3-8b/hacktrace/pooled"
    idx = [json.loads(l) for l in open(out / "index.jsonl")][:20]
    msgs = messages_for([r["id"] for r in idx], "test")
    worst_diff, worst_cos = 0.0, 1.0
    for r in idx:
        m = msgs[r["id"]]["messages"]
        rr = render_turn(tok, m, n_turns(m))
        h = states(model, rr.ids)
        cm = torch.tensor(rr.code_mask, device="cuda")
        if not cm.any():
            cm[rr.assistant_start:] = True
        new = h[:, cm].float().mean(1).half().cpu().numpy().astype(np.float32)  # same fp16 rounding as stored
        old = np.stack([np.load(out / f"shard_{r['shard']:04d}/L{l:02d}_meancode.npy")[r["offset"]] for l in range(37)]).astype(np.float32)
        worst_diff = max(worst_diff, float(np.abs(new - old).max()))
        cos = (new * old).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(old, axis=1) + 1e-9)
        worst_cos = min(worst_cos, float(cos.min()))
    res = {"rows": len(idx), "max_abs_diff": worst_diff, "min_cosine": worst_cos,
           "pass": worst_diff < 1e-2 and worst_cos > 0.9999}
    (ROOT / "results/hacktrace/data/s3.1_repeat_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pooled", "lastprompt", "check"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"pooled": pooled, "lastprompt": lastprompt, "check": check}[a.what](a)
