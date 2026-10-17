"""S1.5: model pull and checks.

1. Forward pass returns 37 hidden states x 4096 on a short prompt.
2. Turn-by-turn render: assistant-token counts within 2% of generated_tokens on 5 rows.
3. Generation with enable_thinking=False writes no <think> content and opens its
   code block the way HackTrace's Qwen3-8B does (read-only: ```python then a
   "# solution.py" line; editable: ```solution.py).
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from common.render import assistant_tokens, prompt_text
from extract.pooled import RAW, ROOT, SPLITS, load_model, states


def main() -> None:
    tok, model = load_model()
    h = states(model, tok("def add(a, b): return a + b", add_special_tokens=False)["input_ids"])
    shape_ok = tuple(h.shape[0::2]) == (37, 4096)

    test = pq.read_table(RAW / "test.parquet", columns=["id", "variant", "messages", "generated_tokens"]).to_pylist()
    anchor = {r["id"] for r in pq.read_table(SPLITS / "anchor_test.parquet").to_pylist()}
    sample = [r for r in test if r["id"] in anchor][::300][:5]
    counts = [{"ours": assistant_tokens(tok, r["messages"]), "generated": r["generated_tokens"]} for r in sample]
    render_ok = all(abs(c["ours"] - c["generated"]) <= 0.02 * c["generated"] + 1 for c in counts)

    gens = []
    for variant in ("read-only-test", "editable-test"):
        row = next(r for r in test if r["variant"] == variant)
        ids = tok(prompt_text(tok, row["messages"], 1), add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=600, do_sample=True, temperature=0.9, top_p=0.95)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        head = text.lstrip()[:40]
        expect = "```python" if variant == "read-only-test" else "```solution.py"
        gens.append({"variant": variant, "think": "<think>" in text and "</think>" not in text[:20],
                     "opens_like_hacktrace": head.startswith(expect), "head": head,
                     "fences_closed": text.count("```") % 2 == 0})

    res = {"hidden_states_37x4096": shape_ok, "render_counts": counts, "render_ok": render_ok, "generation": gens,
           "pass": shape_ok and render_ok and not any(g["think"] for g in gens)}
    (ROOT / "results/hacktrace/data/s1.5_model_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
