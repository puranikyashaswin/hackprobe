"""S2.4 hand-check page: 60 hacks stratified by category x t*, blind.

For each row the page shows the turn where the shortcut first appears, token by
token with small index numbers. You type the index of the first token of any
comment, prose or code that refers to the visible tests or their values (any
wording), then press Reveal to see the locator's answer. The page counts hits
within +/- 3 tokens and can download your marks as JSON. Pass: >= 54 of 60.
"""

import html
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from common.render import render_turn

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"
REV = json.loads((ROOT / "env/revisions.json").read_text())["models"]["Qwen/Qwen3-8B"]
STRATA = [("read-only-test", "hardcode_output"), ("read-only-test", "special_casing"), ("editable-test", "test_tamper")]


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=REV)
    loc = [r for r in pq.read_table(ROOT / "results/hacktrace/labels/fsrt_test.parquet").to_pylist() if r["fsrt"] is not None]
    art = {r["id"]: r["artefact"] for r in pq.read_table(ROOT / "results/hacktrace/labels/dedup_test.parquet", columns=["id", "artefact"]).to_pylist()}
    rng = random.Random(20261016)
    picked = []
    for variant, cat in STRATA:
        for early in (True, False):
            pool = [r for r in loc if r["variant"] == variant and r["judge_category"] == cat
                    and (r["hack_turn"] == 1) == early and not art.get(r["id"], False)]
            rng.shuffle(pool)
            picked += pool[:10]
    rng.shuffle(picked)
    ids = {r["id"] for r in picked}
    msgs = {r["id"]: r["messages"] for r in pq.read_table(RAW / "trajectories/test.parquet", columns=["id", "messages"]).to_pylist() if r["id"] in ids}

    cards = []
    for n, r in enumerate(picked, 1):
        rr = render_turn(tok, msgs[r["id"]], r["hack_turn"])
        toks = []
        for i in range(rr.assistant_start, len(rr.ids)):
            a, b = rr.offsets[i]
            piece = html.escape(rr.text[a:b]).replace("\n", "↵<br>")
            toks.append(f'<span class="t" data-i="{i - rr.assistant_start}"><sub>{i - rr.assistant_start}</sub>{piece}</span>')
        cards.append(f'''<section class="card" data-fsrt="{r["fsrt"]}" data-id="{r["id"]}">
<h3>Row {n} of {len(picked)} <small>turn {r["hack_turn"]} · {html.escape(r["judge_category"])}</small></h3>
<div class="code">{"".join(toks)}</div>
<p>First token referring to the visible tests or their values: <input type="number" min="0" class="mark">
<button onclick="reveal(this)">Reveal</button> <span class="res"></span></p></section>''')

    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FSRT hand-check</title><style>
:root{{--bg:#fff;--fg:#1f1f1f;--muted:#666;--hl:#ffe08a;--ok:#1a7f37;--bad:#c62828;--card:#f6f6f4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#e8e8e8;--muted:#999;--hl:#7a5b00;--ok:#5cc777;--bad:#ff7b7b;--card:#1f1f1f}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}}
.card{{background:var(--card);border-radius:10px;padding:12px 16px;margin:16px 0}}
.code{{font:13px/1.9 ui-monospace,monospace;white-space:normal;word-break:break-word}}
.t sub{{color:var(--muted);font-size:9px;margin-right:1px}} .t.hl{{background:var(--hl)}}
small{{color:var(--muted)}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}} #tally{{position:sticky;top:0;background:var(--bg);padding:8px 0}}
</style></head><body>
<h1>FSRT hand-check (S2.4)</h1>
<p>For each row, type the index (small number) of the first token of any comment, prose or code that refers to the visible tests or their values, in any wording. Then press Reveal. Pass: at least 54 of 60 within ±3 tokens.</p>
<div id="tally">Checked 0 of {len(picked)} · within ±3: 0 · <button onclick="dl()">Download my marks</button></div>
{"".join(cards)}
<script>
let done=0,hit=0,marks={{}};
function reveal(btn){{const c=btn.closest('.card'),m=c.querySelector('.mark'),r=c.querySelector('.res');
if(m.value===''){{r.textContent='Enter your mark first';r.className='res bad';return}}
if(c.dataset.done)return;const f=+c.dataset.fsrt,v=+m.value,ok=Math.abs(f-v)<=3;
c.querySelectorAll('.t').forEach(t=>{{if(+t.dataset.i===f)t.classList.add('hl')}});
r.textContent=(ok?'Within ±3':'Off by '+(v-f))+' (locator: '+f+')';r.className='res '+(ok?'ok':'bad');
c.dataset.done=1;done++;hit+=ok;marks[c.dataset.id]={{mark:v,locator:f,ok}};
document.getElementById('tally').firstChild.textContent='Checked '+done+' of {len(picked)} · within ±3: '+hit+' · ';}}
function dl(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(marks,null,1)],{{type:'application/json'}}));a.download='fsrt_handcheck.json';a.click()}}
</script></body></html>'''
    out = ROOT / "results/hacktrace/labels/fsrt_handcheck.html"
    out.write_text(page)
    print(f"{len(picked)} rows -> {out}")


if __name__ == "__main__":
    main()
