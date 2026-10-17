"""Fast generation with vLLM for the exploratory second-model runs, in three steps so vLLM runs in its own environment.

  prep    (main env)  render prompts with the model's chat template (thinking off) -> prompts_<tag>.jsonl
  run     (vLLM env)  standalone: tensor-parallel sampling, n samples per prompt, fixed seed -> raw_<tag>.jsonl
  post    (main env)  convert to the gen_<tag>.jsonl format the existing labelers read

Sampling matches the Hugging Face path: temperature 0.7, top-p 0.95. Activations for probes are
always extracted afterwards by teacher forcing in the main environment, so the generator does not
enter any probe or monitor.

  env/gen/bin/python -m env.vgen prep hacktrace --model M --rev R --split test --problems 0 --tag T
  env/gen/bin/python -m env.vgen prep impossible --model M --rev R --split oneoff --prompt licensed --tag T
  <vllm-env>/bin/python env/vgen.py run --tag T --model M --rev R --tp 4 --n 16 --max-tokens 1536
  env/gen/bin/python -m env.vgen post hacktrace|impossible --tag T
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/vgen"


def cmd_prep(a):
    from transformers import AutoTokenizer
    from env.hacktrace_gen import render
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.kind == "hacktrace":
        import random
        from env.hacktrace_gen import SEED, prompts
        items = prompts(a.split)
        if a.problems:
            items = random.Random(SEED).sample(items, min(a.problems, len(items)))
        for (tier, name), m in sorted(items):
            rows.append({"base": f"{tier}|{name}", "tier": tier, "problem_name": name, "text": render(tok, m)})
    else:
        from env.impossible import PROMPTS, SYSTEM, problems
        for r in problems(a.split):
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPTS[a.prompt].format(**r)}]
            rows.append({"base": f"{r['task_id']}|{a.tag}", "task_id": r["task_id"], "split": a.split, "prompt_kind": a.tag, "text": render(tok, msgs)})
    with open(OUT / f"prompts_{a.tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "prompts ->", OUT / f"prompts_{a.tag}.jsonl")


def cmd_run(a):
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(OUT / f"prompts_{a.tag}.jsonl")]
    llm = LLM(model=a.model, revision=a.rev, tensor_parallel_size=a.tp, dtype="bfloat16", seed=20261016,
              max_model_len=a.max_model_len, gpu_memory_utilization=0.90, enable_prefix_caching=True)
    sp = SamplingParams(n=a.n, temperature=0.7, top_p=0.95, max_tokens=a.max_tokens, seed=20261016)
    tok = llm.get_tokenizer()
    fits = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) + a.max_tokens <= a.max_model_len for r in rows]
    skipped = [r["base"] for r, f in zip(rows, fits) if not f]
    rows = [r for r, f in zip(rows, fits) if f]
    (OUT / f"skipped_{a.tag}.json").write_text(json.dumps(skipped) + "\n")
    print(f"{len(skipped)} prompts longer than the context skipped (listed in skipped_{a.tag}.json)", flush=True)
    outs = llm.generate([r["text"] for r in rows], sp)
    with open(OUT / f"raw_{a.tag}.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            for s, c in enumerate(o.outputs):
                f.write(json.dumps({**{k: v for k, v in r.items() if k != "text"}, "sample": s, "completion": c.text,
                                    "n_new_tokens": len(c.token_ids), "finish": c.finish_reason}) + "\n")
    print(sum(len(o.outputs) for o in outs), "completions ->", OUT / f"raw_{a.tag}.jsonl")


def cmd_cont(a):
    """Resume every completion that hit the length limit: prompt + partial text, one continuation each."""
    from vllm import LLM, SamplingParams
    prompts = {r["base"]: r["text"] for r in (json.loads(l) for l in open(OUT / f"prompts_{a.tag}.jsonl"))}
    raw = [json.loads(l) for l in open(OUT / f"raw_{a.tag}.jsonl")]
    todo = [i for i, r in enumerate(raw) if r["finish"] == "length"]
    print(f"{len(todo)} of {len(raw)} completions to continue", flush=True)
    if todo:
        llm = LLM(model=a.model, revision=a.rev, tensor_parallel_size=a.tp, dtype="bfloat16", seed=20261017,
                  max_model_len=a.max_model_len, gpu_memory_utilization=0.90, enable_prefix_caching=True)
        sp = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=a.max_tokens, seed=20261017)
        outs = llm.generate([prompts[raw[i]["base"]] + raw[i]["completion"] for i in todo], sp)
        for i, o in zip(todo, outs):
            c = o.outputs[0]
            raw[i]["completion"] += c.text
            raw[i]["n_new_tokens"] += len(c.token_ids)
            raw[i]["finish"] = c.finish_reason
            raw[i]["continued"] = True
    (OUT / f"raw_{a.tag}.jsonl").rename(OUT / f"raw_{a.tag}.before_continue.jsonl")
    with open(OUT / f"raw_{a.tag}.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    print("still truncated:", sum(r["finish"] == "length" for r in raw), "->", OUT / f"raw_{a.tag}.jsonl")


def cmd_post(a):
    raw = [json.loads(l) for l in open(OUT / f"raw_{a.tag}.jsonl")]
    if a.kind == "hacktrace":
        from env.hacktrace_gen import OUT as HOUT, solution_code
        HOUT.mkdir(parents=True, exist_ok=True)
        path = HOUT / f"gen_{a.tag}.jsonl"
        with open(path, "w") as f:
            for r in raw:
                f.write(json.dumps({"key": f"{r['base']}|{r['sample']}", "tier": r["tier"], "problem_name": r["problem_name"], "sample": r["sample"],
                                    "completion": r["completion"], "code": solution_code(r["completion"]), "finish": r["finish"]}) + "\n")
    else:
        from env.impossible import OUT as IOUT, extract_code
        IOUT.mkdir(parents=True, exist_ok=True)
        path = IOUT / f"gen_{a.tag}.jsonl"
        with open(path, "w") as f:
            for r in raw:
                f.write(json.dumps({"key": f"{r['base']}|{r['sample']}", "task_id": r["task_id"], "prompt_kind": r["prompt_kind"], "split": r["split"],
                                    "sample": r["sample"], "completion": r["completion"], "code": extract_code(r["completion"]),
                                    "truncated": r["finish"] == "length", "n_new_tokens": r["n_new_tokens"]}) + "\n")
    print(len(raw), "->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "cont", "post"])
    ap.add_argument("kind", nargs="?", choices=["hacktrace", "impossible"])
    ap.add_argument("--model")
    ap.add_argument("--rev")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prompt", default="licensed")
    ap.add_argument("--problems", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "cont": cmd_cont, "post": cmd_post}[a.cmd](a)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
