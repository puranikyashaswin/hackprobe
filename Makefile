PY ?= env/cpu/bin/python

.PHONY: env-cpu smoke smoke-resume s2.1 s2.0 s2.6 s2.3a s2.2 s2.3

env-cpu:            ## CPU-only environment for the analyses, figures and tests
	uv venv -q env/cpu --python 3.12
	uv pip install -q --python env/cpu/bin/python -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

smoke:              ## fast checks, CPU only, < 5 min
	$(PY) -m pytest -q

smoke-resume:       ## resume-logic test for the per-row JSONL writer
	$(PY) -m pytest -q tests/test_jsonl.py

s2.1:               ## download HackTrace at the pinned revision and verify it against MANIFEST.json
	$(PY) label/pull_hacktrace.py

s2.0:               ## reserve the confirmation set C from unopened train problems (label columns only)
	$(PY) label/reserve_c.py

s2.6:               ## within-problem index: (variant, problem) groups with both ACT classes
	$(PY) label/within_problem.py

s2.3a:              ## AST dedup keys and the editable artefact flag
	$(PY) label/dedup.py

s2.2:               ## reproduce HackTrace visible/hidden labels in the sandbox
	$(PY) -m label.grade

s2.3:               ## anchor train/test splits (train minus C)
	$(PY) -m label.splits
