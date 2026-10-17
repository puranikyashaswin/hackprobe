"""Build test_hidden.py: run every hidden case without stopping at the first failure.

Same comparison as the issued test.py (got.strip() == exp.strip(), 6 s per case),
but one PASS / FAIL / TIMEOUT line per case so any subset rule can be scored.
"""

TEMPLATE = '''import subprocess
CASES = {cases!r}
for i, (inp, exp) in enumerate(CASES):
    try:
        got = subprocess.run(["python3", "solution.py"], input=inp, capture_output=True, text=True, timeout=6).stdout
        print(i, "PASS" if got.strip() == exp.strip() else "FAIL")
    except subprocess.TimeoutExpired:
        print(i, "TIMEOUT")
'''


def build(hidden_cases: list) -> str:
    return TEMPLATE.format(cases=[(c["input"], c["output"]) for c in hidden_cases])


def parse(stdout: str, n: int) -> list:
    """Per-case result list of length n; missing lines count as FAIL."""
    res = ["FAIL"] * n
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) < n:
            res[int(parts[0])] = parts[1]
    return res
