"""S1.3b: run model-written code in a locked-down container.

Only solution.py and test.py go in. No network, read-only root, a small tmpfs to
work in, a process limit, a memory and CPU cap, an unprivileged user, and no
host home, credentials or environment variables. Model-written code never runs
on the host.
"""

import os
import subprocess
import tempfile
from pathlib import Path

IMAGE = "python:3.12-slim"
# Colima shares only $HOME with its VM, so row folders live under it.
SCRATCH = Path.home() / ".cache" / "hackprobe-sandbox"


def run(solution: str, test: str, timeout_s: float = 30.0, image: str = IMAGE) -> dict:
    """Run `python test.py` with solution.py beside it. Returns stdout, exit code, timeout flag."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        Path(d, "solution.py").write_text(solution)
        Path(d, "test.py").write_text(test)
        os.chmod(d, 0o755)  # the unprivileged container user must be able to read the two files
        for f in ("solution.py", "test.py"):
            os.chmod(Path(d, f), 0o644)
        cmd = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--tmpfs", "/work:size=64m,exec,mode=1777", "--pids-limit", "64",
            "--memory", "1g", "--cpus", "1", "--user", "65534", "--env-file", "/dev/null",
            "-v", f"{d}:/src:ro", "-w", "/work", image,
            "sh", "-c", f"cp /src/solution.py /src/test.py /work/ && timeout {int(timeout_s)} python test.py",
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 15)
            return {"stdout": p.stdout, "stderr": p.stderr[-2000:], "code": p.returncode, "timed_out": p.returncode == 124}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "host timeout", "code": -1, "timed_out": True}
