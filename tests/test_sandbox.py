"""S1.3b pass rule: three attacks fail, a normal HackTrace-style row prints ALL PASS."""

import shutil
import subprocess

import pytest

from label.sandbox import run

docker_ok = shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not docker_ok, reason="docker not running")

TEST_PY = """
import subprocess
CASES = [("2 3\\n", "5"), ("10 -4\\n", "6")]
ok = True
for inp, want in CASES:
    out = subprocess.run(["python3", "solution.py"], input=inp, capture_output=True, text=True, timeout=6).stdout.strip()
    ok &= out == want
print("ALL PASS" if ok else "FAIL")
"""


def test_normal_row_passes():
    sol = "a, b = map(int, input().split())\nprint(a + b)\n"
    assert "ALL PASS" in run(sol, TEST_PY)["stdout"]


def test_metadata_endpoint_unreachable():
    attack = "import urllib.request\nurllib.request.urlopen('http://169.254.169.254/latest/meta-data/', timeout=3)\n"
    test = "import subprocess\nr = subprocess.run(['python3','solution.py'])\nprint('REACHED' if r.returncode == 0 else 'BLOCKED')\n"
    assert "BLOCKED" in run(attack, test)["stdout"]


def test_no_aws_credentials():
    attack = "import os\nprint(open(os.path.expanduser('~/.aws/credentials')).read())\n"
    test = "import subprocess\nr = subprocess.run(['python3','solution.py'], capture_output=True)\nprint('REACHED' if r.returncode == 0 else 'BLOCKED')\n"
    assert "BLOCKED" in run(attack, test)["stdout"]


def test_fork_limit():
    attack = (
        "import os, time\nn = 0\n"
        "for _ in range(1000):\n"
        "    try:\n        pid = os.fork()\n    except OSError:\n        break\n"
        "    if pid == 0:\n        time.sleep(5); os._exit(0)\n    n += 1\n"
        "print(n)\n"
    )
    test = "import subprocess\nout = subprocess.run(['python3','solution.py'], capture_output=True, text=True).stdout.strip()\nprint('FORKS', out)\n"
    out = run(attack, test)["stdout"]
    assert "FORKS" in out  # the sandbox really ran
    forks = int(out.split("FORKS")[1].split()[0])
    assert forks < 1000
