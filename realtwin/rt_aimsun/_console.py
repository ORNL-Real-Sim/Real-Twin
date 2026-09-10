"""Console execution shared by Aimsun calibration stages."""

import os
import subprocess


def run_aconsole(cmd: list[str] | str) -> tuple[int, str]:
    """Run an Aimsun command and capture its merged UTF-8 output.

    Keep Python output unbuffered because Aimsun can finish an operation and
    crash during native shutdown. Callers inspect the return code and generated
    artifacts to determine whether their operation succeeded.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    output, _ = process.communicate()
    return process.returncode, output
