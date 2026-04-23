from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_sdk_cli_runs_from_staged_site_packages(tmp_path: Path) -> None:
    repo_root = Path("/workspace/unicorn/greencompute-ai/greencompute")
    sdk_src = repo_root / "sdk/src/greencompute"
    protocol_src = repo_root / "protocol/src/greencompute_protocol"
    staged_site_packages = tmp_path / "site-packages"
    staged_site_packages.mkdir()

    shutil.copytree(sdk_src, staged_site_packages / "greencompute")
    shutil.copytree(protocol_src, staged_site_packages / "greencompute_protocol")

    dependency_site_packages = repo_root / ".venv/lib/python3.12/site-packages"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(staged_site_packages), str(dependency_site_packages)])

    command = [
        sys.executable,
        "-c",
        (
            "import greencompute, subprocess, sys; "
            "print(greencompute.__file__); "
            "subprocess.run([sys.executable, '-m', 'greencompute.cli', '--help'], check=True)"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout
    assert str(staged_site_packages / "greencompute") in output
    assert "GreenCompute SDK and CLI" in output
