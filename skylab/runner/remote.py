"""Remote GPU execution backend via SSH."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from skylab.runner.base import RunResult
from skylab.runner.local import extract_metrics


@dataclass
class RemoteConfig:
    """SSH configuration for remote execution."""

    host: str
    user: str
    key_path: str
    workspace: str = "/workspace/skylab"
    num_gpus: int = 1
    use_container: bool = False
    run_timeout: int = 900


def load_remote_config(experiment_dir: Path) -> RemoteConfig:
    """Load remote.toml from experiment directory or project root."""
    for search_path in [experiment_dir / "remote.toml", Path("remote.toml")]:
        if search_path.exists():
            return _parse_remote_toml(search_path)
    raise FileNotFoundError("No remote.toml found")


def _parse_remote_toml(path: Path) -> RemoteConfig:
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            data[k.strip()] = v.strip().strip('"').strip("'")

    return RemoteConfig(
        host=data["host"],
        user=data["user"],
        key_path=data["key_path"],
        workspace=data.get("workspace", "/workspace/skylab"),
        num_gpus=int(data.get("num_gpus", "1")),
        use_container=data.get("use_container", "false").lower() == "true",
        run_timeout=int(data.get("run_timeout", "900")),
    )


class RemoteRunner:
    """Runs training on a remote GPU host via SSH + rsync."""

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config

    def execute(
        self,
        experiment_dir: str,
        command: str,
        log_file: str,
        timeout: int,
    ) -> RunResult:
        exp_path = Path(experiment_dir)
        cfg = self.config

        # Sync code to remote
        self._rsync(exp_path)

        # Build remote command
        if cfg.num_gpus > 1:
            # Replace "uv run train.py" with torchrun equivalent
            remote_cmd = command.replace(
                "uv run train.py",
                f"uv run torchrun --nproc_per_node={cfg.num_gpus} train.py",
            )
        else:
            remote_cmd = command

        full_cmd = f"cd {cfg.workspace} && timeout {cfg.run_timeout} {remote_cmd}"

        start = time.monotonic()
        timed_out = False

        try:
            result = subprocess.run(
                self._ssh_cmd(full_cmd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
            # exit code 124 from timeout(1) means the command timed out
            if exit_code == 124:
                timed_out = True
        except subprocess.TimeoutExpired as e:
            exit_code = -1
            stdout = (e.stdout or b"").decode(errors="replace")
            stderr = (e.stderr or b"").decode(errors="replace")
            timed_out = True

        duration = time.monotonic() - start

        # Write log locally
        log_path = exp_path / log_file
        log_path.write_text(stdout + stderr)

        metrics = extract_metrics(stdout + stderr)

        return RunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=timed_out,
            metrics=metrics,
        )

    def check(self) -> bool:
        """Verify SSH connectivity and GPU availability."""
        try:
            result = subprocess.run(
                self._ssh_cmd("nvidia-smi --query-gpu=name --format=csv,noheader"),
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _ssh_cmd(self, remote_command: str) -> list[str]:
        cfg = self.config
        return [
            "ssh",
            "-i",
            cfg.key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            f"{cfg.user}@{cfg.host}",
            remote_command,
        ]

    def _rsync(self, experiment_dir: Path) -> None:
        cfg = self.config
        excludes = [
            ".git",
            ".venv",
            "__pycache__",
            "run.log",
            "results.tsv",
            "remote.toml",
            "*.db",
        ]
        cmd = [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -i {cfg.key_path} -o StrictHostKeyChecking=no",
        ]
        for exc in excludes:
            cmd.extend(["--exclude", exc])
        cmd.extend([f"{experiment_dir}/", f"{cfg.user}@{cfg.host}:{cfg.workspace}/"])

        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
