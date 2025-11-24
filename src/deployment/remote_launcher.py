from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from experiments.config import ExperimentConfig

DEFAULT_COMPUTE_NODES = [f"fatanode-{i:02d}" for i in range(1, 19)]
DEFAULT_EXCLUDES = [".git", "__pycache__", "*.pyc", "*.pyo", "*.swp", "data/experiment_results.csv"]


@dataclass
class RemoteClusterConfig:
    gateway: Optional[str]
    username: str = "ekelly"
    head_node: str = "fatanode-head"
    storage_node: str = "fatanode-data"
    compute_nodes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_COMPUTE_NODES))
    remote_base_dir: str = "~/DynamicalGraphModel"
    python_executable: str = "python3"
    max_parallel_jobs: int = 12
    rsync_excludes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    extra_run_args: Sequence[str] = field(default_factory=list)


class RemoteExperimentOrchestrator:
    def __init__(self, cluster: RemoteClusterConfig, dry_run: bool = False) -> None:
        self.cluster = cluster
        self.dry_run = dry_run
        self._common_ssh_options = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
        ]

    @property
    def _proxy_jump(self) -> Optional[str]:
        if not self.cluster.gateway:
            return None
        return f"{self.cluster.username}@{self.cluster.gateway}"

    def _ssh_base(self) -> List[str]:
        return ["ssh", *self._common_ssh_options]

    def _ssh_with_jumps(self, host: str, via_head: bool = False, background: bool = False) -> List[str]:
        cmd = self._ssh_base()
        if background:
            # Detach the ssh client so remote experiments don't block local dispatch.
            cmd.insert(1, "-f")
            cmd.insert(2, "-n")
        jumps: List[str] = []
        proxy = self._proxy_jump
        if proxy:
            jumps.append(proxy)
            if via_head:
                jumps.append(f"{self.cluster.username}@{self.cluster.head_node}")
        if jumps:
            cmd.extend(["-J", ",".join(jumps)])
        cmd.append(f"{self.cluster.username}@{host}")
        return cmd

    def sync_project(self, local_root: Path) -> None:
        if not self.cluster.gateway:
            print("[remote] Sync project to head node:")
            print("           skipping (running on head node)")
            return
        dest = f"{self.cluster.username}@{self.cluster.head_node}:{self.cluster.remote_base_dir}/"
        excludes = [item for pattern in self.cluster.rsync_excludes for item in ("--exclude", pattern)]
        ssh_cmd_parts = self._ssh_base()
        proxy = self._proxy_jump
        if proxy:
            ssh_cmd_parts.extend(["-J", proxy])
        ssh_cmd = " ".join(ssh_cmd_parts)
        cmd = ["rsync", "-az", "--delete", *excludes, "-e", ssh_cmd, f"{local_root}/", dest]
        self._run(cmd, description="Sync project to head node")

    def push_config(self, local_config: Path, remote_config: str) -> None:
        if not self.cluster.gateway:
            dest_path = Path(remote_config).expanduser()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(local_config.read_text())
            print(f"[remote] Upload experiment config:\n           copied locally to {dest_path}")
            return
        cmd = ["scp", *self._common_ssh_options]
        proxy = self._proxy_jump
        if proxy:
            cmd.extend(["-o", f"ProxyJump={proxy}"])
        cmd.extend([str(local_config), f"{self.cluster.username}@{self.cluster.head_node}:{remote_config}"])
        self._run(cmd, description="Upload experiment config")

    def ensure_remote_dirs(self, remote_paths: Iterable[str]) -> None:
        if not self.cluster.gateway:
            for path in remote_paths:
                Path(path).expanduser().mkdir(parents=True, exist_ok=True)
            print("[remote] Create remote directories:\n           ensured locally")
            return
        command = " && ".join(f"mkdir -p {path}" for path in remote_paths)
        self._ssh(self.cluster.head_node, command, description="Create remote directories")

    def launch_partitions(
        self,
        remote_config: str,
        remote_output: str,
        remote_log_dir: str,
        partition_count: int,
    ) -> None:
        targets = list(self.cluster.compute_nodes)[: partition_count]
        for idx, node in enumerate(targets):
            self._launch_on_node(node, idx, partition_count, remote_config, remote_output, remote_log_dir)

    def _launch_on_node(
        self,
        node: str,
        partition_index: int,
        partition_count: int,
        remote_config: str,
        remote_output: str,
        remote_log_dir: str,
    ) -> None:
        log_file = f"{remote_log_dir}/partition-{partition_index:02d}.log"

        def _fmt_path(value: str) -> str:
            if value.startswith("~/"):
                return f"$HOME/{value[2:]}"
            return shlex.quote(value)

        def _fmt_arg(token: str) -> str:
            if token.startswith("$HOME/"):
                return token
            return shlex.quote(token)

        run_args = [
            _fmt_path(self.cluster.python_executable),
            "-m",
            "experiments.run_experiments",
            "--config",
            _fmt_path(remote_config),
            "--output",
            _fmt_path(remote_output),
            "--partition-index",
            str(partition_index),
            "--partition-count",
            str(partition_count),
        ]
        if self.cluster.extra_run_args:
            run_args.extend(self.cluster.extra_run_args)
        run_cmd = " ".join(_fmt_arg(arg) for arg in run_args)
        inner_cmd = " && ".join(
            [
                f"cd {self.cluster.remote_base_dir}",
                f"mkdir -p {remote_log_dir}",
                "export PYTHONPATH=${PYTHONPATH}:$PWD/src",
                f"nohup {run_cmd} > {_fmt_path(log_file)} 2>&1 &",
            ]
        )
        cmd = self._ssh_with_jumps(node, via_head=True, background=True)
        cmd.append(inner_cmd)
        self._run(cmd, description=f"Launch partition {partition_index}/{partition_count} on {node}")

    def _ssh(self, host: str, command: str, description: str) -> None:
        if not self.cluster.gateway and host in {self.cluster.head_node, "localhost", "127.0.0.1"}:
            self._run(["bash", "-lc", command], description)
            return
        cmd = self._ssh_with_jumps(host)
        cmd.append(command)
        self._run(cmd, description=description)

    def _run(self, command: List[str], description: str) -> None:
        print(f"[remote] {description}:")
        print("          ", " ".join(command))
        if self.dry_run:
            return
        subprocess.run(command, check=True)


def load_experiments(config_path: Path) -> List[ExperimentConfig]:
    content = json.loads(config_path.read_text())
    if isinstance(content, dict) and "experiments" in content:
        content = content["experiments"]
    return [ExperimentConfig.from_dict(item) for item in content]


def calculate_partitions(configs: Sequence[ExperimentConfig], max_parallel: int, available_nodes: int) -> int:
    return max(1, min(len(configs), max_parallel, available_nodes))


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy experiments to remote fatanode cluster")
    parser.add_argument("--gateway", help="SSH gateway hostname; omit if already on head node")
    parser.add_argument(
        "--local-config",
        type=Path,
        default=Path("src/experiments/default_config.json"),
        help="Local experiment config JSON",
    )
    parser.add_argument("--local-root", type=Path, default=Path.cwd())
    parser.add_argument("--remote-config", default="~/DynamicalGraphModel/experiments.json")
    parser.add_argument("--remote-output", default="~/DynamicalGraphModel/data/remote_results.csv")
    parser.add_argument(
        "--remote-log-dir",
        default=None,
        help="Remote log directory; defaults to <remote-base-dir>/logs/default-<timestamp>",
    )
    parser.add_argument("--remote-base-dir", default="~/DynamicalGraphModel")
    parser.add_argument("--python", default="~/venvs/dgm/bin/python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--head-node", default="fatanode-head")
    parser.add_argument("--storage-node", default="fatanode-data")
    parser.add_argument(
        "--compute-nodes",
        nargs="*",
        default=None,
        help="Compute nodes to target",
    )
    args = parser.parse_args()

    compute_nodes = args.compute_nodes or [f"fatanode-{i:02d}" for i in range(1, 7)]
    max_parallel = args.max_parallel or len(compute_nodes)
    remote_base_dir = args.remote_base_dir
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    remote_log_dir = args.remote_log_dir or f"{remote_base_dir.rstrip('/')}/logs/default-{timestamp}"
    remote_config = args.remote_config
    remote_output = args.remote_output

    cluster = RemoteClusterConfig(
        gateway=args.gateway,
        head_node=args.head_node,
        storage_node=args.storage_node,
        compute_nodes=compute_nodes,
        remote_base_dir=remote_base_dir,
        python_executable=args.python,
        max_parallel_jobs=max_parallel,
    )

    orchestrator = RemoteExperimentOrchestrator(cluster, dry_run=args.dry_run)
    local_config = args.local_config.expanduser().resolve()
    configs = load_experiments(local_config)
    partitions = calculate_partitions(configs, cluster.max_parallel_jobs, len(cluster.compute_nodes))

    print(f"Planning {len(configs)} experiments across {partitions} partitions")

    orchestrator.sync_project(args.local_root.expanduser().resolve())
    orchestrator.push_config(local_config, remote_config)
    orchestrator.ensure_remote_dirs([remote_log_dir, str(Path(remote_output).parent)])
    orchestrator.launch_partitions(remote_config, remote_output, remote_log_dir, partitions)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
