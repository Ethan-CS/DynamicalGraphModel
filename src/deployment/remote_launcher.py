from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from experiments.config import ExperimentConfig

DEFAULT_COMPUTE_NODES = [f"fatanode-{i:02d}" for i in range(1, 19)]
DEFAULT_EXCLUDES = [".git", "__pycache__", "*.pyc", "*.pyo", "*.swp", "data/experiment_results.csv"]


@dataclass
class RemoteClusterConfig:
    gateway: str
    username: str = "ekelly"
    head_node: str = "fatanode-head"
    storage_node: str = "fatanode-data"
    compute_nodes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_COMPUTE_NODES))
    remote_base_dir: str = "~/DynamicalGraphModel"
    python_executable: str = "python3"
    max_parallel_jobs: int = 12
    rsync_excludes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))


class RemoteExperimentOrchestrator:
    def __init__(self, cluster: RemoteClusterConfig, dry_run: bool = False) -> None:
        self.cluster = cluster
        self.dry_run = dry_run

    @property
    def _proxy_jump(self) -> str:
        return f"{self.cluster.username}@{self.cluster.gateway}"

    def sync_project(self, local_root: Path) -> None:
        dest = f"{self.cluster.username}@{self.cluster.head_node}:{self.cluster.remote_base_dir}/"
        excludes = [item for pattern in self.cluster.rsync_excludes for item in ("--exclude", pattern)]
        ssh_cmd = f"ssh -J {self._proxy_jump}"
        cmd = ["rsync", "-az", "--delete", *excludes, "-e", ssh_cmd, f"{local_root}/", dest]
        self._run(cmd, description="Sync project to head node")

    def push_config(self, local_config: Path, remote_config: str) -> None:
        cmd = [
            "scp",
            "-o",
            f"ProxyJump={self._proxy_jump}",
            str(local_config),
            f"{self.cluster.username}@{self.cluster.head_node}:{remote_config}",
        ]
        self._run(cmd, description="Upload experiment config")

    def ensure_remote_dirs(self, remote_paths: Iterable[str]) -> None:
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
        inner_cmd = " && ".join(
            [
                f"cd {self.cluster.remote_base_dir}",
                f"mkdir -p {remote_log_dir}",
                "export PYTHONPATH=${PYTHONPATH}:$PWD/src",
                "nohup "
                + shlex.join(
                    [
                        self.cluster.python_executable,
                        "-m",
                        "experiments.run_experiments",
                        "--config",
                        remote_config,
                        "--output",
                        remote_output,
                        "--partition-index",
                        str(partition_index),
                        "--partition-count",
                        str(partition_count),
                    ]
                )
                + f" > {log_file} 2>&1 &",
            ]
        )
        proxy = f"{self._proxy_jump},{self.cluster.username}@{self.cluster.head_node}"
        cmd = [
            "ssh",
            "-J",
            proxy,
            f"{self.cluster.username}@{node}",
            inner_cmd,
        ]
        self._run(cmd, description=f"Launch partition {partition_index}/{partition_count} on {node}")

    def _ssh(self, host: str, command: str, description: str) -> None:
        cmd = [
            "ssh",
            "-J",
            self._proxy_jump,
            f"{self.cluster.username}@{host}",
            command,
        ]
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
    parser.add_argument("--gateway", required=True, help="SSH gateway hostname")
    parser.add_argument("--local-config", required=True, type=Path)
    parser.add_argument("--local-root", type=Path, default=Path.cwd())
    parser.add_argument("--remote-config", default="~/DynamicalGraphModel/experiments.json")
    parser.add_argument("--remote-output", default="~/DynamicalGraphModel/data/remote_results.csv")
    parser.add_argument("--remote-log-dir", default="~/DynamicalGraphModel/logs")
    parser.add_argument("--remote-base-dir", default="~/DynamicalGraphModel")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=12)
    parser.add_argument("--head-node", default="fatanode-head")
    parser.add_argument("--storage-node", default="fatanode-data")
    parser.add_argument("--compute-nodes", nargs="*", default=DEFAULT_COMPUTE_NODES)
    args = parser.parse_args()

    cluster = RemoteClusterConfig(
        gateway=args.gateway,
        head_node=args.head_node,
        storage_node=args.storage_node,
        compute_nodes=args.compute_nodes,
        remote_base_dir=args.remote_base_dir,
        python_executable=args.python,
        max_parallel_jobs=args.max_parallel,
    )

    orchestrator = RemoteExperimentOrchestrator(cluster, dry_run=args.dry_run)
    local_config = args.local_config.expanduser().resolve()
    configs = load_experiments(local_config)
    partitions = calculate_partitions(configs, cluster.max_parallel_jobs, len(cluster.compute_nodes))

    print(f"Planning {len(configs)} experiments across {partitions} partitions")

    orchestrator.sync_project(args.local_root.expanduser().resolve())
    orchestrator.push_config(local_config, args.remote_config)
    orchestrator.ensure_remote_dirs([args.remote_log_dir, str(Path(args.remote_output).parent)])
    orchestrator.launch_partitions(args.remote_config, args.remote_output, args.remote_log_dir, partitions)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
