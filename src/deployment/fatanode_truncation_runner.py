from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from deployment.remote_launcher import (
    RemoteClusterConfig,
    RemoteExperimentOrchestrator,
    calculate_partitions,
)
from experiments.truncation_plan import (
    DEFAULT_CAP_FRACTIONS,
    DEFAULT_GRAPH_SPECS,
    GraphSpec,
    build_truncation_experiments,
    write_experiment_config,
)


def _filter_specs(graph_types: Sequence[str]) -> List[GraphSpec]:
    if not graph_types:
        return list(DEFAULT_GRAPH_SPECS)
    wanted = {g.lower() for g in graph_types}
    return [spec for spec in DEFAULT_GRAPH_SPECS if spec.graph_type.lower() in wanted]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and launch truncation experiments across fatanode 01-04."
    )
    parser.add_argument("--gateway", help="SSH gateway hostname; omit when already on the head node")
    parser.add_argument("--username", default="ekelly", help="Cluster username")
    parser.add_argument("--head-node", default="fatanode-head", help="Head node hostname")
    parser.add_argument("--storage-node", default="fatanode-data", help="Storage node hostname")
    parser.add_argument(
        "--compute-nodes",
        nargs="*",
        default=[f"fatanode-{i:02d}" for i in range(1, 5)],
        help="Compute nodes to target",
    )
    parser.add_argument("--local-root", type=Path, default=Path.cwd(), help="Local repository root")
    parser.add_argument(
        "--local-config",
        type=Path,
        help="Where to write the generated experiment config",
    )
    parser.add_argument("--remote-base-dir", default="~/DynamicalGraphModel")
    parser.add_argument("--remote-config", help="Remote path for experiment config JSON")
    parser.add_argument("--remote-output", help="Remote CSV to append experiment rows to")
    parser.add_argument("--remote-solution-dir", help="Remote directory for serialized solutions")
    parser.add_argument("--remote-log-dir", help="Remote log directory for nohup output")
    parser.add_argument("--python", default="~/venvs/dgm/bin/python", help="Remote python executable")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--max-parallel", type=int, help="Override max parallel runs")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per experiment")
    parser.add_argument("--timeout", type=int, default=900, help="Per experiment timeout in seconds")
    parser.add_argument("--t-max", type=int, default=20, help="Time horizon for equation solver")
    parser.add_argument(
        "--num-initial-infected",
        type=int,
        default=1,
        help="How many vertices start infected in the ODE solves",
    )
    parser.add_argument(
        "--cap-fractions",
        nargs="*",
        type=float,
        default=list(DEFAULT_CAP_FRACTIONS),
        help="Fractions of |V| used as truncation caps",
    )
    parser.add_argument(
        "--graph-types",
        nargs="*",
        default=[],
        help="Subset of graph families to include (path, cycle, erdos_renyi, barabasi_albert)",
    )
    parser.add_argument("--plan-only", action="store_true", help="Generate config but skip remote launch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = _filter_specs(args.graph_types)
    if not specs:
        raise SystemExit("No graph specs available after filtering")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    experiments = build_truncation_experiments(
        graph_specs=specs,
        cap_fractions=args.cap_fractions,
        iterations=args.iterations,
        timeout=args.timeout,
        t_max=args.t_max,
        num_initial_infected=args.num_initial_infected,
    )

    local_config = args.local_config
    if local_config is None:
        local_config = Path("data/configs") / f"fatanode_truncation_{timestamp}.json"
    local_config = local_config.expanduser().resolve()
    write_experiment_config(local_config, experiments)
    print(f"Wrote {len(experiments)} experiments to {local_config}")

    if args.plan_only:
        return 0

    remote_base = args.remote_base_dir.rstrip("/")
    remote_config = args.remote_config or f"{remote_base}/experiments/fatanode_truncation_{timestamp}.json"
    remote_output = args.remote_output or f"{remote_base}/data/fatanode_truncation_{timestamp}.csv"
    remote_solution_dir = args.remote_solution_dir or f"{remote_base}/data/truncation_solutions/{timestamp}"
    remote_log_dir = args.remote_log_dir or f"{remote_base}/logs/truncation-{timestamp}"
    remote_paths = [remote_log_dir, str(Path(remote_output).parent), remote_solution_dir]

    compute_nodes = args.compute_nodes or [f"fatanode-{i:02d}" for i in range(1, 5)]
    max_parallel = args.max_parallel or len(compute_nodes)

    cluster = RemoteClusterConfig(
        gateway=args.gateway,
        username=args.username,
        head_node=args.head_node,
        storage_node=args.storage_node,
        compute_nodes=compute_nodes,
        remote_base_dir=args.remote_base_dir,
        python_executable=args.python,
        max_parallel_jobs=max_parallel,
        extra_run_args=["--solution-dir", remote_solution_dir],
    )
    orchestrator = RemoteExperimentOrchestrator(cluster, dry_run=args.dry_run)

    partitions = calculate_partitions(experiments, cluster.max_parallel_jobs, len(cluster.compute_nodes))
    print(
        f"Dispatching {len(experiments)} experiments across {partitions} partitions "
        f"onto {len(cluster.compute_nodes)} nodes"
    )

    orchestrator.sync_project(args.local_root.expanduser().resolve())
    orchestrator.push_config(local_config, remote_config)
    orchestrator.ensure_remote_dirs(remote_paths)
    orchestrator.launch_partitions(remote_config, remote_output, remote_log_dir, partitions)
    print("Remote launch initiated")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
