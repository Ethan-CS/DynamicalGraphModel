from __future__ import annotations

import argparse
import csv
import json
import math
import random
import signal
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import networkx as nx
import numpy as np
import sympy as sym

from equation.generation import generate_equations
from equation.solving import initial_conditions, solve_equations
from experiments.config import ExperimentConfig
from model_params.cmodel import CModel
from monte_carlo.mc_sim import MonteCarloTimeoutError, run_to_average, set_initial_state


RESULT_FIELDS = [
    "timestamp",
    "graph_type",
    "num_vertices",
    "graph_params",
    "method",
    "iteration",
    "status",
    "runtime_seconds",
    "num_equations",
    "unique_equations",
    "max_equation_length",
    "solve_runtime_seconds",
    "mc_rounds",
    "mc_final_mean",
    "timeout_seconds",
    "seed",
    "notes",
]

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")
CLI_DEFAULTS = {
    "graph_types": ["erdos_renyi"],
    "graph_param": [],
    "num_vertices": 100,
    "iterations": 5,
    "timeout": 600,
    "t_max": 15,
    "method": "both",
    "disable_closures": False,
    "solve_equations": False,
    "num_initial_infected": 1,
    "tolerance": 1e-2,
    "seed": None,
    "term_length_cap": None,
}


@contextmanager
def time_limit(seconds: Optional[float]):
    if seconds is None or seconds <= 0:
        yield
        return

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("time_limit may only be used from the main thread")

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"Operation exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class TimeoutBudget:
    def __init__(self, seconds: Optional[float]):
        self._total = float(seconds) if seconds and seconds > 0 else None
        self._deadline = time.monotonic() + self._total if self._total else None

    def remaining(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    @contextmanager
    def limit(self, label: str):
        remaining = self.remaining()
        if remaining is None:
            yield
            return
        if remaining <= 0:
            total_desc = f" ({self._total:.2f} seconds)" if self._total is not None else ""
            raise TimeoutError(f"Timeout expired before {label} started{total_desc}")
        try:
            with time_limit(remaining):
                yield
        except TimeoutError as exc:
            total_desc = f" ({self._total:.2f} seconds)" if self._total is not None else ""
            raise TimeoutError(f"{label} exceeded allotted timeout{total_desc}") from exc


def build_graph(config: ExperimentConfig) -> nx.Graph:
    params = dict(config.graph_params)
    seed = params.pop("seed", config.seed)
    graph_type = config.graph_type.lower()

    if graph_type == "path":
        return nx.path_graph(config.num_vertices)
    if graph_type == "cycle":
        return nx.cycle_graph(config.num_vertices)
    if graph_type == "complete":
        return nx.complete_graph(config.num_vertices)
    if graph_type == "grid":
        width = int(params.get("width", round(math.sqrt(config.num_vertices))))
        height = int(params.get("height", math.ceil(config.num_vertices / width)))
        g = nx.grid_2d_graph(width, height)
        mapping = {node: idx for idx, node in enumerate(g.nodes())}
        relabelled = nx.relabel_nodes(g, mapping)
        if relabelled.number_of_nodes() > config.num_vertices:
            nodes = list(range(config.num_vertices))
            return relabelled.subgraph(nodes).copy()
        return relabelled
    if graph_type in {"erdos_renyi", "gnp"}:
        p = params.get("p")
        if p is None:
            raise ValueError("erdos_renyi graphs require a 'p' parameter")
        return nx.erdos_renyi_graph(config.num_vertices, p, seed=seed)
    if graph_type == "watts_strogatz":
        k = params.get("k")
        p = params.get("p")
        if k is None or p is None:
            raise ValueError("watts_strogatz graphs require 'k' and 'p' parameters")
        return nx.watts_strogatz_graph(config.num_vertices, int(k), float(p), seed=seed)
    if graph_type == "barabasi_albert":
        m = params.get("m")
        if m is None:
            raise ValueError("barabasi_albert graphs require an 'm' parameter")
        return nx.barabasi_albert_graph(config.num_vertices, int(m), seed=seed)
    if graph_type == "random_regular":
        d = params.get("d")
        if d is None:
            raise ValueError("random_regular graphs require a 'd' parameter")
        return nx.random_regular_graph(int(d), config.num_vertices, seed=seed)
    if graph_type in {"random_geometric", "geometric"}:
        radius = params.get("r", params.get("radius"))
        if radius is None:
            raise ValueError("random_geometric graphs require an 'r' (radius) parameter")
        return nx.random_geometric_graph(config.num_vertices, float(radius), seed=seed)

    raise ValueError(f"Unsupported graph type: {config.graph_type}")


def collect_lhs_functions(equations: Dict[int, Sequence[sym.Eq]]) -> List[sym.Expr]:
    functions: List[sym.Expr] = []
    for eq_list in equations.values():
        for eq in eq_list:
            functions.append(cast(sym.Expr, sym.Integral(eq.lhs).doit()))
    return functions


def count_equations(equations: Dict[int, Sequence[sym.Eq]]) -> int:
    return sum(len(eq_list) for eq_list in equations.values())


def count_unique_equations(equations: Dict[int, Sequence[sym.Eq]]) -> int:
    unique = set()
    for eq_list in equations.values():
        for eq in eq_list:
            unique.add(sym.srepr(eq))
    return len(unique)


def run_equations_trial(config: ExperimentConfig, graph: nx.Graph, model: CModel, iteration: int) -> Dict[str, object]:
    row = base_row(config, method="equations", iteration=iteration)
    start = time.monotonic()
    budget = TimeoutBudget(config.timeout)

    try:
        with budget.limit("equation generation"):
            equations = generate_equations(
                graph,
                model,
                closures=config.use_closures,
                term_cap=config.term_length_cap,
            )
        typed_equations = cast(Dict[int, Sequence[sym.Eq]], equations)
        if not isinstance(equations, dict):
            raise RuntimeError("Equation generation returned unexpected data structure")
        row["num_equations"] = count_equations(typed_equations)
        row["unique_equations"] = count_unique_equations(typed_equations)
        row["max_equation_length"] = max(typed_equations.keys()) if typed_equations else 0

        if config.solve_equations:
            lhs_functions = collect_lhs_functions(typed_equations)
            init_cond = initial_conditions(
                list(graph.nodes),
                lhs_functions,
                num_initial_infected=config.num_initial_infected,
                symbol=sym.symbols("t"),
            )
            solve_start = time.monotonic()
            with budget.limit("equation solving"):
                solve_equations(typed_equations, init_cond, graph, config.t_max)
            row["solve_runtime_seconds"] = time.monotonic() - solve_start

        row["runtime_seconds"] = time.monotonic() - start
        row["status"] = "ok"
    except TimeoutError as exc:
        row["status"] = "timeout"
        row["runtime_seconds"] = time.monotonic() - start
        row["notes"] = str(exc)
    except Exception as exc:  # pylint: disable=broad-except
        row["status"] = "error"
        row["runtime_seconds"] = time.monotonic() - start
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return row


def run_monte_carlo_trial(config: ExperimentConfig, graph: nx.Graph, model: CModel, iteration: int) -> Dict[str, object]:
    row = base_row(config, method="monte_carlo", iteration=iteration)
    start = time.monotonic()
    budget = TimeoutBudget(config.timeout)

    init_state = set_initial_state(model, graph)
    try:
        mc_timeout = budget.remaining() or config.timeout
        with budget.limit("monte carlo simulation"):
            averages = run_to_average(
                graph,
                model,
                init_state,
                config.t_max,
                solution=config.target_average,
                tolerance=config.tolerance,
                timeout=mc_timeout,
                num_rounds=max(5, config.iterations),
            )
        row["runtime_seconds"] = time.monotonic() - start
        row["mc_rounds"] = len(averages.index)
        if len(averages.index) > 0:
            row["mc_final_mean"] = json.dumps([float(averages[i].iloc[-1]) for i in averages.columns])
        row["status"] = "ok"
    except MonteCarloTimeoutError as exc:
        row["status"] = "timeout"
        row["runtime_seconds"] = time.monotonic() - start
        averages = exc.averages
        row["mc_rounds"] = len(averages.index)
        if len(averages.index) > 0:
            row["mc_final_mean"] = json.dumps([float(averages[i].iloc[-1]) for i in averages.columns])
        row["notes"] = str(exc)
    except TimeoutError as exc:
        row["status"] = "timeout"
        row["runtime_seconds"] = time.monotonic() - start
        row["notes"] = str(exc)
    except Exception as exc:  # pylint: disable=broad-except
        row["status"] = "error"
        row["runtime_seconds"] = time.monotonic() - start
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return row


def base_row(config: ExperimentConfig, method: str, iteration: int) -> Dict[str, object]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_type": config.graph_type,
        "num_vertices": config.num_vertices,
        "graph_params": json.dumps(config.graph_params, sort_keys=True, default=str),
        "method": method,
        "iteration": iteration,
        "status": "pending",
        "runtime_seconds": None,
        "num_equations": None,
        "unique_equations": None,
        "max_equation_length": None,
        "solve_runtime_seconds": None,
        "mc_rounds": None,
        "mc_final_mean": None,
        "timeout_seconds": config.timeout,
        "seed": config.seed,
        "notes": None,
    }


def build_model(beta: float, gamma: float) -> CModel:
    model = CModel.make_SIR(beta=beta, gamma=gamma)
    return model


def run_experiment(config: ExperimentConfig, beta: float, gamma: float) -> List[Dict[str, object]]:
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    graph = build_graph(config)
    model = build_model(beta, gamma)
    results: List[Dict[str, object]] = []

    for iteration in range(config.iterations):
        if config.method in {"equations", "both"}:
            results.append(run_equations_trial(config, graph, model, iteration))
        if config.method in {"monte_carlo", "mc", "both"}:
            results.append(run_monte_carlo_trial(config, graph, model, iteration))

    return results


def load_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    if args.config is not None:
        return load_config_file(Path(args.config))

    if all(getattr(args, key) == value for key, value in CLI_DEFAULTS.items()):
        if not DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Default experiment config not found at {DEFAULT_CONFIG_PATH}")
        print(f"Using default experiment config at {DEFAULT_CONFIG_PATH}")
        return load_config_file(DEFAULT_CONFIG_PATH)

    param_dict = parse_graph_params(args.graph_param)
    configs = [
        ExperimentConfig(
            graph_type=graph_type,
            num_vertices=args.num_vertices,
            graph_params=param_dict,
            method=args.method,
            iterations=args.iterations,
            timeout=args.timeout,
            t_max=args.t_max,
            use_closures=not args.disable_closures,
            term_length_cap=args.term_length_cap,
            solve_equations=args.solve_equations,
            num_initial_infected=args.num_initial_infected,
            tolerance=args.tolerance,
            seed=args.seed,
        )
        for graph_type in args.graph_types
    ]
    return configs


def load_config_file(path: Path) -> List[ExperimentConfig]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "experiments" in raw:
        experiments = raw["experiments"]
    else:
        experiments = raw
    return [ExperimentConfig.from_dict(item) for item in experiments]


def parse_graph_params(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not pairs:
        return params
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Graph parameter '{pair}' must use key=value syntax")
        key, value = pair.split("=", 1)
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return params


def partition(configs: List[ExperimentConfig], index: Optional[int], total: Optional[int]) -> List[ExperimentConfig]:
    if index is None or total is None:
        return configs
    if not 0 <= index < total:
        raise ValueError("partition index must be within [0, total)")
    return [config for idx, config in enumerate(configs) if idx % total == index]


def write_results(path: Path, results: Iterable[Dict[str, object]]) -> None:
    results = list(results)
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run runtime measurement experiments")
    parser.add_argument("--config", type=str, help="Path to JSON experiment configuration file")
    parser.add_argument("--graph-types", nargs="+", default=["erdos_renyi"], help="Graph families to run")
    parser.add_argument("--graph-param", action="append", default=[], dest="graph_param",
                        help="Graph parameter overrides expressed as key=value")
    parser.add_argument("--num-vertices", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--t-max", type=int, default=15, dest="t_max")
    parser.add_argument("--method", choices=["equations", "monte_carlo", "mc", "both"], default="both")
    parser.add_argument("--disable-closures", action="store_true")
    parser.add_argument("--solve-equations", action="store_true")
    parser.add_argument("--num-initial-infected", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--term-length-cap", type=int, dest="term_length_cap",
                        help="Maximum term size that should receive its own equation")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="data/experiment_results.csv")
    parser.add_argument("--partition-index", type=int)
    parser.add_argument("--partition-count", type=int)

    args = parser.parse_args(argv)

    configs = load_configs(args)
    configs = partition(configs, args.partition_index, args.partition_count)

    if not configs:
        print("No experiments scheduled after partitioning; exiting.")
        return 0

    output_path = Path(args.output)
    total = 0
    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] Running experiment set for graph_type={config.graph_type} "
            f"(method={config.method}, iterations={config.iterations})",
            flush=True,
        )
        try:
            results = run_experiment(config, beta=args.beta, gamma=args.gamma)
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[{index}/{len(configs)}] FAILED ({type(exc).__name__}: {exc})",
                flush=True,
            )
            continue

        write_results(output_path, results)
        total += len(results)
        print(
            f"[{index}/{len(configs)}] Appended {len(results)} rows (cumulative {total}) to {output_path}",
            flush=True,
        )

    print(f"Recorded {total} result rows to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
