"""Run truncation experiments on small graphs mirroring the truncation demo logic.

For each requested graph we generate the full, closed and some truncated
systems, solve them with a shared set of initial conditions, and compare the
truncated trajectories to the closed/full baselines. Generation and solve times
plus aggregate error metrics are appended to a CSV for later analysis.
"""

from __future__ import annotations

import argparse
import copy
import csv
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import sympy as sym

# Ensure local src imports when run directly
import sys as _sys
import pathlib as _pl

_root = _pl.Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

from equation.generation import generate_equations
from equation.solving import initial_conditions, solve_equations
from model_params.cmodel import get_SIR

EquationDict = Dict[int, Sequence[sym.Eq]]


@dataclass(frozen=True)
class GraphSpec:
    """Describe a named graph whose builder returns a fresh networkx object."""

    name: str
    builder: Callable[[], nx.Graph]


@dataclass(frozen=True)
class VariantConfig:
    """Configuration for a particular system (full/closed/truncated)."""

    label: str
    kind: str  # "full", "closed", or "truncated"
    closures: bool
    cap: Optional[int]


@dataclass
class VariantResult:
    """Hold timings, solution metadata, and errors for a single variant."""

    graph_name: str
    num_vertices: int
    config: VariantConfig
    equation_count: int = 0
    generation_time: float = 0.0
    solve_time: Optional[float] = None
    solve_success: bool = False
    generation_error: Optional[str] = None
    solve_error: Optional[str] = None
    equations: Optional[EquationDict] = None
    lhs: Sequence[sym.Expr] = ()
    solution: Any = None
    mean_abs_error_full: Optional[float] = None
    max_abs_error_full: Optional[float] = None
    mean_abs_error_closed: Optional[float] = None
    max_abs_error_closed: Optional[float] = None


def build_lollipop_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])
    return graph


def _path_builder(size: int) -> Callable[[], nx.Graph]:
    def _builder() -> nx.Graph:
        return nx.path_graph(size)

    return _builder


GRAPH_SPECS: Dict[str, GraphSpec] = {
    "lollipop": GraphSpec("lollipop", build_lollipop_graph),
    "path_5": GraphSpec("path_5", _path_builder(5)),
    "path_10": GraphSpec("path_10", _path_builder(10)),
}
DEFAULT_GRAPHS = tuple(GRAPH_SPECS.keys())
DEFAULT_CAP_FRACTIONS = (0.25, 0.5, 0.75)

CSV_FIELDS = [
    "timestamp",
    "hostname",
    "graph",
    "num_vertices",
    "variant",
    "variant_kind",
    "closures",
    "cap",
    "equation_count",
    "generation_time",
    "solve_time",
    "solve_success",
    "mean_abs_error_full",
    "max_abs_error_full",
    "mean_abs_error_closed",
    "max_abs_error_closed",
    "generation_error",
    "solve_error",
]


def collect_lhs_functions(equations: EquationDict) -> List[sym.Expr]:
    functions: List[sym.Expr] = []
    for eq_list in equations.values():
        for eq in eq_list:
            functions.append(sym.Integral(eq.lhs).doit())
    return functions


def compute_caps(num_vertices: int, fractions: Sequence[float]) -> List[int]:
    raw = {max(1, int(round(num_vertices * frac))) for frac in fractions}
    return sorted(raw)


def variant_plan(num_vertices: int, fractions: Sequence[float]) -> List[VariantConfig]:
    variants = [
        VariantConfig("full", "full", closures=False, cap=None),
        VariantConfig("closed", "closed", closures=True, cap=None),
    ]
    for cap in compute_caps(num_vertices, fractions):
        variants.append(
            VariantConfig(label=f"cap_{cap}", kind="truncated", closures=False, cap=cap)
        )
    return variants


def generate_variant(
    graph_name: str,
    num_vertices: int,
    graph: nx.Graph,
    model: Any,
    config: VariantConfig,
) -> VariantResult:
    start = time.perf_counter()
    try:
        equations = generate_equations(
            copy.deepcopy(graph),
            model,
            closures=config.closures,
            term_cap=config.cap,
        )
        gen_time = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001
        return VariantResult(
            graph_name=graph_name,
            num_vertices=num_vertices,
            config=config,
            generation_time=0.0,
            generation_error=str(exc),
        )

    eq_count = sum(len(group) for group in equations.values())
    lhs = collect_lhs_functions(equations)
    return VariantResult(
        graph_name=graph_name,
        num_vertices=num_vertices,
        config=config,
        equation_count=eq_count,
        generation_time=gen_time,
        equations=equations,
        lhs=lhs,
    )


def build_initial_conditions(
    graph: nx.Graph,
    results: Sequence[VariantResult],
    num_initial_infected: int,
) -> Optional[Dict[sym.Expr, float]]:
    lhs_pool: List[sym.Expr] = []
    for res in results:
        lhs_pool.extend(res.lhs)
    if not lhs_pool:
        return None
    deduped = list(dict.fromkeys(lhs_pool))
    return initial_conditions(
        list(graph.nodes),
        deduped,
        num_initial_infected=num_initial_infected,
        symbol=0,
    )


def solve_variant(
    graph: nx.Graph,
    init_cond: Dict[sym.Expr, float],
    t_max: float,
    result: VariantResult,
) -> None:
    if result.equations is None:
        return
    start = time.perf_counter()
    try:
        solution = solve_equations(result.equations, init_cond, graph, t_max)
        result.solve_time = time.perf_counter() - start
        result.solve_success = bool(getattr(solution, "success", False))
        result.solution = solution
    except Exception as exc:  # noqa: BLE001
        result.solve_time = None
        result.solve_success = False
        result.solve_error = str(exc)


def _total_infected(solution: Any) -> Optional[np.ndarray]:
    y = np.asarray(getattr(solution, "y", []))
    if y.ndim != 2:
        return None
    infected_rows = list(range(1, y.shape[0], 3))
    if not infected_rows:
        return None
    return y[infected_rows, :].sum(axis=0)


def compare_total_infected(reference: Any, candidate: Any) -> Optional[Tuple[float, float]]:
    ref_curve = _total_infected(reference)
    cand_curve = _total_infected(candidate)
    if ref_curve is None or cand_curve is None:
        return None
    t_ref = np.asarray(getattr(reference, "t", []))
    t_cand = np.asarray(getattr(candidate, "t", []))
    if t_ref.size == 0 or t_cand.size == 0:
        return None
    cand_interp = np.interp(t_ref, t_cand, cand_curve)
    diff = np.abs(cand_interp - ref_curve)
    return float(diff.mean()), float(diff.max())


def attach_total_infected_errors(results: Sequence[VariantResult]) -> None:
    full = next((r for r in results if r.config.kind == "full" and r.solve_success), None)
    closed = next((r for r in results if r.config.kind == "closed" and r.solve_success), None)
    for res in results:
        if not res.solve_success or res.config.kind == "full":
            continue
        if full is not None:
            metrics = compare_total_infected(full.solution, res.solution)
            if metrics is not None:
                res.mean_abs_error_full, res.max_abs_error_full = metrics
        if closed is not None and res.config.kind != "closed":
            metrics = compare_total_infected(closed.solution, res.solution)
            if metrics is not None:
                res.mean_abs_error_closed, res.max_abs_error_closed = metrics


def run_for_graph(
    spec: GraphSpec,
    cap_fractions: Sequence[float],
    model: Any,
    num_initial_infected: int,
    t_max: float,
) -> List[VariantResult]:
    graph = spec.builder()
    num_vertices = graph.number_of_nodes()
    configs = variant_plan(num_vertices, cap_fractions)
    results = [generate_variant(spec.name, num_vertices, graph, model, cfg) for cfg in configs]

    init_cond = build_initial_conditions(graph, results, num_initial_infected)
    if init_cond is None:
        return results

    for res in results:
        solve_variant(graph, init_cond, t_max, res)

    attach_total_infected_errors(results)
    return results


def row_from_result(result: VariantResult, hostname: str) -> Dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": hostname,
        "graph": result.graph_name,
        "num_vertices": result.num_vertices,
        "variant": result.config.label,
        "variant_kind": result.config.kind,
        "closures": result.config.closures,
        "cap": result.config.cap,
        "equation_count": result.equation_count,
        "generation_time": result.generation_time,
        "solve_time": result.solve_time,
        "solve_success": result.solve_success,
        "mean_abs_error_full": result.mean_abs_error_full,
        "max_abs_error_full": result.max_abs_error_full,
        "mean_abs_error_closed": result.mean_abs_error_closed,
        "max_abs_error_closed": result.max_abs_error_closed,
        "generation_error": result.generation_error,
        "solve_error": result.solve_error,
    }


def append_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run truncation sweeps on small graphs")
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=list(DEFAULT_GRAPHS),
        choices=sorted(GRAPH_SPECS.keys()),
        help="Subset of graph names to evaluate",
    )
    parser.add_argument(
        "--cap-fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_CAP_FRACTIONS),
        help="Fractions of |V| used to derive truncation caps",
    )
    parser.add_argument("--num-initial-infected", type=int, default=1)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fatanode_truncation_summary.csv"),
        help="CSV file to append results to",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    specs = [GRAPH_SPECS[name] for name in args.graphs]
    model = get_SIR(args.beta, args.gamma)
    hostname = socket.gethostname()

    all_rows: List[Dict[str, Any]] = []
    for spec in specs:
        graph = spec.builder()
        print(f"Running {spec.name} graph (n={graph.number_of_nodes()})")
        results = run_for_graph(
            spec=spec,
            cap_fractions=args.cap_fractions,
            model=model,
            num_initial_infected=args.num_initial_infected,
            t_max=args.t_max,
        )
        for res in results:
            all_rows.append(row_from_result(res, hostname))
        print(f"  Completed {spec.name}: {len(results)} variants")

    append_rows(args.output, all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
