from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import networkx as nx
import numpy as np
import sympy as sym

# Ensure local src imports when run as a script
import sys as _sys, pathlib as _pl
_root = _pl.Path(__file__).resolve().parents[2]
_src = _root / 'src'
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

from equation.generation import generate_equations
from equation.solving import initial_conditions, solve_equations
from model_params.cmodel import get_SIR

EquationDict = Dict[int, Sequence[sym.Eq]]

CAP_FRACTIONS_DEFAULT = (0.25, 0.5, 0.75)
RESULT_FIELDS = [
    "timestamp",
    "hostname",
    "graph_type",
    "seed",
    "num_vertices",
    "graph_params",
    "variant",
    "closures",
    "cap",
    "generation_time",
    "solve_time",
    "equation_count",
    "solve_success",
    "mean_abs_error_full",
    "max_abs_error_full",
    "mean_abs_error_closed",
    "max_abs_error_closed",
    "solution_final_time",
    "solution_final_values",
    "generation_error",
    "solve_error",
]

@dataclass
class VariantResult:
    name: str
    closures: bool
    cap: Optional[int]
    generation_time: float
    solve_time: Optional[float]
    equation_count: int
    solve_success: bool
    mean_abs_error_full: Optional[float] = None
    max_abs_error_full: Optional[float] = None
    mean_abs_error_closed: Optional[float] = None
    max_abs_error_closed: Optional[float] = None
    generation_error: Optional[str] = None
    solve_error: Optional[str] = None
    solution: Any = None
    lhs: Sequence[sym.Expr] = ()


def build_graph(graph_type: str, num_vertices: int, seed: int, p: float, m: int) -> nx.Graph:
    gt = graph_type.lower()
    if gt == "path":
        return nx.path_graph(num_vertices)
    if gt == "cycle":
        return nx.cycle_graph(num_vertices)
    if gt in {"erdos_renyi", "gnp"}:
        return nx.erdos_renyi_graph(num_vertices, p, seed=seed)
    if gt == "barabasi_albert":
        return nx.barabasi_albert_graph(num_vertices, m, seed=seed)
    raise ValueError(f"Unsupported graph type {graph_type}")


def variant_plan(n: int, fractions: Sequence[float]) -> List[Tuple[str, bool, Optional[int]]]:
    plan: List[Tuple[str, bool, Optional[int]]] = [
        ("full", False, None),
        ("closed", True, None),
    ]
    for frac in fractions:
        cap = max(1, round(n * frac))
        plan.append((f"cap_{int(round(frac*100))}", False, cap))
    return plan


def make_lhs(equations: EquationDict) -> List[sym.Expr]:
    lhs: List[sym.Expr] = []
    for eqs in equations.values():
        for eq in eqs:
            lhs.append(cast(sym.Expr, sym.Integral(eq.lhs).doit()))
    return lhs


def initial_cond(graph: nx.Graph, lhs: Sequence[sym.Expr], num_initial: int) -> Dict[sym.Expr, float]:
    return initial_conditions(
        list(graph.nodes),
        lhs,
        num_initial_infected=num_initial,
        symbol=sym.symbols("t"),
    )


def generate_and_solve(graph: nx.Graph, model, name: str, closures: bool, cap: Optional[int], num_initial: int, t_max: float) -> VariantResult:
    gen_start = time.monotonic()
    try:
        equations = generate_equations(graph, model, closures=closures, term_cap=cap)
        gen_time = time.monotonic() - gen_start
    except Exception as exc:  # noqa: BLE001
        return VariantResult(name=name, closures=closures, cap=cap, generation_time=0.0, solve_time=None, equation_count=0, solve_success=False, generation_error=str(exc))

    lhs = make_lhs(equations)
    solve_time: Optional[float] = None
    solution = None
    solve_success = False
    solve_error: Optional[str] = None
    if lhs:
        try:
            ic = initial_cond(graph, lhs, num_initial)
            solve_start = time.monotonic()
            solution = solve_equations(equations, ic, graph, t_max)
            solve_time = time.monotonic() - solve_start
            solve_success = bool(getattr(solution, "success", False))
        except Exception as exc:  # noqa: BLE001
            solve_error = str(exc)
            solve_time = None
            solve_success = False
    count = sum(len(v) for v in equations.values())
    return VariantResult(name=name, closures=closures, cap=cap, generation_time=gen_time, solve_time=solve_time, equation_count=count, solve_success=solve_success, solve_error=solve_error, solution=solution, lhs=lhs)


def interpolation_grid(a: Any, b: Any, t_max: float, samples: int) -> Optional[np.ndarray]:
    if not all(getattr(sol, "success", False) for sol in (a, b)):
        return None
    end = min(getattr(a, "t", [t_max])[-1], getattr(b, "t", [t_max])[-1], t_max)
    if end <= 0:
        return None
    return np.linspace(0, end, samples)


def interpolate(solution: Any, lhs: Sequence[sym.Expr], grid: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(solution.y)
    times = np.asarray(solution.t)
    mapping = {str(fn): i for i, fn in enumerate(lhs)}
    out: Dict[str, np.ndarray] = {}
    for name, idx in mapping.items():
        out[name] = np.interp(grid, times, arr[idx])
    return out


def attach_errors(reference: VariantResult, candidate: VariantResult, field_prefix: str, t_max: float, samples: int) -> None:
    if not (reference.solve_success and candidate.solve_success):
        return
    grid = interpolation_grid(reference.solution, candidate.solution, t_max, samples)
    if grid is None:
        return
    ref_vals = interpolate(reference.solution, reference.lhs, grid)
    cand_vals = interpolate(candidate.solution, candidate.lhs, grid)
    shared = set(ref_vals) & set(cand_vals)
    if not shared:
        return
    means: List[float] = []
    maxes: List[float] = []
    for key in shared:
        diff = np.abs(ref_vals[key] - cand_vals[key])
        means.append(float(np.mean(diff)))
        maxes.append(float(np.max(diff)))
    setattr(candidate, f"mean_abs_error_{field_prefix}", float(np.mean(means)))
    setattr(candidate, f"max_abs_error_{field_prefix}", float(np.max(maxes)))


def hostname_index(hostname: str) -> Optional[int]:
    # Expect names like fatanode01, fatanode02, etc.
    if "fatanode" in hostname:
        for part in hostname.split("fatanode"):
            if part and part.isdigit():
                return int(part) - 1  # zero-based
    return None


def partition(items: Sequence[Any], index: Optional[int], count: Optional[int]) -> List[Any]:
    if index is None or count is None or count <= 0:
        return list(items)
    return [item for i, item in enumerate(items) if i % count == index]


def write_results(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in RESULT_FIELDS})


def serialize_solution(sol: Any) -> Tuple[Optional[float], Optional[str]]:
    if sol is None or not getattr(sol, "success", False):
        return None, None
    final_t = float(getattr(sol, "t", [None])[-1])
    try:
        final_vec = [float(v) for v in np.asarray(sol.y)[:, -1]]
    except Exception:  # noqa: BLE001
        final_vec = []
    return final_t, json.dumps(final_vec)


def run_job(graph_type: str, seed: int, num_vertices: int, p: float, m: int, cap_fracs: Sequence[float], num_initial: int, t_max: float, samples: int, beta: float, gamma: float) -> List[VariantResult]:
    graph = build_graph(graph_type, num_vertices, seed, p=p, m=m)
    model = get_SIR(beta, gamma)
    results: List[VariantResult] = []
    for name, closures, cap in variant_plan(graph.number_of_nodes(), cap_fracs):
        vr = generate_and_solve(graph, model, name, closures, cap, num_initial, t_max)
        results.append(vr)
    full = next((r for r in results if r.name == "full"), None)
    closed = next((r for r in results if r.name == "closed"), None)
    for r in results:
        if r is full or not r.solve_success:
            continue
        if full and full.solve_success:
            attach_errors(full, r, "full", t_max, samples)
        if closed and closed.solve_success:
            attach_errors(closed, r, "closed", t_max, samples)
    return results


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel capped system experiments for fatanodes")
    p.add_argument("--graph-types", nargs="+", default=["path", "cycle", "erdos_renyi", "barabasi_albert"], help="Graph types")
    p.add_argument("--num-vertices", type=int, default=30)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--erdos-renyi-p", type=float, default=0.1, dest="er_p")
    p.add_argument("--barabasi-m", type=int, default=2, dest="ba_m")
    p.add_argument("--cap-fractions", nargs="+", type=float, default=list(CAP_FRACTIONS_DEFAULT))
    p.add_argument("--num-initial-infected", type=int, default=1)
    p.add_argument("--t-max", type=float, default=6.0)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--beta", type=float, default=0.8)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--output", type=str, default="data/fatanode_results.csv")
    p.add_argument("--parallel", type=int, default=os.cpu_count() or 4)
    p.add_argument("--node-index", type=int, help="Explicit node index (0-based)")
    p.add_argument("--node-count", type=int, help="Total nodes participating")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = args_parser()
    args = parser.parse_args(argv)

    hostname = socket.gethostname()
    auto_index = hostname_index(hostname)
    node_index = args.node_index if args.node_index is not None else auto_index
    node_count = args.node_count

    jobs: List[Tuple[str, int]] = []
    for g in args.graph_types:
        for s in args.seeds:
            jobs.append((g, s))
    jobs = partition(jobs, node_index, node_count)

    if not jobs:
        print("No jobs for this node; exiting.")
        return 0

    all_rows: List[Dict[str, Any]] = []
    for graph_type, seed in jobs:
        results = run_job(
            graph_type=graph_type,
            seed=seed,
            num_vertices=args.num_vertices,
            p=args.er_p,
            m=args.ba_m,
            cap_fracs=args.cap_fractions,
            num_initial=args.num_initial_infected,
            t_max=args.t_max,
            samples=args.samples,
            beta=args.beta,
            gamma=args.gamma,
        )
        for r in results:
            final_t, final_vals = serialize_solution(r.solution)
            row = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "hostname": hostname,
                "graph_type": graph_type,
                "seed": seed,
                "num_vertices": args.num_vertices,
                "graph_params": json.dumps({"p": args.er_p, "m": args.ba_m}, sort_keys=True),
                "variant": r.name,
                "closures": r.closures,
                "cap": r.cap,
                "generation_time": r.generation_time,
                "solve_time": r.solve_time,
                "equation_count": r.equation_count,
                "solve_success": r.solve_success,
                "mean_abs_error_full": r.mean_abs_error_full,
                "max_abs_error_full": r.max_abs_error_full,
                "mean_abs_error_closed": r.mean_abs_error_closed,
                "max_abs_error_closed": r.max_abs_error_closed,
                "solution_final_time": final_t,
                "solution_final_values": final_vals,
                "generation_error": r.generation_error,
                "solve_error": r.solve_error,
            }
            all_rows.append(row)
        print(f"Completed {graph_type} seed={seed} ({len(results)} variants)")

    write_results(Path(args.output), all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
