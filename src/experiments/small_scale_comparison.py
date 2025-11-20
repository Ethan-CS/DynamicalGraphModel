from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import networkx as nx
import numpy as np
import sympy as sym

from equation.generation import generate_equations
from equation.solving import initial_conditions, solve_equations
from model_params.cmodel import CModel, get_SIR

EquationDict = Dict[int, Sequence[sym.Eq]]


@dataclass
class VariantMetrics:
    name: str
    closures: bool
    term_cap: Optional[int]
    generation_time: float
    solve_time: Optional[float]
    equation_count: int
    solve_success: bool
    mean_abs_error: Optional[float] = None
    max_abs_error: Optional[float] = None
    generation_error: Optional[str] = None
    solve_error: Optional[str] = None


@dataclass
class VariantResult:
    metrics: VariantMetrics
    equations: EquationDict
    lhs_functions: Sequence[sym.Expr]
    solution: Optional[Any]


def flatten_equations(equations: EquationDict) -> List[sym.Eq]:
    return [eq for eqs in equations.values() for eq in eqs]


def build_graph(graph_type: str, num_vertices: int, seed: Optional[int]) -> nx.Graph:
    graph_type = graph_type.lower()
    if graph_type == "path":
        return nx.path_graph(num_vertices)
    if graph_type == "cycle":
        return nx.cycle_graph(num_vertices)
    if graph_type == "tree":
        tree_fn = getattr(nx, "random_tree")
        return tree_fn(num_vertices, seed=seed)
    raise ValueError(f"Unsupported graph type '{graph_type}'")


def make_initial_conditions(
    graph: nx.Graph,
    lhs_functions: Sequence[sym.Expr],
    initial_choice: Sequence[int],
    num_initial_infected: int,
) -> Dict[sym.Expr, float]:
    return initial_conditions(
        list(graph.nodes),
        lhs_functions,
        choice=list(initial_choice),
        num_initial_infected=num_initial_infected,
        symbol=0,
        yes=0.95,  # type: ignore[arg-type]
        no=0.05,  # type: ignore[arg-type]
    )


def generate_and_solve(
    graph: nx.Graph,
    model: CModel,
    name: str,
    closures: bool,
    term_cap: Optional[int],
    initial_choice: Sequence[int],
    num_initial_infected: int,
    t_max: float,

) -> VariantResult:
    gen_start = time.monotonic()
    try:
        equations = generate_equations(graph, model, closures=closures, term_cap=term_cap)
        generation_time = time.monotonic() - gen_start
    except Exception as exc:  # pylint: disable=broad-except
        metrics = VariantMetrics(
            name=name,
            closures=closures,
            term_cap=term_cap,
            generation_time=0.0,
            solve_time=None,
            equation_count=0,
            solve_success=False,
            generation_error=str(exc),
        )
        return VariantResult(metrics=metrics, equations={}, lhs_functions=[], solution=None)

    eq_list = flatten_equations(equations)
    lhs_functions: List[sym.Expr] = []
    for eq in eq_list:
        lhs_functions.append(cast(sym.Expr, sym.Integral(eq.lhs).doit()))

    solve_time: Optional[float] = None
    solution = None
    solve_success = False
    solve_error: Optional[str] = None

    if lhs_functions:
        try:
            init_cond = make_initial_conditions(
                graph,
                lhs_functions,
                initial_choice=initial_choice,
                num_initial_infected=num_initial_infected,
            )
            solve_start = time.monotonic()
            solution = solve_equations(equations, init_cond, graph, t_max)
            solve_time = time.monotonic() - solve_start
            solve_success = getattr(solution, "success", False)
        except Exception as exc:  # pylint: disable=broad-except
            solve_error = str(exc)
            solution = None
            solve_time = None
            solve_success = False

    metrics = VariantMetrics(
        name=name,
        closures=closures,
        term_cap=term_cap,
        generation_time=generation_time,
        solve_time=solve_time,
        equation_count=len(eq_list),
        solve_success=solve_success,
        solve_error=solve_error,
    )
    cast_lhs: Sequence[sym.Expr] = cast(Sequence[sym.Expr], lhs_functions)
    return VariantResult(metrics=metrics, equations=equations, lhs_functions=cast_lhs, solution=solution)


def interpolation_grid(reference: Any, candidate: Any, t_max: float, samples: int) -> Optional[np.ndarray]:
    if reference is None or candidate is None:
        return None
    if not getattr(reference, "success", False) or not getattr(candidate, "success", False):
        return None
    ref_end = getattr(reference, "t", [t_max])[-1]
    cand_end = getattr(candidate, "t", [t_max])[-1]
    horizon = min(ref_end, cand_end, t_max)
    if horizon <= 0:
        return None
    return np.linspace(0, horizon, samples)


def interpolate_solution(solution: Any, lhs_functions: Sequence[sym.Expr], time_grid: np.ndarray) -> Dict[str, np.ndarray]:
    values = np.asarray(solution.y)
    times = np.asarray(solution.t)
    mapping = {str(fn): idx for idx, fn in enumerate(lhs_functions)}
    interpolated: Dict[str, np.ndarray] = {}
    for name, idx in mapping.items():
        interpolated[name] = np.interp(time_grid, times, values[idx])
    return interpolated


def attach_error_metrics(reference: VariantResult, candidate: VariantResult, t_max: float, samples: int) -> None:
    if reference.solution is None or candidate.solution is None:
        return
    grid = interpolation_grid(reference.solution, candidate.solution, t_max, samples)
    if grid is None:
        return
    ref_values = interpolate_solution(reference.solution, reference.lhs_functions, grid)
    cand_values = interpolate_solution(candidate.solution, candidate.lhs_functions, grid)
    shared_keys = set(ref_values).intersection(cand_values)
    if not shared_keys:
        return
    diffs: List[float] = []
    for key in shared_keys:
        diff = np.abs(ref_values[key] - cand_values[key])
        diffs.append(float(np.mean(diff)))
        diffs.append(float(np.max(diff)))
    candidate.metrics.mean_abs_error = float(np.mean(diffs))
    candidate.metrics.max_abs_error = float(np.max(diffs))


def variant_plan(num_vertices: int, cap_fractions: Sequence[float]) -> List[Tuple[str, bool, Optional[int]]]:
    plan: List[Tuple[str, bool, Optional[int]]] = [
        ("full", False, None),
        ("closed", True, None),
    ]
    for frac in cap_fractions:
        cap = max(1, round(num_vertices * frac))
        label = f"cap_{int(round(frac * 100))}"
        plan.append((label, False, cap))
    return plan


def run_for_graph(
    graph_type: str,
    graph: nx.Graph,
    model: CModel,
    cap_fractions: Sequence[float],
    num_initial_infected: int,
    t_max: float,
    samples: int,
) -> List[VariantResult]:
    initial_choice = list(range(min(num_initial_infected, graph.number_of_nodes())))
    results: List[VariantResult] = []
    for name, closures, cap in variant_plan(graph.number_of_nodes(), cap_fractions):
        result = generate_and_solve(
            graph=graph,
            model=model,
            name=name,
            closures=closures,
            term_cap=cap,
            initial_choice=initial_choice,
            num_initial_infected=num_initial_infected,
            t_max=t_max,
        )
        results.append(result)
    reference = next((r for r in results if r.metrics.name == "full"), None)
    if reference and reference.metrics.solve_success:
        for candidate in results:
            if candidate is reference:
                continue
            if candidate.metrics.solve_success:
                attach_error_metrics(reference, candidate, t_max=t_max, samples=samples)
    return results


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value >= 1:
        return f"{value:5.2f}s"
    return f"{value * 1000:5.1f}ms"


def print_results(graph_type: str, graph: nx.Graph, results: Iterable[VariantResult]) -> None:
    header = f"Results for {graph_type} graph (n={graph.number_of_nodes()}, m={graph.number_of_edges()})"
    print("\n" + header)
    print("-" * len(header))
    print(
        f"{'Variant':<12}{'Cap':>6}{'Clos?':>7}{'#Eqns':>8}{'Gen':>10}{'Solve':>10}{'OK?':>6}{'Mean |Δ|':>12}{'Max |Δ|':>12}"
    )
    notes: List[str] = []
    for result in results:
        m = result.metrics
        cap_display = m.term_cap if m.term_cap is not None else "-"
        mean_err = f"{m.mean_abs_error:.3e}" if m.mean_abs_error is not None else "-"
        max_err = f"{m.max_abs_error:.3e}" if m.max_abs_error is not None else "-"
        print(
            f"{m.name:<12}{cap_display:>6}{str(m.closures):>7}{m.equation_count:>8}"
            f"{format_seconds(m.generation_time):>10}{format_seconds(m.solve_time):>10}"
            f"{str(m.solve_success):>6}{mean_err:>12}{max_err:>12}"
        )
        if m.generation_error:
            notes.append(f"{m.name}: generation failed ({m.generation_error})")
        if m.solve_error:
            notes.append(f"{m.name}: solve failed ({m.solve_error})")
    if notes:
        print("  Notes:")
        for note in notes:
            print(f"    - {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full, closed, and capped equation systems on small graphs")
    parser.add_argument("--graph-types", nargs="+", default=["path", "cycle", "tree"], help="Graph families to run")
    parser.add_argument("--num-vertices", type=int, default=10, help="Number of vertices in each graph")
    parser.add_argument("--beta", type=float, default=1.0, help="Infection rate for SIR model")
    parser.add_argument("--gamma", type=float, default=0.1, help="Recovery rate for SIR model")
    parser.add_argument("--num-initial-infected", type=int, default=1, help="How many vertices begin infected")
    parser.add_argument("--t-max", type=float, default=5.0, help="Time horizon for solving")
    parser.add_argument(
        "--cap-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="Fractions of |V| to use as term caps",
    )
    parser.add_argument("--samples", type=int, default=50, help="Number of sample points for error integration")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random graph generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = get_SIR(args.beta, args.gamma)
    for graph_type in args.graph_types:
        graph = build_graph(graph_type, args.num_vertices, seed=args.seed)
        results = run_for_graph(
            graph_type,
            graph,
            model,
            cap_fractions=args.cap_fractions,
            num_initial_infected=max(1, min(args.num_initial_infected, graph.number_of_nodes())),
            t_max=args.t_max,
            samples=args.samples,
        )
        print_results(graph_type, graph, results)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
