"""Utility script to compare equation truncation versus closure-based solving.

The demo performs three sequential tasks:

1. Generate and print the full and closed systems for a lollipop graph.
2. Produce truncated systems to illustrate how many equations remain at each term-length cap.
3. Solve both the closed system and a couple of truncated systems, reporting timing information and approximation errors in the aggregate infected population.
"""

import copy
import time

import networkx as nx
import sympy as sym
import numpy as np

from equation.generation import generate_equations
from equation.solving import solve_equations, initial_conditions
from model_params.cmodel import get_SIR


def build_lollipop_graph():
    """Return the fixed 4-node lollipop graph
    """
    g = nx.Graph()
    g.add_nodes_from(range(4))
    edges = [(0, 1), (0, 2), (1, 2), (1, 3)]
    g.add_edges_from(edges)
    return g


def generate_systems(graph, model):
    """Generate and display the full and closed equation sets for the graph/model pair."""
    print("=== Generating full and closed systems for lollipop graph ===")

    start = time.perf_counter()
    full_eqs = generate_equations(copy.deepcopy(graph), model, closures=False, term_cap=None)
    full_time = time.perf_counter() - start
    full_terms = sum(len(group) for group in full_eqs.values())
    print(f"Full system: {full_terms} equations (generation {full_time:.4f}s)")

    start = time.perf_counter()
    closed_eqs = generate_equations(copy.deepcopy(graph), model, closures=True, term_cap=None)
    closed_time = time.perf_counter() - start
    closed_terms = sum(len(group) for group in closed_eqs.values())
    print(f"Closed system: {closed_terms} equations (generation {closed_time:.4f}s)")

    print("\n--- Full system equations ---")
    for k in sorted(full_eqs.keys()):
        for eq in full_eqs[k]:
            print(eq)

    print("\n--- Closed system equations ---")
    for k in sorted(closed_eqs.keys()):
        for eq in closed_eqs[k]:
            print(eq)

    return full_eqs, closed_eqs


def generate_truncated_systems(graph, model, max_length):
    """Generate and report truncated systems for every cap up to ``max_length``."""
    print("\n=== Generating truncated systems ===")
    systems = {}
    for cap in range(1, max_length + 1):
        start = time.perf_counter()
        truncated = generate_equations(copy.deepcopy(graph), model, closures=False, term_cap=cap)
        elapsed = time.perf_counter() - start
        num_eqs = sum(len(group) for group in truncated.values())
        print(f"Term length cap {cap}: {num_eqs} equations (generation {elapsed:.4f}s)")
        systems[cap] = truncated
    return systems


def collect_lhs_functions(equations):
    """Return a flat list of function symbols that appear on the LHS of the equation set."""
    functions = []
    for eq_list in equations.values():
        for eq in eq_list:
            functions.append(sym.Integral(eq.lhs).doit())
    return functions


def _total_infected(solution, label):
    """Sum every third row starting from index 1, assuming an S/I/R ordering, to approximate total infections."""
    y = np.asarray(getattr(solution, "y", []))
    if y.ndim != 2:
        print(f"Warning: unexpected shape for {label} solution; skipping error computation.")
        return None
    n_rows = y.shape[0]
    infected_rows = list(range(1, n_rows, 3))
    if not infected_rows:
        print(f"Warning: no infected-like rows inferred for {label} solution; skipping.")
        return None
    return y[infected_rows, :].sum(axis=0)


def _compare_to_closed(label, sol_trunc, sol_closed):
    """Compare total infected trajectories between a truncated solution and the closed baseline."""
    t_trunc = np.asarray(getattr(sol_trunc, "t", []))
    t_closed = np.asarray(getattr(sol_closed, "t", []))
    if t_trunc.size == 0 or t_closed.size == 0:
        print("\nCould not compute error: one of the solutions is empty.")
        return

    I_trunc = _total_infected(sol_trunc, f"truncated-{label}")
    I_closed = _total_infected(sol_closed, "closed")
    if I_trunc is None or I_closed is None:
        return

    I_trunc_interp = np.interp(t_closed, t_trunc, I_trunc)
    abs_diff = np.abs(I_trunc_interp - I_closed)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    print(f"\nApproximation error in total infected ({label} vs closed):")
    print(f"max |ΔI_total| = {max_diff:.4e}, mean |ΔI_total| = {mean_diff:.4e}")


def time_solves(graph, model, closed_eqs, t_max=5):
    """Solve the closed system and selected truncations, reporting timings and accuracy."""
    print("\n=== Solving selected systems (caps = 2 and 3) ===")

    caps = (2, 3)
    lhs_functions = collect_lhs_functions(closed_eqs)
    truncated_systems = {}
    trunc_gen_times = {}
    trunc_eq_counts = {}

    for cap in caps:
        start = time.perf_counter()
        truncated = generate_equations(copy.deepcopy(graph), model, closures=False, term_cap=cap)
        trunc_gen_times[cap] = time.perf_counter() - start
        truncated_systems[cap] = truncated
        trunc_eq_counts[cap] = sum(len(group) for group in truncated.values())
        lhs_functions.extend(collect_lhs_functions(truncated))

    deduped_lhs = list(dict.fromkeys(lhs_functions))
    init_cond = initial_conditions(list(graph.nodes), deduped_lhs, num_initial_infected=1, symbol=0)

    start = time.perf_counter()
    sol_closed = solve_equations(closed_eqs, init_cond, graph, t_max)
    solve_closed_time = time.perf_counter() - start
    closed_num_eqs = sum(len(group) for group in closed_eqs.values())

    for cap in caps:
        start = time.perf_counter()
        sol_trunc = solve_equations(truncated_systems[cap], init_cond, graph, t_max)
        solve_trunc_time = time.perf_counter() - start

        print("\nSolution summaries:")
        print(f"Truncated (cap={cap}): generation {trunc_gen_times[cap]:.4f}s, solve {solve_trunc_time:.4f}s")
        print(f"  Truncated (cap={cap}): success={getattr(sol_trunc, 'success', None)}, dimension={trunc_eq_counts[cap]}, t_len={len(getattr(sol_trunc, 't', []))}")
        print(f"  Closed:              success={getattr(sol_closed, 'success', None)}, dimension={closed_num_eqs}, t_len={len(getattr(sol_closed, 't', []))}")

        _compare_to_closed(f"cap={cap}", sol_trunc, sol_closed)

    print(f"\nClosed system: solve {solve_closed_time:.4f}s")


def main():
    """Drive the full demo pipeline for the default SIR model and lollipop graph."""
    model = get_SIR()
    graph = build_lollipop_graph()

    full_eqs, closed_eqs = generate_systems(graph, model)

    max_length = max(full_eqs.keys()) if full_eqs else 1
    generate_truncated_systems(graph, model, max_length)

    time_solves(graph, model, closed_eqs, t_max=5)


if __name__ == "__main__":
    main()
