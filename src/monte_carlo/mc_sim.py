from time import time
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from model_params.cmodel import CModel

class MonteCarloTimeoutError(TimeoutError):
    def __init__(self, seconds: float, averages: pd.DataFrame):
        super().__init__(f"Monte Carlo simulation exceeded {seconds:.2f} seconds")
        self.averages = averages


def _build_average_frame(rows: List[List[float]], columns: List[int]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns, dtype=float)
    return pd.DataFrame(rows, columns=columns, dtype=float)


def monte_carlo_sim(graph: nx.Graph, model: CModel, init_state: dict, t_max: int):
    """
    Given a graph-based compartmental model, run a MCMC simulation using specified initial conditions to a given
    timestep.
    :param graph: underlying graph for the model.
    :param model: compartmental modelling framework.
    :param init_state: initial conditions (which vertices are initially infected, which are susceptible for SIR).
    :param t_max: maximum timestep to which we should run the simulation.
    :return:
    """
    beta = float(model.couplings['beta'][1])  # Rate of infection
    gamma = float(model.couplings['gamma'][1])  # Rate of recovery

    state = dict(init_state)
    adjacency = graph.adj
    for _ in range(t_max):
        next_timestep = state.copy()
        for vertex in graph.nodes:
            if state[vertex] != 'I':
                continue
            for neighbor in adjacency[vertex]:
                if state[neighbor] == 'S' and np.random.random() < beta:
                    next_timestep[neighbor] = 'I'
            if np.random.random() < gamma:
                next_timestep[vertex] = 'R'
        state = next_timestep
    return [1 if state[node] == 'I' else 0 for node in graph.nodes]


def set_initial_state(model, tree, choice=None):
    initial_state = {node: 'S' for node in tree.nodes}
    if choice is None:
        choice = np.random.choice(list(tree.nodes))
    initial_state[choice] = 'I'
    return initial_state


def run_to_average(
    graph,
    model,
    init_state,
    t_max,
    solution=None,
    tolerance: float = 0.1,
    timeout: Optional[float] = 60,
    num_rounds: int = 10,
):
    head = list(range(len(init_state)))
    target_bounds = None
    if solution is not None:
        values = list(solution.values()) if isinstance(solution, dict) else list(solution)
        target_bounds = [(float(s) - tolerance, float(s) + tolerance) for s in values]

    averages_rows: List[List[float]] = []
    cumulative = np.zeros(len(init_state), dtype=float)
    counter = 1
    prev_mean: Optional[List[float]] = None
    start = time()

    while True:
        trial_values = np.array(monte_carlo_sim(graph, model, init_state, t_max), dtype=float)
        cumulative += trial_values
        count = len(averages_rows) + 1
        mean = (cumulative / count).tolist()
        averages_rows.append(mean)

        if target_bounds is None and prev_mean is not None:
            bounds = [(m - tolerance, m + tolerance) for m in prev_mean]
        else:
            bounds = target_bounds

        within = False
        if bounds is not None:
            within = all(lower < val < upper for val, (lower, upper) in zip(mean, bounds))

        if within:
            if counter == num_rounds:
                return _build_average_frame(averages_rows, head)
            counter += 1
        else:
            counter = 1

        prev_mean = mean

        if timeout and timeout > 0 and time() - start >= timeout:
            raise MonteCarloTimeoutError(timeout, _build_average_frame(averages_rows, head))
