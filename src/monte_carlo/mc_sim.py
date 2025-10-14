from time import time
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sym
from matplotlib import pyplot as plt

from equation.generation import generate_equations
from equation.solving import initial_conditions, Vertex, solve
from model_params.cmodel import CModel, get_SIR
from plotting import plot_mc_averages

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
    beta = model.couplings['beta'][1]  # Rate of infection
    gamma = model.couplings['gamma'][1]  # Rate of recovery

    state = dict(init_state)  # The initial state of the system
    for _ in range(t_max):
        next_timestep = dict(state)  # Where we will store the states at next timestep
        for v in graph.nodes:
            if state[v] == 'I':  # This vertex was previously infected
                # Get list of vertices that could now become infected
                for n in nx.neighbors(graph, v):
                    if state[n] == 'S':
                        if np.random.random(1) < beta:  # Does the susceptible neighbour get infected?
                            next_timestep[n] = 'I'
                if np.random.random(1) < gamma:  # Does this thing now recover?
                    next_timestep[v] = 'R'
        state = next_timestep
        # print(state)
    to_return = []
    for s in state.keys():
        to_return.append(1) if state[s] == 'I' else to_return.append(0)
    return to_return


def example_monte_carlo(to_avg=False):
    # An example usage of the Monte Carlo simulation function
    tree = nx.path_graph(10)
    model = get_SIR(0.8, 0.1)  # type: ignore[arg-type]
    t_max = 5
    print('Beginning with', tree)
    print('Model:', model.couplings)
    init_state = set_initial_state(model, tree, choice=9)
    print('Initial state:', init_state)
    tol = 0.05
    if not to_avg:
        print('Single MC simulation:')
        print('result:\n', monte_carlo_sim(tree, model, init_state, t_max))
    else:
        print('Running to average:')
        results_1 = run_to_average(tree, model, init_state, t_max, tolerance=1e-3, num_rounds=100)
        last_1 = [results_1[i].mean() for i in results_1.columns]
        print(f'final average for t=:{t_max}', last_1)
        print(results_1.tail(1))
        results_1.plot()
        plt.show()

        to_avg = run_to_average(tree, model, init_state, 5, solution=last_1, tolerance=tol, num_rounds=10, timeout=180)
        last_2 = [to_avg[i].mean() for i in to_avg.columns]
        print('Solved to average:', last_2)
        to_avg.plot()
        plt.show()


def set_initial_state(model, tree, choice=None):
    initial_state = dict()
    for node in tree.nodes:
        initial_state[node] = 'S'
    if choice is None:
        initial_state[np.random.choice(tree.nodes)] = 'I'
    else:
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
    head = [i for i in range(len(init_state))]
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


def try_run_to_avg():
    t = sym.symbols('t')
    graph = nx.cycle_graph(3)
    beta, gamma = 0.5, 0.0
    model = get_SIR(beta, gamma)  # type: ignore[arg-type]
    tol = 1e-1
    timeout = 30
    t_max = 5
    num_rounds = 100

    soln = numerical_solve(model, beta, graph, t, t_max)
    # soln = [0.59, 0.66, 0.73, 0.77, 0.67, 0.33, 0, 0, 0, 0, 0]

    print(f'solution:\n{soln}')

    target_for_mc = {}
    for i in range(graph.number_of_nodes()):
        fun = sym.Function(str(Vertex('I', i)))(t)
        target_for_mc[fun] = soln[fun]

    choice = 0
    print('\ninitial infected:', choice)
    initial_for_mc = set_initial_state(model, graph, choice=choice)
    mc_defined = solve_with_mc(model, target_for_mc, graph, initial_for_mc, num_rounds, t_max, timeout, tol)
    plot_mc_averages(mc_defined, soln)
    print(mc_defined.values[-1].tolist())

    mc_avg = solve_with_mc(model, None, graph, initial_for_mc, t_max=t_max, tol=tol, timeout=timeout, num_rounds=num_rounds)
    plot_mc_averages(mc_avg, soln)
    print(mc_avg.values[-1].tolist())


def solve_with_mc(model, target_average, graph, init_for_MC, num_rounds, t_max, timeout, tol):
    if target_average is not None:
        print(f'\nsolving using MC to defined average using:\n  '
              f'- initial conditions: {init_for_MC}\n  - defined average: {target_average}')
    else:
        print(f'\nsolving using MC to consistent internal average using:\n  '
              f'- initial conditions: {init_for_MC}')
    start = time()
    try:
        mc_defined = run_to_average(graph, model, init_for_MC, t_max=t_max, solution=target_average,
                                    tolerance=tol, timeout=timeout,
                                    num_rounds=num_rounds)
        timed_out = False
    except MonteCarloTimeoutError as exc:
        mc_defined = exc.averages
        timed_out = True
    sol = time() - start
    print(f'time to solve: {"TO" if timed_out else sol}')
    print(f'solution:\n{mc_defined.tail(1)}')
    return mc_defined


def numerical_solve(model, beta, graph, t, t_max):
    start = time()
    equations = generate_equations(graph, model, closures=True)
    generate = time() - start
    for e in set().union(*equations.values()):
        print(e)
    print(f'\ntime to get {len(set().union(*equations.values()))} equations: {generate}')
    LHS = [sym.Integral(each.lhs).doit() for each in set().union(*equations.values())]
    init_cond = initial_conditions(list(graph.nodes), list(LHS), choice=[9], symbol=t)
    init_cond_for_analytic = dict()
    for i in init_cond:
        init_cond_for_analytic[i.subs(t, 0)] = round(init_cond[i], 3)
    print(f'\nInitial conditions:\n{init_cond_for_analytic}\n')
    numerical_solution = solve(equations, graph, init_cond=init_cond, t_max=t_max, step=0.01, rtol=1, atol=1,
                               print_option='full')
    print(f'\n{numerical_solution["message"]}')
    analytic = False
    soln = dict(zip(init_cond.keys(), numerical_solution['y'][-1].tolist()))
    print(f'solution:\n{soln}')
    all_equations = []
    for e in equations:
        all_equations.extend(equations[e])
    return soln


def find_infected_from_init_cond(graph, init_cond, t):
    choice = -1
    for v in graph.nodes:
        if init_cond[sym.Function(str(Vertex('S', v)))(t)] < init_cond[sym.Function(str(Vertex('I', v)))(t)]:
            choice = v
            break
    if choice < 0:
        print('there\'s a problem!')
        print(init_cond)
    return choice
