import sys
from datetime import datetime
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sym

from equation import generate_equations, initial_conditions, Vertex
from equation.testing_numerical_solvers import solve, plot_soln
from model_params.cmodel import CModel, get_SIR


def monte_carlo_sim(graph: nx.Graph, model: CModel, init_state: dict, t_max: int):
    beta = model.couplings['beta'][1]
    gamma = model.couplings['gamma'][1]

    state = dict(init_state)
    for _ in range(t_max):
        next_timestep = dict(state)
        for v in graph.nodes:
            if next_timestep[v] == 'I':
                # Get list of vertices that could now become infected
                neighbours = nx.neighbors(graph, v)
                for n in neighbours:
                    if state[n] == 'S':
                        if np.random.random(1) < beta:
                            next_timestep[n] = 'I'
                if np.random.random(1) < gamma:
                    next_timestep[v] = 'R'
        state = next_timestep
    to_return = []
    for s in state.keys():
        to_return.append(1) if state[s] == 'I' else to_return.append(0)
    return to_return


def example_monte_carlo():
    tree = nx.random_tree(10)
    SIR = get_SIR(0.7, 0.1)
    initial_state = set_initial_state(SIR, tree)
    print(initial_state)
    print('result:\n', monte_carlo_sim(tree, SIR, initial_state, 5))


def set_initial_state(model, tree, choice=None):
    initial_state = dict()
    for node in tree.nodes:
        initial_state[node] = 'S'
    if choice is None:
        initial_state[np.random.choice(tree.nodes)] = 'I'
    else:
        initial_state[choice] = 'I'
    return initial_state


def run_to_average(graph, model, init_state, t_max, solution=None, tolerance=0.1, timeout=60, num_rounds=10):
    solution_range = []
    if solution is not None:
        solution = solution[:len(init_state)]
        solution_range = [(s-tolerance/2, s+tolerance/2) for s in solution]
        # TODO output results for (up to) all same probabilities that are solved in equations case?
    results = pd.DataFrame(columns=[i for i in range(len(init_state))])
    averages = pd.DataFrame(columns=[i for i in range(len(init_state))])
    start = time()
    counter = 1
    while True:
        results = pd.concat([results, pd.DataFrame(data=[monte_carlo_sim(graph, model, init_state, t_max)])])
        within = True  # set to False if averages not inside acceptable range
        prev_mean = [-1 for _ in range(len(results.columns))] if averages.empty else averages.values[-1].tolist()

        mean = [results[i].mean() for i in range(len(results.columns))]
        averages = pd.concat([averages, pd.DataFrame(data=[mean])])
        for i in range(len(mean)):
            if solution is None:
                solution_range = [(s - tolerance / 2, s + tolerance / 2) for s in prev_mean]
            if not solution_range[i][0] < mean[i] < solution_range[i][1]:
                within = False
                counter = 1  # Reset the counter
                break
        if within:
            if solution is not None or counter == num_rounds:
                # print(f'succeeded in {time() - start}s')
                # print('final avg:', [results[i].mean() for i in results.columns])
                return results
            else:
                counter += 1

        if time() - start >= timeout:
            print('timeout!')
            # print('final avg:', [results[i].mean() for i in results.columns])
            return results


def try_run_to_avg():
    print('setting up...')
    tree = nx.path_graph(10)
    beta, gamma = 0.7, 0.3
    SIR = get_SIR(beta, gamma)
    tolerance = 1e-1
    t_max = 5
    print('finished set-up, generating equations...')

    start = time()
    equations = generate_equations(tree, SIR, closures=True)
    generate = time()-start
    print(f'time to get {len(set().union(*equations.values()))} equations: {generate}')

    LHS = [sym.Integral(each.lhs).doit() for each in set().union(*equations.values())]
    init_cond = initial_conditions(list(tree.nodes), list(LHS), beta=beta, symbol=sym.symbols('t'))

    start = time()
    solution = solve(equations, tree, init_cond=init_cond.values(), t_max=t_max, atol=tolerance, rtol=tolerance,
                     step=0.1, print_option='full')
    sol = time() - start
    print(f'time to solve equations: {sol}')

    all_equations = []
    for e in equations:
        all_equations.extend(equations[e])

    print(solution['message'])
    sub_solution = list(solution['y'])[-1][:5]
    print('solution[:5] is', [(0 if i < 0 else round(min(i, 1), 5)) for i in sub_solution])

    choice = -1
    t = sym.symbols('t')
    for v in tree.nodes:
        if init_cond[sym.Function(str(Vertex('S', v)))(t)] < init_cond[sym.Function(str(Vertex('I', v)))(t)]:
            choice = v
            break
    if choice < 0:
        print('there\'s a problem!')
        print(init_cond)

    # run_to_average(tree, SIR, set_initial_state(SIR, tree, choice=choice), 10, solution['y'], 0.25, 60)

    print('solving using MC to consistent average...')
    start = time()
    soln_mc_avg = run_to_average(tree, SIR, init_state=set_initial_state(SIR, tree, choice=choice), t_max=t_max,
                                 tolerance=tolerance, timeout=60, num_rounds=100)
    sol = time() - start
    print(f'time to solve to consistent average: {sol}')
    # print(soln_mc_avg)
