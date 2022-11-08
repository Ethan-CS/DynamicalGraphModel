from time import time

import networkx as nx
import numpy as np
import sympy as sym

from equation import generate_equations
from equation.testing_numerical_solvers import solve
from model_params.cmodel import CModel, get_SIR


def monte_carlo_sim(graph: nx.Graph, model: CModel, init_state: dict, t_max: int):
    state = dict(init_state)
    for _ in range(t_max):
        next_timestep = dict(state)
        for v in graph.nodes:
            if next_timestep[v] == 'I':
                # Get list of vertices that could now become infected
                neighbours = nx.neighbors(graph, v)
                for n in neighbours:
                    if state[n] == 'S':
                        if np.random.random(1) < model.couplings['beta'][1]:
                            next_timestep[n] = 'I'
                if np.random.random(1) < model.couplings['gamma'][1]:
                    next_timestep[v] = 'R'
        state = next_timestep
    return state


def example_monte_carlo():
    tree = nx.random_tree(10)
    SIR = get_SIR(0.8, 0.1)
    initial_state = set_initial_state(SIR, tree)
    print(initial_state)
    print('result:\n', monte_carlo_sim(tree, SIR, initial_state, 5))


def set_initial_state(model, tree):
    initial_state = dict()
    print(str(model.states))
    for node in tree.nodes:
        initial_state[node] = 'S'
    initial_state[np.random.choice(tree.nodes)] = 'I'
    return initial_state


def run_to_average(graph, model, init_state, t_max, solution, tolerance, timeout=60):
    solution_range = [range(s * (1 - tolerance), s * (1 + tolerance)) for s in solution]
    averages = []
    start = time()
    while True:
        averages.append(monte_carlo_sim(graph, model, init_state, t_max))
        avg = [sum(a) / len(a) for a in zip(*averages)]
        within = True
        for i in range(len(avg)):
            if avg[i] not in solution_range[i]:
                within = False
                break
        if within:
            print(f'succeeded in {time() - start}s')
            break
        if time() - start >= timeout:
            print('timeout!')


def try_run_to_avg():
    tree = nx.random_tree(10)
    SIR = get_SIR()
    initial_state = set_initial_state(SIR, tree)
    equations = generate_equations(tree, SIR)
    with open('equations.txt', 'w') as f:
        for e in equations:
            f.write(f'{sym.Integral(e.lhs).doit()}\' = {e.rhs}\n')
    solution = solve(equations, tree)
    print(solution)
    run_to_average(tree, SIR, initial_state, 10, solution, 0.05, 60)


# try_run_to_avg()
