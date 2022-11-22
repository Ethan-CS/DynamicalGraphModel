import sys
from datetime import datetime
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sym
from matplotlib import pyplot as plt

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
    results = pd.DataFrame(columns=[i for i in range(len(init_state))], dtype=object)
    averages = pd.DataFrame(columns=[i for i in range(len(init_state))], dtype=object)
    start = time()
    counter = 1
    while True:
        results = pd.concat([results, pd.DataFrame(data=[monte_carlo_sim(graph, model, init_state, t_max)])])
        within = True  # set to False if averages not inside acceptable range
        prev_mean = [-1 for _ in range(len(results.columns))] if averages.empty else averages.iloc[-1].tolist()
        mean = [results[i].mean() for i in range(len(results.columns))]
        averages = pd.concat([averages, pd.DataFrame(data=[mean])]).reset_index(drop=True)
        for i in range(len(mean)):
            if solution is None:
                solution_range = [(s - tolerance / 2, s + tolerance / 2) for s in prev_mean]
            if not solution_range[i][0] < mean[i] < solution_range[i][1]:
                within = False
                counter = 1  # Reset the counter
                break
        if within:
            if counter == num_rounds:
                # print(f'succeeded in {time() - start}s')
                # print('final avg:', [results[i].mean() for i in results.columns])
                return averages
            else:
                counter += 1

        if time() - start >= timeout:
            print('timeout!')
            # print('final avg:', [results[i].mean() for i in results.columns])
            return averages


def try_run_to_avg():
    t = sym.symbols('t')
    graph = nx.path_graph(5)
    beta, gamma = 0.7, 0.3
    SIR = get_SIR(beta, gamma)
    tol = 1e-3
    timeout = 180
    t_max = 5
    num_rounds = 100
    print('finished set-up, generating equations...')

    start = time()
    equations = generate_equations(graph, SIR, closures=True)
    generate = time()-start
    print(f'\ntime to get {len(set().union(*equations.values()))} equations: {generate}')

    LHS = [sym.Integral(each.lhs).doit() for each in set().union(*equations.values())]
    init_cond = initial_conditions(list(graph.nodes), list(LHS), beta=beta, symbol=t)

    init_cond_for_analytic = dict()
    for i in init_cond:
        init_cond_for_analytic[i.subs(t, 0)] = round(init_cond[i], 3)
    print(f'\nInitial conditions:\n{init_cond_for_analytic}\n')

    numerical_solution = solve(equations, graph, init_cond=init_cond, t_max=t_max, step=0.1,
                               print_option='full')
    print(f'\n{numerical_solution["message"]}')
    analytic = False
    try:
        analytical_solution = sym.solvers.ode.systems.dsolve_system(eqs=set().union(*equations.values()), funcs=LHS,
                                                                    t=t, ics=init_cond_for_analytic)
        print(f'\nAnalytical solution:\n{analytical_solution}')
        soln = dict(zip(init_cond.keys(), analytical_solution[-1]))
        analytic = True
    except NotImplementedError:
        soln = dict(zip(init_cond.keys(), numerical_solution['y'][-1].tolist()))
        print('\nCould not solve system analytically.')

    all_equations = []
    for e in equations:
        all_equations.extend(equations[e])

    choice = find_infected_from_init_cond(graph, init_cond, t)
    print('\ninitial infected:', choice)

    conditions_for_mc = {}
    for i in range(graph.number_of_nodes()):
        fun = sym.Function(str(Vertex('I', i)))(t)
        if analytic:
            val = round(soln[fun].rhs.subs(t, t_max), 2)
        else:
            val = round(soln[fun], 2)
        conditions_for_mc[fun] = 0 if val < 0 else min(val, 1)

    print(f'\nsolving using MC to defined average using:\n{conditions_for_mc}')
    start = time()
    mc_defined = run_to_average(graph, SIR, set_initial_state(SIR, graph, choice=choice), 10,
                                list(conditions_for_mc.values()), tolerance=tol, timeout=timeout,
                                num_rounds=num_rounds)
    sol = time() - start
    print(f'time to solve to defined average: {sol if sol<timeout else "TO"}')
    print(f'solution:\n{mc_defined.tail(1)}')
    mc_defined.plot()
    plt.show()

    # print('\nsolving using MC to consistent average...')
    # start = time()
    # mc_avg = run_to_average(graph, SIR, init_state=set_initial_state(SIR, graph, choice=choice), t_max=t_max,
    #                         tolerance=tol, timeout=timeout, num_rounds=num_rounds)
    # sol = time() - start
    # print(f'time to solve to consistent average: {sol if sol<timeout else "TO"}')
    # print(f'solution:\n{mc_avg.tail(1)}')


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
