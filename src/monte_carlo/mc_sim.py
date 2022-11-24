import sys
from datetime import datetime
from time import time

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import sympy as sym

from matplotlib import pyplot as plt
from equation import generate_equations, initial_conditions, Vertex
from equation.testing_numerical_solvers import solve
from model_params.cmodel import CModel, get_SIR

matplotlib.use('module://backend_interagg')


def monte_carlo_sim(graph: nx.Graph, model: CModel, init_state: dict, t_max: int):
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
    tree = nx.path_graph(10)
    model = get_SIR(0.8, 0.1)
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
        print('final average:', last_1)
        print(results_1.tail(1))
        results_1.plot()
        plt.show()

        to_avg = run_to_average(tree, model, init_state, 5, solution=last_1, tolerance=tol, num_rounds=10, timeout=180)
        last_2 = [to_avg[i].mean() for i in to_avg.columns]
        print('Solved to average:', last_2)
        to_avg.plot()
        plt.show()
        diff = [abs(last_1[i] - last_2[i]) / ((last_1[i] + last_2[i]) / 2) for i in range(len(last_1))]
        print(diff)
        print([diff[i] < tol for i in range(len(diff))])


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
        solution_range = [(s - tolerance, s + tolerance) for s in list(solution.values())]
        print(f'Solution range:\n{solution_range}')
    results = pd.DataFrame(columns=[i for i in range(len(init_state))], dtype=object)
    averages = pd.DataFrame(columns=[i for i in range(len(init_state))], dtype=object)
    start = time()
    counter = 1
    while True:
        results = pd.concat([results, pd.DataFrame(data=[monte_carlo_sim(graph, model, init_state, t_max)])])
        if averages.empty:
            prev_mean = [-1 for _ in range(len(results.columns))]
        else:
            prev_mean = averages.iloc[-1].tolist()

        if solution is None:
            solution_range = [(s - tolerance, s + tolerance) for s in prev_mean]

        mean = [results[i].mean() for i in range(len(results.columns))]
        averages = pd.concat([averages, pd.DataFrame(data=[mean])]).reset_index(drop=True)

        within = True  # set to False if averages not inside acceptable range
        for i in range(len(mean)):
            if not solution_range[i][0] < mean[i] < solution_range[i][1]:
                within = False
                counter = 1  # Reset the counter
                break
            # else:
            #     print(f'{mean[i]} is within ({round(solution_range[i][0], 5)}, {round(solution_range[i][1], 5)})')

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
    graph = nx.path_graph(10)
    beta, gamma = 0.8, 0.1
    model = get_SIR(beta, gamma)
    tol = 1e-6
    timeout = 30
    t_max = 5
    num_rounds = 1000

    # soln = numerical_solve(model, beta, graph, t, t_max)
    soln = [((1-gamma)**t_max), beta*(1-gamma)**4, (beta**2)*(1-gamma)**3, (beta**3)*(1-gamma)**2,
            (beta**4)*(1-gamma), beta ** 5, 0, 0, 0, 0]

    print(f'hard-coded solution:\n{soln}')

    # target_for_mc = {}
    # for i in range(graph.number_of_nodes()):
    #     fun = sym.Function(str(Vertex('I', i)))(t)
    #     val = round(soln[i], 2)
    #     target_for_mc[fun] = val
    #
    choice = 0
    # print('\ninitial infected:', choice)
    initial_for_mc = set_initial_state(model, graph, choice=choice)
    # mc_defined = solve_with_mc(model, target_for_mc, graph, initial_for_mc, num_rounds, t_max, timeout, tol)
    # mc_defined.plot()
    # plt.show()
    # print(mc_defined.values[-1].tolist())
    #
    mc_avg = solve_with_mc(model, None, graph, initial_for_mc, t_max=t_max, tol=tol, timeout=timeout, num_rounds=num_rounds)
    # mc_avg.plot()
    # plt.show()
    print(mc_avg.values[-1].tolist())


def solve_with_mc(model, target_average, graph, init_for_MC, num_rounds, t_max, timeout, tol):
    if target_average is not None:
        print(f'\nsolving using MC to defined average using:\n  '
              f'- initial conditions: {init_for_MC}\n  - defined average: {target_average}')
    else:
        print(f'\nsolving using MC to consistent internal average using:\n  '
              f'- initial conditions: {init_for_MC}')
    start = time()
    mc_defined = run_to_average(graph, model, init_for_MC, t_max=t_max, solution=target_average,
                                tolerance=tol, timeout=timeout,
                                num_rounds=num_rounds)
    sol = time() - start
    print(f'time to solve: {sol if sol < timeout else "TO"}')
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
    init_cond = initial_conditions(list(graph.nodes), list(LHS), choice=9, beta=beta, symbol=t)
    init_cond_for_analytic = dict()
    for i in init_cond:
        init_cond_for_analytic[i.subs(t, 0)] = round(init_cond[i], 3)
    print(f'\nInitial conditions:\n{init_cond_for_analytic}\n')
    numerical_solution = solve(equations, graph, init_cond=init_cond, t_max=t_max, step=0.01, atol=1, rtol=1,
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
