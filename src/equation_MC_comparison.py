import sys

import signal
from datetime import datetime

import networkx as nx
import numpy as np
import sympy as sym

from equation.generation import generate_equations
from equation.solving import solve_equations, initial_conditions
from model_params.cmodel import CModel
from monte_carlo.mc_sim import run_to_average
from monte_carlo.mc_sim import set_initial_state

SYS_STDOUT = sys.stdout


def handler(signum, frame):
    raise Exception("TIMEOUT")


def measure_generation_runtimes(g, num_iter, model, timeout, f, solve, closures_only, t_max):
    for _ in range(0, num_iter):
        sys.stdout = SYS_STDOUT
        print(f'iter {_ + 1} of {num_iter}')
        full_equations = {}
        full_time_taken = 0

        signal.signal(signal.SIGALRM, handler)  # set-up timeout handler

        if not closures_only:
            full_start = datetime.now()
            signal.alarm(timeout)  # start the timer
            try:
                full_equations = generate_equations(g, model)
                if solve:
                    init_conditions = initial_conditions(list(g.nodes), get_functions_from_equations(full_equations))
                    solve_equations(full_equations, init_conditions, g, t_max)
            except Exception as exc:
                print(exc)  # process has timed out, break and move on
            signal.alarm(0)  # cancel timer if process completed before timeout
            full_time_taken = datetime.now() - full_start

        signal.alarm(timeout)  # start the timer
        closed_start = datetime.now()
        closed_equations = []
        try:
            closed_equations = generate_equations(g, model, closures=True)
            if solve:
                init_conditions = initial_conditions(list(g.nodes), get_functions_from_equations(closed_equations))
                solve_equations(closed_equations, init_conditions, g, t_max)
        except Exception as exc:
            print(exc)  # process has timed out, break and move on
        signal.alarm(0)  # cancel timer if process completed before timeout
        closed_time_taken = datetime.now() - closed_start

        if type(closed_equations) is list:
            print(f'closed equations is a list:\n{closed_equations}')
        if type(full_equations) is list:
            print(f'full equations is a list:\n{full_equations}')

        if (type(closed_equations) is not list) and (type(full_equations) is not list):
            sys.stdout = f  # Change the standard output to the file
            if not closures_only:
                print(f'{len(g.nodes())},{len(list(set().union(*full_equations.values())))},{full_time_taken.seconds}.{full_time_taken.microseconds},'
                      f'{len(list(set().union(*closed_equations.values())))},{closed_time_taken.seconds}.{closed_time_taken.microseconds}')
            else:
                print(f'{len(g.nodes())},{len(list(set().union(*closed_equations.values())))},{closed_time_taken.seconds}.'
                      f'{closed_time_taken.microseconds}')
        sys.stdout = SYS_STDOUT  # Reset the standard output to its original value


def measure_mcmc_runtimes(g, p, num_iter, model, timeout, f, t_max):
    for _ in range(0, num_iter):
        sys.stdout = SYS_STDOUT
        print(f'iter {_ + 1} of {num_iter}')
        full_start = datetime.now()  # Get the start time

        signal.signal(signal.SIGALRM, handler)  # set-up timeout handler
        signal.alarm(timeout)  # if we get past timeout, will break out
        try:
            # Do the MCMC simulation
            init_state = set_initial_state(model, g)
            run_to_average(g, model, init_state, t_max, tolerance=1e-3, timeout=timeout, num_rounds=num_iter)
        except Exception as exc:
            print(exc)
        full_time_taken = datetime.now() - full_start
        signal.alarm(0)  # cancel timer if process completed before timeout

        sys.stdout = f  # print to file
        print(f'{len(g.nodes())},{f"{p}," if p != 0 else ""}{full_time_taken.seconds}.{full_time_taken.microseconds}')
        sys.stdout = SYS_STDOUT


def get_functions_from_equations(equations, symbol=sym.symbols('t')):
    LHS = []
    for list_of_eqn in equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())
    return [sym.Function(str(type(f)))(symbol) for f in list(LHS)]


def measure_runtimes(graph_type, num_vertices, iterations, p, t_max, method, timeout, f):
    sir_model = CModel.make_SIR(0.5, 0.1)
    sys.stdout = f  # Change the standard output to the file we created.

    sys.stdout = SYS_STDOUT
    print(f'\nnumber of vertices: {num_vertices}')

    # Get the specified graph
    g = None
    if graph_type == 'path':
        g = nx.path_graph(num_vertices)
    elif graph_type == 'cycle':
        g = nx.cycle_graph(num_vertices)
    elif graph_type == 'random':
        g = nx.erdos_renyi_graph(num_vertices, p)

    # Create (and solve) the model
    if method == 'eq' or method == 'equations':
        measure_generation_runtimes(g, iterations, sir_model, timeout, f, True, True, t_max)
    elif method == 'mc' or method == 'mcmc' or method == 'monte carlo':
        measure_mcmc_runtimes(g, p, iterations, sir_model, timeout, f, t_max)


def run_measure():
    graph_type = 'random'
    timeout = 60
    v = 20
    iterations = 5
    t_max = 5

    # range_of_probs = np.linspace(0.02, 0.2, 10)
    range_of_probs = [0.5]
    # method = 'mc'
    # print(f'\n - Monte Carlo -')
    # with open(f'data/{graph_type}_{method.replace(" ", "_")}_same_v_data.csv', 'w+') as file:
    #     # for i in range(2, num_vertices + 1):
    #     sys.stdout = file
    #     print(f'num of vertices,p,time to solve')
    #     for p in range_of_probs:
    #         sys.stdout = SYS_STDOUT
    #         print(f'\n *** p={p} ***')
    #         measure_runtimes(graph_type, v, iterations, p, t_max, method, timeout, file)

    method = 'equations'
    print(f'\n - Equations -')
    with open(f'data/{graph_type}_{method.replace(" ", "_")}_same_v_data.csv', 'w+') as file:
        # for i in range(2, num_vertices + 1):
        sys.stdout = file
        print(f'num of vertices,p,num equations,time to solve')
        for p in range_of_probs:
            sys.stdout = SYS_STDOUT
            print(f'\n *** p={p} ***')
            measure_runtimes(graph_type, v, iterations, p, t_max, method, timeout, file)


run_measure()
