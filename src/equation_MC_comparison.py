import sys
from datetime import datetime

import networkx as nx
import sympy as sym
from scipy.integrate import solve_ivp, odeint

import equation
from equation import generate_equations, solve_equations, initial_conditions
from model_params.cmodel import get_SIR
from monte_carlo.monte_carlo import monte_carlo_sim


def measure_generation_runtimes(g, num_iter=10, model=get_SIR(), timeout=60, f=sys.stdout, solve=False,
                                closures_only=False):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = f  # Change the standard output to the file we created.
    print(f'number of vertices,{"number equations (full),time (full)," if not closures_only else ""}'
          f'num equations (closed),time (closed)')
    sys.stdout = original_stdout  # Reset the standard output to its original value
    for _ in range(0, num_iter):
        print(f'iter {_ + 1} of {num_iter}')
        full_equations = []
        full_time_taken = 0
        if not closures_only:
            full_start = datetime.now()
            full_equations = generate_equations(g, model)
            if solve:
                init_conditions = initial_conditions(list(g.nodes), get_functions_from_equations(full_equations))
                solve_equations(full_equations, init_conditions)
            full_time_taken = datetime.now() - full_start

        closed_start = datetime.now()
        closed_equations = generate_equations(g, model, closures=True)
        if solve:
            init_conditions = initial_conditions(list(g.nodes), get_functions_from_equations(closed_equations))
            solve_equations(closed_equations, init_conditions)
        closed_time_taken = datetime.now() - closed_start

        sys.stdout = f  # Change the standard output to the file we created.
        if not closures_only:
            print(f'{len(g.nodes())},{len(full_equations)},{full_time_taken.seconds}.{full_time_taken.microseconds},'
                  f'{len(closed_equations)},{closed_time_taken.seconds}.{closed_time_taken.microseconds}')
        else:
            print(f'{len(g.nodes())},{len(closed_equations)},{closed_time_taken.seconds}.'
                  f'{closed_time_taken.microseconds}')
        sys.stdout = original_stdout  # Reset the standard output to its original value


def check_in_range(all_results, solution, closeness):
    num_results = len(all_results)
    average = [0 for _ in range(len(all_results[list(all_results.keys())[0]]))]
    for result in all_results.keys():
        for i in range(len(all_results[result])):
            average[i] += all_results[result][i]

    for i in range(len(average)):
        average[i] = average[i] / num_results

    print(average)

    for i in range(len(average)):
        if not solution[i] - closeness * solution[i] <= average[i] <= solution[i] + closeness * solution[i]:
            return False
    return True


def measure_mc_runtimes(g, initial_state, t_max, solution, accepted_closeness=0.005, model=get_SIR(), num_iter=10,
                        timeout=180, f=sys.stdout):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = f  # Change the standard output to the file we created.
    print('number of vertices,time to solve')
    sys.stdout = original_stdout  # Reset the standard output to its original value
    for _ in range(0, num_iter):
        print(f'iter {_ + 1} of {num_iter}')
        start = datetime.now()
        in_range = False
        all_results = {}
        i = 0
        while not in_range:
            all_results[i] = monte_carlo_sim(g, model, initial_state, t_max)
            in_range = check_in_range(all_results, solution, accepted_closeness)
        time_taken = datetime.now() - start


def get_functions_from_equations(equations, symbol=sym.symbols('t')):
    return [x for x in [sym.Function(
        str(each.lhs).replace('Derivative', '').replace('t', '').replace(')', '').replace('(', '').replace(',', ''))
                             (symbol) for each in equations] if x]


# def get_exact_solution(g: nx.Graph):
    # all_equations = generate_equations(g, get_SIR(), closures=True)
    # functions = get_functions_from_equations(all_equations, sym.symbols('t'))
    # state_0 = initial_conditions(list(g.nodes), functions)
    # print(state_0)
    # # solution = odeint(eq=all_equations, func=functions, ics=state_0, doit=True)
    # # print(solution)


all_equations = generate_equations(nx.path_graph(5), get_SIR(), closures=True)
functions = get_functions_from_equations(all_equations, sym.symbols('t'))
print(functions)
