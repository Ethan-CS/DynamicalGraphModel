import sys
from datetime import datetime

import sympy as sym

from equation import generate_equations
from equation.solving import solve_equations, initial_conditions
from model_params.cmodel import get_SIR


def measure_generation_runtimes(g, num_iter=10, model=get_SIR(), timeout=60, f=sys.stdout, solve=False,
                                closures_only=False, t_max=10):
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
                solve_equations(full_equations, init_conditions, g, t_max)
            full_time_taken = datetime.now() - full_start

        closed_start = datetime.now()
        closed_equations = generate_equations(g, model, closures=True)
        if solve:
            init_conditions = initial_conditions(list(g.nodes), get_functions_from_equations(closed_equations))
            solve_equations(closed_equations, init_conditions, g, t_max)
        closed_time_taken = datetime.now() - closed_start

        sys.stdout = f  # Change the standard output to the file we created.
        if not closures_only:
            print(f'{len(g.nodes())},{len(full_equations)},{full_time_taken.seconds}.{full_time_taken.microseconds},'
                  f'{len(closed_equations)},{closed_time_taken.seconds}.{closed_time_taken.microseconds}')
        else:
            print(f'{len(g.nodes())},{len(closed_equations)},{closed_time_taken.seconds}.'
                  f'{closed_time_taken.microseconds}')
        sys.stdout = original_stdout  # Reset the standard output to its original value


def get_functions_from_equations(equations, symbol=sym.symbols('t')):
    LHS = []
    for list_of_eqn in equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())
    return [sym.Function(str(type(f)))(symbol) for f in list(LHS)]
