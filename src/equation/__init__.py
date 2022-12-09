from time import time

import networkx as nx
import sympy as sym

from equation import generation, closing
from equation.Term import Term, Vertex
from equation.generation import get_single_equations, generate_equations
from equation.solving import initial_conditions, solve_equations
from equation.testing_numerical_solvers import solve
from equation_MC_comparison import measure_generation_runtimes
from model_params.cmodel import get_SIR


def main():
    # measure_runtimes()
    print('starting')
    i = 5
    path = nx.path_graph(i)
    start = time()
    equations = generate_equations(path, get_SIR(beta=0.7, gamma=0.1), closures=True)
    print(f'time to generate {len(set().union(*equations.values()))} equations: {time() - start}s')
    # solution = solve(equations, path, beta=0.5, t_max=7, atol=1e-6, rtol=1e-6, step=0.01, print_option='full')
    for j in equations.keys():
        for e in equations[j]:
            print(f'{sym.Integral(e.lhs).doit().func}\'={e.rhs}')
    print(f'generated {len(set().union(*equations.values()))} equations, expected {5 * i - 3}.')  # 5i-3 for closed path


def measure_runtimes():
    with open(f'data/path_data.csv', 'w') as f:
        for i in range(1, 11):
            print(f'i={i}')
            measure_generation_runtimes(g=nx.path_graph(i), num_iter=10, timeout=100, f=f)
    with open(f'data/random_tree_data.csv', 'w') as f:
        for i in range(1, 11):
            print(f'i={i}')
            measure_generation_runtimes(g=nx.random_tree(i), num_iter=10, timeout=100, f=f)
    with open(f'data/cycle_data.csv', 'w') as f:
        for i in range(1, 11):
            print(f'i={i}')
            measure_generation_runtimes(g=nx.cycle_graph(i), num_iter=10, timeout=100, f=f)


if __name__ == '__main__':
    main()
