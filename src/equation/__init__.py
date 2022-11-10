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
    path = nx.complete_graph(5)
    start = time()
    equations = generate_equations(path, get_SIR(beta=0.9, gamma=0.3), closures=True)
    print(f'time to generate {len(set().union(*equations.values()))} equations: {time() - start}s')
    solution = solve(equations, path, beta=0.5, t_max=7, atol=1e-6, rtol=1e-6, step=0.01, print_option='full')
    equations = list(set().union(*equations.values()))
    # for i in range(len(equations)):
    #     print(sym.Integral(equations[i].lhs).doit().func, '(5)=', round(solution[-1][i], 2))


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
