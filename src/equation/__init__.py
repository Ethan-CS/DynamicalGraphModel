import networkx as nx
import sympy as sym

from equation import generation, closing
from equation.Term import Term, Vertex
from equation.generation import get_single_equations, generate_equations
from equation.solving import initial_conditions, solve_equations
from equation_MC_comparison import measure_generation_runtimes
from model_params.cmodel import get_SIR


def main():
    # measure_runtimes()
    print('starting')
    path = nx.path_graph(10)
    equations = generate_equations(path, get_SIR(), closures=False)
    LHS = [sym.Integral(each.lhs).doit() for each in set().union(*equations.values())]
    RHS_expressions = [each.rhs for each in set().union(*equations.values())]
    RHS = []
    # for r in RHS_expressions:
    #     RHS.extend([f.func for f in r.atoms(sym.Function)])
    for r in RHS_expressions:
        RHS.extend([f.func for f in r.atoms(sym.Function)])
    print('symbols with no equations:')
    for r in RHS:
        if r(sym.symbols('t')) not in LHS:
            print(str(r))
    print(f'{len(LHS)}, {len(set(RHS))}')
    print('DONE')
    # for e in equations:
    #     print(f'{sym.Integral(e.lhs).doit()}\'={e.rhs}')


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
