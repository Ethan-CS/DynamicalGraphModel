import sys
from time import time

import networkx as nx
import numpy as np
import sympy as sym

from equation import generation, closing
from equation.Term import Term, Vertex
from equation_MC_comparison import measure_runtimes
from equation.generation import get_single_equations, generate_equations
from equation.solving import initial_conditions, solve_equations, solve
from model_params.cmodel import get_SIR, CModel

SYS_STDOUT = sys.stdout


def main():
    timeout = 60
    v = 10
    iterations = 5
    t_max = 5
    for p in np.linspace(0.01, 0.2, 20):
        p = round(p, 2)
        print(f'\n *** p={p} ***')
        print(f'\n - Monte Carlo -')
        measure_runtimes('random', v, iterations, p, t_max, 'mc', timeout)
        print(f'\n - Equations -')
        measure_runtimes('random', v, iterations, p, t_max, 'equations', timeout)


def runtime_large_model():
    print('starting')
    i = 3
    path = nx.path_graph(i)
    model = CModel('SEIQRDV')
    model.set_coupling_rate('S:S=>V', name='\\alpha')  # vaccination
    model.set_coupling_rate('S*I:S=>E', name='\\beta')  # exposure
    model.set_coupling_rate('V:V=>S', name='\\zeta')  # loss of vaccine immunity
    model.set_coupling_rate('E:E=>I', name='\\gamma')  # symptom development
    model.set_coupling_rate('I:I=>Q', name='\\delta')  # quarantine
    model.set_coupling_rate('Q:Q=>R', name='\\epsilon_1')  # recovery
    model.set_coupling_rate('Q:Q=>D', name='\\epsilon_2')  # death
    model.set_coupling_rate('R:R=>S', name='\\eta')  # loss of natural immunity
    start = time()
    equations = generate_equations(path, get_SIR(0, 0), closures=False)
    print(f'time to generate {len(set().union(*equations.values()))} equations: {time() - start}s')
    # solution = solve(equations, path, t_max=7, atol=1e-6, rtol=1e-6, step=0.01)
    # print(f'time to solve: {time() - start}')
    for j in equations.keys():
        for e in equations[j]:
            print(f'\\dot{{{sym.Integral(e.lhs).doit().func}}} &= {e.rhs} \\\\'
                  .replace('\u3008', '\\langle ').replace('\u3009', '\\rangle').replace('(t)', '').replace('*', ''))
    print(f'generated {len(set().union(*equations.values()))} equations.')  # 5i-3 for closed path


if __name__ == '__main__':
    main()
