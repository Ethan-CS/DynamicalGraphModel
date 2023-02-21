import signal
from time import time

import networkx as nx
import numpy as np
import sympy as sym

from equation.generation import generate_equations
from equation.solving import initial_conditions, solve_equations
from model_params.cmodel import CModel, get_SIR


def main():
    def handler(signum, frame):  # Timeout handler
        print('solving TO')
        raise Exception("Solving timeout")
    csv_data = ""
    signal.signal(signal.SIGALRM, handler)
    num_v = 10
    num_iter = 5
    time_to_solve_to = 5
    timeout = 60
    for p in [round(p, 2) for p in np.linspace(0.02, 0.5, 25)]:
        print(f' --- p={p} ---')
        for i in range(0, num_iter):
            print(f'iter {i+1} of {num_iter}')
            graph = nx.erdos_renyi_graph(n=num_v, p=p)

            # print(f'graph: {nx.info(graph)} edges')
            # Register the signal function handler

            csv_data += f'{p},{2*graph.number_of_edges() / float(graph.number_of_nodes())},' \
                        f'{get_and_solve_equations(graph, timeout, False, time_to_solve_to)},' \
                        f'{get_and_solve_equations(graph, timeout, True, time_to_solve_to)}\n'

    with open('data/erdos-renyi-equations.csv', 'w') as writer:
        writer.write('probability,avg degree,size no closures,size closures')
        writer.write(csv_data)


def get_and_solve_equations(graph, timeout, closures, t_max):
    signal.alarm(timeout)
    try:
        start = time()
        equations = generate_equations(graph, get_SIR(), closures=closures)
        LHS = []
        for list_of_eqn in equations.values():
            for each_eqn in list_of_eqn:
                LHS.append(sym.Integral(each_eqn.lhs).doit())
        func = [sym.Function(str(type(f)))('t') for f in list(LHS)]
        init_conditions = initial_conditions(list(graph.nodes), functions=func)
        solve_equations(equations, init_conditions, graph, t_max)
        end = time() - start
        print(f'SIR {"with" if closures else "without"} closures: number={count_equations(equations)}, time={end}')
        signal.alarm(0)
        return end
    except Exception as exc:
        return timeout


def get_SEIQRDV_model():
    model = CModel('SEIQRDV')
    model.set_coupling_rate('S:S=>V', name='\\alpha')  # vaccination
    model.set_coupling_rate('S*I:S=>E', name='\\beta_1')  # exposure 1
    model.set_coupling_rate('V:V=>S', name='\\zeta')  # loss of vaccine immunity
    model.set_coupling_rate('E:E=>I', name='\\gamma')  # symptom development
    model.set_coupling_rate('I:I=>Q', name='\\delta')  # quarantine
    model.set_coupling_rate('Q:Q=>R', name='\\epsilon_1')  # recovery
    model.set_coupling_rate('Q:Q=>D', name='\\epsilon_2')  # death
    model.set_coupling_rate('R:R=>S', name='\\eta')  # loss of natural immunity
    return model


def count_equations(equations):
    count = 0
    for num in equations:
        for e in equations[num]:
            count += 1
            # print(f'\dot{{{sym.Integral(e.lhs).doit()}}} &= {e.rhs}\\\\'.replace('〈', '\langle ')
            #       .replace('〉', '\\rangle ').replace('*', ''))
    return count


if __name__ == '__main__':
    main()
