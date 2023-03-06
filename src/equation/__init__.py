import signal
from pathlib import Path
from time import time

import networkx as nx
import numpy as np
import sympy as sym

from equation.generation import generate_equations
from equation.solving import initial_conditions, solve_equations
from model_params.cmodel import CModel, get_SIR


def main():
    times = 'n,time\n'
    filePath = Path("data/cycle_data.csv")
    filePath.touch(exist_ok=True)
    for i in range(3, 4):
        print(f' --- i={i} ---')
        for _ in range(2):
            cycle = nx.cycle_graph(i)
            print(f'should be {3*i**2-3*i} equations')
            time_to_solve = get_and_solve_equations(cycle, 60, False, 5)
            print(f'time: {time_to_solve}')
            times += f'{i},{time()}\n'
    print(times)
    with open(filePath, 'w+') as f:
        f.write(times)


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
        signal.alarm(0)
        return end
    except Exception as exc:
        return timeout


def get_SEIQRDV_model():
    model = CModel('SEIQRDV')
    model.set_coupling_rate('S:S=>V', name='\\alpha')  # vaccination
    model.set_coupling_rate('S*I:S=>E', name='\\beta_1')  # exposure 1
    model.set_coupling_rate('S*E:S=>E', name='\\beta_2')  # exposure 2
    model.set_coupling_rate('V:V=>S', name='\\zeta')  # loss of vaccine immunity
    model.set_coupling_rate('E:E=>I', name='\\gamma')  # symptom development
    model.set_coupling_rate('I:I=>Q', name='\\delta')  # quarantine
    model.set_coupling_rate('Q:Q=>R', name='\\epsilon_1')  # recovery
    model.set_coupling_rate('Q:Q=>D', name='\\epsilon_2')  # death
    model.set_coupling_rate('R:R=>S', name='\\eta')  # loss of natural immunity
    return model


def count_equations(equations, p=False):
    count = 0
    for num in equations:
        for e in equations[num]:
            count += 1
            if p:
                print(f'\dot{{{sym.Integral(e.lhs).doit()}}} &= {e.rhs}\\\\'
                      .replace('〈', '\langle ').replace('〉', '\\rangle ').replace('*', ''))
    return count


if __name__ == '__main__':
    main()
