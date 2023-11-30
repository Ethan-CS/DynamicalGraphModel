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
    """
    Example usage of generating full_equations method on path graphs from 2 to 10 vertices, which are then printed to console
    """
    model = CModel('SEIRV')
    model.set_coupling_rate('S*I:S=>E', 1, name='\\beta_1')  # Infection rate
    model.set_coupling_rate('S*E:S=>E', 1, name='\\beta_2')  # Infection rate
    model.set_coupling_rate('I:I=>R', 3, name='\\gamma')  # Recovery rate
    model.set_coupling_rate('S:S=>V', 5, name='\\tau')  # Vaccination rate
    model.set_coupling_rate('V:V=>E', 4, name='\\delta')  # Vaccination rate

    print(model.couplings)

    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)])

    print(f'without closures={count_equations(generate_equations(graph, model, closures=False), False)}')
    print(f'   with closures={count_equations(generate_equations(graph, model, closures=True), False)}')


def get_and_solve_equations(graph, closures, t_max, model=get_SIR()):
    """
    Example function that takes a specified graph, generates and solves equations and returns the length of time this
    took.

    :param graph: the underlying graph of the model.
    :param closures: true if closures are to be used, false if not.
    :param t_max: the timestep up to which to solve the equations.
    :param model: a compartmental model to apply to the underlying graph (defauls to SIR)
    :return: the time taken for code to generate and solve the system of equations.
    """
    start = time()
    equations = generate_equations(graph, model, closures=closures)
    LHS = []
    for list_of_eqn in equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())
    func = [sym.Function(str(type(f)))('t') for f in list(LHS)]
    init_conditions = initial_conditions(list(graph.nodes), functions=func)
    solve_equations(equations, init_conditions, graph, t_max)
    end = time() - start
    return end


def count_equations(equations, p=False):
    """
    Counts the number of equations in a system.

    :param equations: the system of equations to count.
    :param p: true if equations should be printed as counted, false otherwise (defaults to false).
    :return: the number of equations in the system.
    """
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
