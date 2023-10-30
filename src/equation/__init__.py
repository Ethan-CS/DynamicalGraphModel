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
    Example usage of generating equations method on path graphs from 2 to 10 vertices, which are then printed to console
    """
    for i in range(2, 11):
        print(f'-----{i}-----')
        equations = generate_equations(nx.path_graph(i), get_SIR(beta=0, gamma=0), closures=True)
        for length in equations:
            for eq in equations[length]:
                print(f'\dot{{{sym.integrate(eq.lhs)}}} &= {eq.rhs}'.replace('(t)', '').replace('〈', '\\langle ')
                      .replace('〉', '\\rangle ').replace('\\beta', '\\beta_{}').replace('\\gamma', '\\gamma_{}'))


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
