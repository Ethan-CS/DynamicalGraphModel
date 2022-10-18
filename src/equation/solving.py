import random

import sympy as sym

from equation import Vertex


def initial_conditions(nodes, num_initial_infected=1):
    initial_values = {}
    for node in nodes:
        initial_values[sym.Function(str(Vertex('S', node)))(0)] = 1
        initial_values[sym.Function(str(Vertex('I', node)))(0)] = 1

    for _ in range(0, num_initial_infected):
        initial_infected = random.choice(nodes)
        initial_values[sym.Function(str(Vertex('S', initial_infected)))(0)] = 0
        initial_values[sym.Function(str(Vertex('I', initial_infected)))(0)] = 1

    return initial_values


def solve_equations(full_equations, init_conditions):
    functions = [x for x in [sym.Function(
        str(each.lhs).replace('Derivative', '').replace('t', '').replace(')', '').replace('(', '').replace(',', ''))
                             (sym.symbols('t')) for each in full_equations] if x]
    return sym.solvers.ode.systems.dsolve_system(eqs=full_equations, funcs=functions, t=sym.symbols('t'),
                                                 ics=init_conditions)
