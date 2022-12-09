import random

import sympy as sym

from equation import Vertex
from equation.generation import format_term


def initial_conditions(nodes, functions, choice=None, num_initial_infected=1, symbol=0):
    initial_values = {}
    for node in list(nodes):
        initial_values[sym.Function(str(Vertex('S', node)))(symbol)] = 0.999
        initial_values[sym.Function(str(Vertex('I', node)))(symbol)] = 0.001
    if choice is not None and type(choice) is list:
        num_initial_infected = len(choice)

    for i in range(num_initial_infected):
        if choice is None:
            initial_infected = random.choice(nodes)
        else:
            initial_infected = choice[i]
        initial_values[sym.Function(str(Vertex('S', initial_infected)))(symbol)] = 0.001
        initial_values[sym.Function(str(Vertex('I', initial_infected)))(symbol)] = 1.0

    for f in list(functions):
        f = f.subs(sym.symbols('t'), symbol)
        formatted = format_term(str(f).split('\u3009')[0])
        split = formatted.split(" ")
        split = [x for x in split if x != '']
        if len(split) > 1:
            formatted = sym.Function(str('\u3008' + split[0] + '\u3009'))(symbol)
            initial_values[f] = initial_values[formatted]
            for i in range(1, len(split)):
                initial_values[f] *=\
                    initial_values[sym.Function(str('\u3008' + split[i] + '\u3009'))(symbol)]

    return initial_values


def solve_equations(full_equations, init_conditions):
    functions = [x for x in [sym.Function(
        str(each.lhs).replace('Derivative', '').replace('t', '').replace(')', '').replace('(', '').replace(',', ''))
                             (sym.symbols('t')) for each in full_equations] if x]
    return sym.solvers.ode.systems.dsolve_system(eqs=full_equations, funcs=functions, t=sym.symbols('t'),
                                                 ics=init_conditions)
