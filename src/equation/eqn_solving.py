import random

import sympy as sym

from equation import Vertex


def initial_conditions(nodes, num_initial_infected):
    initial_values = {}
    for node in nodes:
        initial_values[sym.Function(str(Vertex('S', node)))] = 1
        initial_values[sym.Function(str(Vertex('I', node)))] = 1

    for _ in range(0, num_initial_infected):
        initial_infected = random.choice(nodes)
        initial_values[sym.Function(str(Vertex('S', initial_infected)))] = 0
        initial_values[sym.Function(str(Vertex('I', initial_infected)))] = 1

    return initial_values
