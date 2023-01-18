import copy

import networkx
import numpy as np

from equation.generation import generate_equations
from equation.Term import Term, Vertex
from equation.closing import can_be_closed
from model_params.cmodel import get_SIR

SIR = get_SIR()


def test_can_be_closed():
    path = networkx.path_graph(10)
    vertices = [Vertex(' ', 1), Vertex(' ', 2), Vertex(' ', 3)]
    term = Term(vertices)
    assert can_be_closed(term, path)

    vertices = [Vertex(' ', 2), Vertex(' ', 3), Vertex(' ', 4), Vertex(' ', 5), Vertex(' ', 6)]
    term = Term(vertices)
    assert can_be_closed(term, path)

    vertices = [Vertex(' ', 7), Vertex(' ', 8)]
    term = Term(vertices)
    assert not can_be_closed(term, path)


def test_num_eqn_lollipop():
    lollipop_adj = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    lollipop = networkx.from_numpy_matrix(lollipop_adj)
    all_equations = generate_equations(copy.deepcopy(lollipop), SIR)
    assert len(set().union(*all_equations.values())) == 35

    closed_equations = generate_equations(copy.deepcopy(lollipop), SIR, closures=True)
    assert len(set().union(*closed_equations.values())) == 26


def test_num_eqn_path():
    for i in range(3, 10):
        actual_without_closures = 1 / 2 * (3 * i * i - i + 2)
        actual_with_closures = 5 * i - 3

        path = networkx.path_graph(i)

        all_equations = generate_equations(copy.deepcopy(path), SIR)
        assert len(all_equations) == actual_without_closures, f'expected\n{actual_without_closures}, ' \
                                                              f'got\n{[each.lhs.lhs for each in all_equations]}'

        closed_equations = generate_equations(copy.deepcopy(path), SIR, closures=True)
        assert len(closed_equations) == actual_with_closures, f'expected\n{actual_with_closures}, ' \
                                                              f'got\n{[each.lhs for each in closed_equations]}'


def run_all():
    test_can_be_closed()
    test_num_eqn_lollipop()
    test_num_eqn_path()

