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
    lollipop = networkx.from_numpy_array(lollipop_adj)
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
        total_equations = sum(len(group) for group in all_equations.values())
        assert total_equations == actual_without_closures, (
            f'expected {actual_without_closures}, got {total_equations}'
        )

        closed_equations = generate_equations(copy.deepcopy(path), SIR, closures=True)
        closed_total = sum(len(group) for group in closed_equations.values())
        assert closed_total == actual_with_closures, (
            f'expected {actual_with_closures}, got {closed_total}'
        )


def run_all():
    test_can_be_closed()
    test_num_eqn_lollipop()
    test_num_eqn_path()

