import copy

import networkx
import numpy as np

from equation import generate_equations
from equation.Term import Term
from equation.closures import can_be_closed
from model_params.cmodel import CModel

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')  # Recovery rate


def test_can_be_closed():
    path = networkx.path_graph(10)
    vertices = [1, 2, 3]
    term = Term(vertices)
    assert can_be_closed(term, path)

    vertices = [2, 3, 4, 5, 6]
    term = Term(vertices)
    assert can_be_closed(term, path)

    vertices = [7, 8]
    term = Term(vertices)
    assert not can_be_closed(term, path)


def test_num_eqn_lollipop():
    lollipop_adj = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    lollipop = networkx.from_numpy_matrix(lollipop_adj)
    all_equations = generate_equations(copy.deepcopy(lollipop), SIR)
    assert len(all_equations) == 35

    closed_equations = generate_equations(copy.deepcopy(lollipop), SIR, closures=True)
    assert len(closed_equations) == 26


def test_num_eqn_path():
    for i in range(3, 45):
        actual_without_closures = 1 / 2 * (3 * i * i - i + 2)
        actual_with_closures = 5 * i - 3

        path = networkx.path_graph(i)

        all_equations = generate_equations(copy.deepcopy(path), SIR)
        assert len(all_equations) == actual_without_closures

        closed_equations = generate_equations(copy.deepcopy(path), SIR, closures=True)
        assert len(closed_equations) == actual_with_closures
