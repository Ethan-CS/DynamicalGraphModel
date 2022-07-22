import networkx

from equation.Term import Term
from equation.closures import can_be_closed


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

