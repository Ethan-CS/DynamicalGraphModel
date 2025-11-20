import re
from bisect import insort

import sympy as sym


def is_valid(candidate: str):
    try:
        Term(candidate)
    except AssertionError:
        print('Bad candidate:', candidate)
        return False
    return True


class Term:
    """
    Defines a term in an equation, using SymPy for later algebraic manipulation.
    """
    def __init__(self, _vertices):
        assert _vertices != [], 'Vertex list is empty'
        if isinstance(_vertices, Term):
            _vertices = list(_vertices.vertices)

        if isinstance(_vertices, sym.Function):
            _vertices = str(_vertices.func)
        elif not isinstance(_vertices, list):
            _vertices = str(_vertices)

        if isinstance(_vertices, str):
            _vertices = vertices_to_list(_vertices)
        assert isinstance(_vertices, list), f'Vertex set must be a list, you provided:' \
                                            f'\nVertex set: {_vertices}, type: {type(_vertices)}'
        for v in _vertices:
            assert isinstance(v, Vertex), f'not all entries in provided list were vertices: {v}'
        self._vertices = sorted(_vertices)

    def __str__(self):
        string = " ".join(
            f"{v.state}_{v.node}" if isinstance(v, Vertex) else f"{v}"
            for v in self._vertices
        )
        return "\u3008" + string + "\u3009"

    def latex_print(self):
        string = " ".join(
            f"{v.state}_{v.node}" if isinstance(v, Vertex) else f"{v}"
            for v in self._vertices
        )
        return "\\langle" + string + "\\rangle"

    def add(self, vertex):
        insort(self._vertices, vertex)

    def function(self):
        return sym.Function(str(self))

    @property
    def vertices(self):
        return tuple(self._vertices)

    def __eq__(self, other):
        if isinstance(other, Term):
            return self._vertices == other._vertices
        return False

    def __hash__(self):
        return hash(tuple(self._vertices))

    def node_list(self):
        return [v.node if isinstance(v, Vertex) else v for v in self._vertices]

    def state_of(self, vertex):
        for v in self._vertices:
            if v == vertex:
                return ' '
            if isinstance(v, Vertex) and v.node == vertex:
                return v.state

    def remove(self, v):
        self._vertices.remove(v)


class Vertex:
    def __init__(self, state: str, node: int):
        self.state = state
        # assert type(node) == int, f'the vertex {node} is not an int, instead {type(node)}'
        self.node = int(node)

    def __str__(self):
        return f"\u3008{self.state}_{self.node}\u3009"

    def __lt__(self, other):
        if type(other) == Vertex:
            return self.node < other.node
        elif type(other) == int:
            return self.node < other
        else:
            return False

    def __gt__(self, other):
        if type(other) == Vertex:
            return self.node > other.node
        elif type(other) == int:
            return self.node > other
        else:
            return False

    def __eq__(self, other):
        if type(other) == Vertex:
            return self.node == other.node and self.state == other.state
        else:
            return False

    def __le__(self, other):
        return self.node <= other.node

    def __ge__(self, other):
        return self.node >= other.node

    def __ne__(self, other):
        return self.node != other.node or self.state != other.state

    def __hash__(self):
        return 1 + self.node.__hash__()


def vertices_to_list(term):
    if '/' in str(term):
        term = str(term).replace('/', '*')
    if '*' in str(term):
        term = list(str(term).split('*', 1))[1]
    term = term.replace('(t)', '').replace(' ', '').replace('\u3008', '').replace('\u3009', '')
    vertices = []
    if type(term) == str:
        split = re.findall(r"[^\W\d_]+|\d+", term)
        for i in range(0, len(split) - 1, 2):
            vertices.append(Vertex(split[i], split[i + 1]))
    return vertices
