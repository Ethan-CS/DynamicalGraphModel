import re

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
        if type(_vertices) == Term:
            _vertices = _vertices.vertices()

        if type(_vertices) == sym.Function:
            _vertices = str(_vertices.func)
        elif type(_vertices) is not list:
            _vertices = str(_vertices)

        if type(_vertices) == str:
            _vertices = vertices_to_list(_vertices)
        assert type(_vertices) == list, f'Vertex set must be a list, you provided:' \
                                        f'\nVertex set: {_vertices}, type: {type(_vertices)}'
        for v in _vertices:
            assert type(v) == Vertex, f'not all entries in provided list were vertices: {v}'
        self._vertices = _vertices

    def __str__(self):
        self._vertices.sort()
        string = ""
        for v in self._vertices:
            if type(v) == Vertex:
                string += f"{v.state}_{v.node} "
            else:
                string += f"{v} "
        string = "\u3008" + string[:-1] + "\u3009"
        return string

    def latex_print(self):
        self._vertices.sort()
        string = ""
        for v in self._vertices:
            if type(v) == Vertex:
                string += f"{v.state}_{v.node} "
            else:
                string += f"{v} "
        string = "\\langle" + string[:-1] + "\\rangle"
        return string

    def add(self, vertex):
        self._vertices.append(vertex)
        self._vertices.sort()

    def function(self):
        return sym.Function(str(self))

    @property
    def vertices(self):
        return self._vertices

    def __eq__(self, other):
        if type(other) == Term:
            if len(self.vertices) != len(other.vertices):
                return False
            for v in self.vertices:
                if v not in other.vertices:
                    return False
            for w in other.vertices:
                if w not in self.vertices:
                    return False
            return True
        else:
            return False

    def __hash__(self):
        return tuple(self._vertices).__hash__()

    def node_list(self):
        nodes = []
        for v in self.vertices:
            if type(v) == Vertex:
                nodes.append(v.node)
            else:
                nodes.append(v)
        return nodes

    def state_of(self, vertex):
        for v in self._vertices:
            if v == vertex:
                return ' '
            elif v.node == vertex:
                return v.state

    def remove(self, v):
        self._vertices.remove(v)


class Vertex:
    def __init__(self, state: chr, node: int):
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
