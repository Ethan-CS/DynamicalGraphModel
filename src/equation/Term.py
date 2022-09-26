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
    def __init__(self, _vertices):
        assert _vertices != [], 'Vertex list is empty'
        if type(_vertices) == Term:
            _vertices = _vertices.vertices()

        if type(_vertices) == sym.Symbol or type(_vertices) == sym.Mul:
            _vertices = str(_vertices)

        if type(_vertices) == str:
            if '/' in _vertices:
                _vertices = _vertices.split('/', 1)[0]
            if '*' in _vertices:
                _vertices = _vertices.split('*', 1)[1]
            # Remove left and right angle brackets, if necessary
            _vertices = _vertices.replace('\u3008', '')
            _vertices = _vertices.replace('\u3009', '')
            # Remove all spaces
            _vertices = _vertices.replace(" ", "")
            new_vert = []
            separated = re.findall(r"[^\W\d_]+|\d+", _vertices)  # creates array with chars and ints separated
            if type(separated[0]) == str:
                for i in range(0, len(separated)-1, 2):
                    new_vert.append(Vertex(separated[i], separated[i+1]))
            else:
                for i in range(0, len(_vertices)):
                    new_vert.append(Vertex(' ', int(_vertices[i])))
            _vertices = new_vert
            _vertices.sort()

        if type(_vertices[0]) == int:
            new_vertices = []
            for v in _vertices:
                new_vertices.append(Vertex(' ', v))
                self._vertices = _vertices
        else:
            self._vertices = _vertices

    def __str__(self):
        self._vertices.sort()
        string = "\u3008"
        for v in self._vertices:
            if type(v) == Vertex:
                string += f"{v.state}{v.node} "
            else:
                string += f"{v} "
        string = string[:-1] + "\u3009"
        return string

    def add(self, vertex):
        self._vertices.append(vertex)
        self._vertices.sort()

    @property
    def vertices(self):
        return self._vertices

    def __eq__(self, other):
        o = Term(other)
        return len(list(set(self.vertices) - set(o.vertices))) == 0 \
               and len(list(set(o.vertices) - set(self.vertices))) == 0

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
        return f"\u3008{self.state}{self.node}\u3009"

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
        elif type(other) == int:
            return self.node == other
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
