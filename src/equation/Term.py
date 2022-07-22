class Term:
    def __init__(self, _vertices: list):
        if type(_vertices) == tuple:
            print("tuple!", _vertices[0])
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
        return len(list(set(self.vertices)-set(other.vertices))) == 0 \
               and len(list(set(other.vertices)-set(self.vertices))) == 0

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
        self.node = node

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
        return 1+self.node.__hash__()
