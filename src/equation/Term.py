class Term:
    def __init__(self, _vertices):
        _vertices.sort()
        self._vertices = _vertices

    def __str__(self):
        self._vertices.sort()
        string = "\u3008"
        for v in self._vertices:
            string += f"{v.state}{v.node} "
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


class Vertex:
    def __init__(self, state: chr, node: int):
        self.state = state
        self.node = node

    def __str__(self):
        return f"\u3008{self.state}{self.node}\u3009"

    def __lt__(self, other):
        return self.node < other.node

    def __gt__(self, other):
        return self.node > other.node

    def __eq__(self, other):
        return self.node == other.node and self.state == other.state

    def __le__(self, other):
        return self.node <= other.node

    def __ge__(self, other):
        return self.node >= other.node

    def __ne__(self, other):
        return self.node != other.node or self.state != other.state

    def __hash__(self):
        return 1+self.node.__hash__()
