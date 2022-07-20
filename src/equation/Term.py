class Term:
    def __init__(self, vertices):
        vertices.sort()
        self.vertices = vertices

    def __str__(self):
        self.vertices.sort()
        string = "\u3008"
        for v in self.vertices:
            string += f"{v.state}{v.node} "
        string = string[:-1] + "\u3009"
        return string

    def add(self, vertex):
        self.vertices.append(vertex)
        self.vertices.sort()


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
        return id(self)
