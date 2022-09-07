import copy

import networkx as nx
from networkx import Graph

from equation.Term import Term, Vertex


def find_cut_vertices(graph: Graph):
    """
    Uses networkx built-in articulation point finding function. This uses a depth-first search to identify cut-vertices.
    (Recursive version) Create a recursive function that takes the index of the node and a visited array: (1) Mark the
    current node as visited and print the node, (2) Traverse all the adjacent and unmarked nodes and call the recursive
    function with the index of the adjacent node. NB, the networkx function employs a non-recursive DFS.

    Complexity
    ----------
    - Time complexity: O(V + E), where V is the number of vertices and E is the number of edges in the graph.
    - Space Complexity: O(V), since an extra visited array of size V is required.

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    A list of all articulation points in the graph
    """
    cuts = nx.articulation_points(graph)
    return cuts


def can_be_closed(term: Term, graph: Graph):
    """
    Returns true if the specified term can be closed (i.e. contains a cut-vertex) and false otherwise.
    """
    if len(term.vertices) < 3:
        return False
    return len([nx.articulation_points(nx.subgraph(graph, term.node_list()))]) > 0


def replace_with_closures(term: Term, graph: Graph):
    sub_terms = []
    induced_subgraph = Graph(nx.subgraph(copy.deepcopy(graph), list(term.node_list())))
    cut_vertices = list(nx.articulation_points(induced_subgraph))
    cut = cut_vertices[0]
    induced_subgraph.remove_node(cut)
    cc = list(nx.connected_components(induced_subgraph))
    graph_components = []
    for c in cc:
        c.add(cut)
        graph_components.append(c)

    original_states = {}
    for vertex in term.vertices:
        original_states[vertex.node] = vertex.state

    for component in graph_components:
        as_vertices = []
        for v in component:
            as_vertices.append(Vertex(original_states[v], v))
        if not can_be_closed(Term(list(as_vertices)), graph):
            sub_terms.append(as_vertices)
        else:
            replace_with_closures(Term(as_vertices), graph)

    return sub_terms
