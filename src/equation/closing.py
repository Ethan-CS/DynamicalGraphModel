import copy

import networkx as nx
import sympy as sym
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


# True if the specified term can be closed (i.e. contains a cut-vertex), false otherwise
def can_be_closed(term: Term, g: Graph):
    return False if len(term.vertices) < 3 else len(list(nx.articulation_points(nx.subgraph(g, term.node_list())))) != 0


def replace_with_closures(term: Term, graph: Graph):
    if type(term) == sym.Symbol or sym.Function:
        term = Term(str(term))

    # We work on the subgraph induced by the vertices in the term
    induced_subgraph = Graph(nx.subgraph(copy.deepcopy(graph), list(term.node_list())))
    # Find cut-vertices in induced subgraph and arbitrarily select first in list
    # If there are several, they are identified later on and same procedure followed to replace
    cut = list(nx.articulation_points(induced_subgraph))[0]
    induced_subgraph.remove_node(cut)

    sub_terms = [1 / sym.Symbol(str(Vertex('S', cut)))]  # Will contain all terms to replace original with

    graph_components = []  # List of the connected components after cut-vertex removal with CV replaced in each
    for c in list(nx.connected_components(induced_subgraph)):
        c.add(cut)
        graph_components.append(c)

    original_states = {}
    for vertex in term.vertices:
        original_states[vertex.node] = vertex.state

    for component in graph_components:
        as_vertices = [Vertex(original_states[v], v) for v in component]
        if not can_be_closed(Term(list(as_vertices)), graph):
            # Cannot be closed further, add to list of substitutions and continue
            sub_terms.append(sym.Symbol(str(Term(as_vertices))))
        else:
            # There are remaining cut-vertices in the term, so close the term on those too
            sub_terms.append(replace_with_closures(Term(as_vertices), graph))

    return sub_terms
