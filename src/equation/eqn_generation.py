"""
To generate equations:
- GET_SINGLES (base-case)
- GET_EQUATIONS (recursive helper)

GET_SINGLES(input graph, model):
- initialise an empty list to store equations for singles
- For each vertex v in the graph:
    - For each model state s in the model:
        - add EQN(v in state s) i.e. eqn for v being in state s to list of eqns
- Return list  of equations for singles

GET_EQUATIONS(graph, model, prev_eqns, length)
- If length <= num vertices in graph: > Not reached full system size
    - initialise a new list, next_eqns
    - For eqn in prev_eqns:
        - For term in eqn:
            - If length of term == length:
                - Add EQN(term) if not already added
    - Add all from next_eqns to list to return
    - Add all from recursive call to GET_EQUATIONS(graph, model, prev_eqns, length+1)
    - Return equations
Else return an empty list
"""
from cpyment import CModel
import networkx
from equation.Term import Vertex, Term
from equation.model.helpers import dynamically_relevant
from model_params.helpers import Coupling, coupling_types


def get_single_equations(graph, model):
    """
    Given a contact graph and a compartmental model, returns the list of tuples containing only one vertex for which we
    require equations in the dynamical system. These equations are used as the base case for equation generation.

    :type graph: networkx.Graph
    :type model: CModel
    :param graph: A networkx graph representing the contact network in the model
    :param model: A specified compartmental model
    :return: The list of single-vertex tuples for which we require equations
    """
    singles_terms = []
    singles_equations = {}
    for state in dynamically_relevant(model):
        for node in graph.nodes:
            term = Vertex(state, node)
            singles_terms.append(term)
            singles_equations[Term([term])] = chain_rule(Term([term]), graph, model)

    return singles_equations


def generate_equations(singles, length, graph, model):
    equations = dict(singles)
    if length <= len(graph.nodes):
        for LHS in singles.keys():
            for term in singles[LHS]:
                if len(term[0].vertices) == length and (not term[0] in equations):
                    equations[term[0]] = chain_rule(term[0], graph, model)

        next_eqns = generate_equations(equations, length + 1, graph, model)
        for eqn in next_eqns:
            if eqn not in equations:
                equations[eqn] = next_eqns[eqn]

    return equations


def add_terms(v, term_clone, transition, neighbours_of_v):
    neighbours_of_v_not_in_tuple = list(neighbours_of_v)
    for vertex in neighbours_of_v:
        for other_vertex in term_clone.vertices:
            if vertex == other_vertex.node:
                neighbours_of_v_not_in_tuple.remove(vertex)
                break
    terms = []
    if transition[0] == Coupling.NEIGHBOUR_ENTER:
        # e.g. v is in state I, so change v to S and each neighbour in turn to I
        other_state_for_v = transition[1][0]
        for n in neighbours_of_v:
            vertices = set(term_clone.vertices)
            vertices.add(Vertex(other_state_for_v, v.node))
            for vertex in vertices:
                if vertex.node == n:
                    vertices.remove(vertex)
                    break
            neighbour = Vertex(v.state, n)
            vertices.add(neighbour)
            terms.append((Term(list(vertices)), transition[2]))
    elif transition[0] == Coupling.NEIGHBOUR_EXIT:
        # e.g. v in state S so can exit if any neighbour in I
        other_state_for_neighbours = transition[1].split('*')[1][0]
        for n in neighbours_of_v:
            vertices = set(term_clone.vertices)
            vertices.add(v)
            for vertex in vertices:
                if vertex.node == n:
                    vertices.remove(vertex)
                    break
            vertices.add(Vertex(other_state_for_neighbours, n))
            terms.append((Term(list(vertices)), f'-{transition[2]}'))
    elif transition[0] == Coupling.ISOLATED_ENTER:
        # e.g. v in state R, gets there through recovery after being in i
        other_state_for_v = transition[1][0]
        vertices = set(term_clone.vertices)
        vertices.add(Vertex(other_state_for_v, v.node))
        terms.append((Term(list(vertices)), transition[2]))
    elif transition[0] == Coupling.ISOLATED_EXIT:
        # e.g. v in state I, transitions out without input of neighbours
        vertices = set(term_clone.vertices)
        vertices.add(v)
        terms.append((Term(list(vertices)), f'-{transition[2]}'))
    else:
        print('nothing I could do!')
    return terms


def derive(v, term_clone, graph, model):
    """
    Computes the equation for the specified term.
    :param v: the vertex we are deriving (in the chain rule)
    :param term_clone: the rest of the terms (not currently derived)
    :param graph: the model contact graph
    :param model: definition of the compartmental model
    :return: a list of tuples, each containing a term and the coefficient of the term
    """
    # Get neighbours
    neighbours_of_v = [n for n in graph.neighbors(v.node)]
    # Get mapping of states to how they are entered/exited
    transitions = coupling_types(model)
    all_terms = []
    for transition in transitions[v.state]:
        terms = add_terms(v, term_clone, transition, neighbours_of_v)
        for t in terms:
            all_terms.append(t)
    return all_terms


# d(vw)/dt = (dv/dt)w + v(dw/dt)
def chain_rule(term, graph, model):
    vertices = list(term.vertices)
    all_terms = []
    for v in vertices:
        term_clone = Term(vertices)
        term_clone.vertices.remove(v)
        terms = derive(v, term_clone, graph, model)
        for term in terms:
            all_terms.append(term)
    return all_terms
