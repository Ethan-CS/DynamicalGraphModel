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
from enum import Enum
from cpyment import CModel
import networkx
from equation.Term import Vertex, Term


def get_singles_terms(graph, model):
    """
    Given a contact graph and a compartmental model, returns the list of tuples containing only one vertex for which we
    require equations in the dynamical system.

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


class Coupling(Enum):
    NEIGHBOUR_ENTER = 0
    NEIGHBOUR_EXIT = 1
    ISOLATED_ENTER = 2
    ISOLATED_EXIT = 3


def add_terms(v, term_clone, transition, neighbours_of_v):
    terms = []
    if transition[0] == Coupling.NEIGHBOUR_ENTER:
        # e.g. v is in state I, so change v to S and each neighbour in turn to I
        other_state_for_v = transition[1][0]
        for n in neighbours_of_v:
            vertices = set(term_clone.vertices)
            vertices.add(Vertex(other_state_for_v, v.node))
            neighbour = Vertex(v.state, n)
            vertices.add(neighbour)
            terms.append((Term(list(vertices)), transition[2]))
    elif transition[0] == Coupling.NEIGHBOUR_EXIT:
        # e.g. v in state S so can exit if any neighbour in I
        other_state_for_neighbours = transition[1].split('*')[1][0]
        for n in neighbours_of_v:
            vertices = set(term_clone.vertices)
            vertices.add(v)
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


# Maintain a global variable so that we do not
# need to re-determine map each time it is needed
_coupling_map = {}


def coupling_types(model):
    if _coupling_map == {}:
        # Initialise the dictionary keys (one for each model state)
        for state in model.states:
            _coupling_map[state] = []

        for state in model.states:
            for couple in model.couplings:
                # Get the situation under which a transition occurs
                transition = model.couplings[couple][0].split(':')[0]
                # Does the transition contain a state we are currently interested in?
                if transition.count(state) > 0:
                    if transition[0] == state:  # EXIT TRANSITION
                        if transition.count('*') > 0:  # NEEDS A NEIGHBOUR
                            _coupling_map[state].append((Coupling.NEIGHBOUR_EXIT, model.couplings[couple][0],
                                                         model.couplings[couple][1]))
                        else:
                            _coupling_map[state].append((Coupling.ISOLATED_EXIT, model.couplings[couple][0],
                                                         model.couplings[couple][1]))
                    else:  # ENTRY TRANSITION
                        if transition.count('*') > 0:  # NEEDS A NEIGHBOUR
                            _coupling_map[state].append((Coupling.NEIGHBOUR_ENTER, model.couplings[couple][0],
                                                         model.couplings[couple][1]))
                        else:
                            _coupling_map[state].append(Coupling.ISOLATED_ENTER, model.couplings[couple][0],
                                                        model.couplings[couple][1])
    return _coupling_map


# d(vw)/dt = (dv/dt)w + v(dw/dt)
def chain_rule(term: Term, graph, model):
    vertices = term.vertices
    all_terms = []
    for v in vertices:
        term_clone = Term(vertices)
        term_clone.vertices.remove(v)
        terms = derive(v, term_clone, graph, model)
        for term in terms:
            all_terms.append(term)
    return all_terms


def dynamically_relevant(model):
    """
    Given a model, returns list of dynamically relevant model states.
    That is, all those states that are involved in active transition
    :type model: CModel
    """
    relevant = list()
    for state in model.states:
        for coupling in model.couplings:
            if model.couplings[coupling][0].split(':')[0].count(state) > 0:
                relevant.append(state)
                break
    return relevant
