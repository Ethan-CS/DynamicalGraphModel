import copy

import networkx
import sympy as sym
from networkx import Graph

from equation.Term import Vertex, Term
from equation.closures import can_be_closed, replace_with_closures
from model_params.cmodel import CModel
from model_params.helpers import dynamically_relevant, Coupling, coupling_types


def get_single_equations(g, model):
    """
    Given a contact graph and a compartmental model, returns the list of tuples containing only one vertex for which we
    require equations in the dynamical system. These equations are used as the base case for equation generation.

    :type g: networkx.Graph
    :type model: CModel
    :param g: A networkx graph representing the contact network in the model
    :param model: A specified compartmental model
    :return: The list of single-vertex tuples for which we require equations
    """
    graph = copy.deepcopy(g)
    singles_terms = []
    singles_equations = []
    for state in dynamically_relevant(model):
        for node in graph.nodes:
            term = Vertex(state, node)
            singles_terms.append(term)
            singles_equations.append(sym.Eq(sym.Symbol(str(term)), chain_rule(Term([term]), graph, model)))

    return singles_equations


def generate_equations(g, model, length=2, closures=False, prev_equations=None):
    """
   Generates the required equations for the model defined by the specified contact network and compartmental model.

   Parameters
   ----------
    prev_equations : list
        LHS terms mapped to tuples of terms and coefficients on the RHS of the equation - the 'base-case,' of singles,
         previously generated equations in subsequent recursive steps,
    length : int
        the length of term we are currently generating equations for (to avoid generating equations more than once)
    g : networkx.Graph
        the model contact graph.
    model : CModel
        definition of the compartmental model.
    closures : bool
        True if we are introducing moment closures on cut-vertices, false otherwise.

   Returns
   -------
   A dictionary containing the LHS terms mapped to the terms on the RHS
   """
    if prev_equations is None:
        prev_equations = get_single_equations(g, model)
    graph = copy.deepcopy(g)
    equations = prev_equations.copy()
    for eq in prev_equations:
        for term in eq.rhs.args:
            if '*' in str(term):
                term = str(term).split('*', 1)[1]
            lhs_terms = [str(each.lhs) for each in prev_equations]
            # If term is up to length we're considering
            # and not already in system, add equation for it
            if str(term) not in lhs_terms:
                if sum(c.isdigit() for c in str(term)) <= length:
                    term_as_symbol = sym.Symbol(str(term))
                    next_equation = sym.Eq(term_as_symbol, chain_rule(Term(term), graph, model, closures))
                    if next_equation not in equations:
                        equations.append(next_equation)

        # Increase length we are interested in by 1 and recur if length < num vertices in graph
    if length + 1 <= graph.number_of_nodes():
        next_equations = generate_equations(graph, model, length + 1, closures, equations)
        for eqn in next_equations:
            if eqn not in equations:
                equations.append(eqn)
    return equations


def add_terms(v: Vertex, term: Term, transition: tuple, neighbours_of_v: list):
    term_clone = copy.deepcopy(term)  # make sure we don't amend the original term
    neighbours_of_v_not_in_tuple = list(set(neighbours_of_v) - set(term_clone.node_list()))
    terms = []
    if transition[0] == Coupling.NEIGHBOUR_ENTER:
        # e.g. v is in state I, so change v to S and each neighbour in turn to I
        other_state_for_v = transition[1][0]  # the state v would be in before transitioning to current state
        for n in neighbours_of_v_not_in_tuple:
            # Make sure new term contains all same terms as original
            vertices = set(term_clone.vertices)
            vertices.add(Vertex(other_state_for_v, v.node))
            # Add the neighbour in the transition-inducing state
            vertices.add(Vertex(v.state, n))
            terms.append((sym.Symbol(str(Term(list(vertices)))), f'+{transition[2]}'))
    elif transition[0] == Coupling.NEIGHBOUR_EXIT:
        # e.g. v in state S so can exit if any neighbour in I
        other_state_for_neighbours = transition[1].split('*')[1][0]
        for n in neighbours_of_v_not_in_tuple:
            vertices = set(term_clone.vertices)
            vertices.add(v)
            vertices.add(Vertex(other_state_for_neighbours, n))
            terms.append((sym.Symbol(str(Term(list(vertices)))), f'-{transition[2]}'))
    elif transition[0] == Coupling.ISOLATED_ENTER:
        # e.g. v in state R, gets there through recovery after being in i
        other_state_for_v = transition[1][0]
        vertices = set(term_clone.vertices)
        vertices.add(Vertex(other_state_for_v, v.node))
        terms.append((sym.Symbol(str(Term(list(vertices)))), f'+{transition[2]}'))
    elif transition[0] == Coupling.ISOLATED_EXIT:
        # e.g. v in state I, transitions out without input of neighbours
        vertices = set(term_clone.vertices)
        vertices.add(v)
        terms.append((sym.Symbol(str(Term(list(vertices)))), f'-{transition[2]}'))
    else:
        print('nothing I could do!')
    return terms


def derive(v: Vertex, term_without_v: Term, g: Graph, model: CModel, closures=False):
    graph = copy.deepcopy(g)
    # Get neighbours
    neighbours_of_v = [n for n in graph.neighbors(v.node)]
    # Get mapping of states to how they are entered/exited
    transitions = coupling_types(model)
    all_terms = 0
    for transition in transitions[v.state]:
        terms = add_terms(v, term_without_v, transition, neighbours_of_v)
        for t in terms:
            if not closures:
                all_terms += float(t[1]) * sym.Symbol(str(t[0]))
            else:
                if not can_be_closed(t[0], graph):
                    all_terms += float(t[1]) * sym.Symbol(str(t[0]))
                else:
                    closure_terms = replace_with_closures(t[0], graph)
                    sub_terms = 0
                    for each_term in closure_terms:
                        sub_terms += each_term
                    all_terms += float(t[1]) * sub_terms
    return all_terms


# d(vw)/dt = (dv/dt)w + v(dw/dt)
def chain_rule(term: Term, graph: Graph, model: CModel, closures=False):
    if type(term) == Term:
        term_clone = copy.deepcopy(term)
    elif type(term) == tuple:
        term_clone = copy.deepcopy(term[0])
    else:
        term_clone = Term(str(term))

    all_terms = 0
    for v in list(term_clone.vertices):
        try:
            term_clone.remove(v)
        except ValueError:
            print(f"{str(v)} isn't in the term {term_clone}")
        terms = 0
        try:
            terms = derive(v, term_clone, copy.deepcopy(graph), model, closures)
        except networkx.NetworkXError:
            print('the graph looked like this:', graph, graph.nodes, 'and networkx could not find a vertex')
        all_terms += terms
        term_clone.add(v)
    return all_terms
