import copy
import re

import networkx
import sympy as sym
from networkx import Graph

from equation.Term import Vertex, Term
from equation.closures import can_be_closed, replace_with_closures
from model_params.cmodel import CModel
from model_params.helpers import dynamically_relevant, Coupling, coupling_types

t = sym.symbols('t')


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
            singles_equations.append(
                sym.Eq(sym.Derivative(sym.Function(str(term))(t)), chain_rule(Term([term]), graph, model)))

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
        rhs_terms = list(eq.rhs.args)
        new_terms = list()
        # Pre-formatting - get rid of coefficients and separate closure terms into individual terms
        for term in rhs_terms:
            try:
                float(term)
                continue
            except TypeError:
                pass
            formatted_term = copy.deepcopy(term)
            formatted_term = str(formatted_term).replace('(t)', '').replace('-', '') \
                .replace('〈', '').replace('〉', '').replace(' ', '')
            formatted_term = format_term(formatted_term)

            # If term still contains *, means it's a closure term and needs splitting up into individual terms
            if '*' in str(formatted_term):
                terms = str(formatted_term).split('*')
                for each_term in terms:
                    each_term = each_term.replace('*', '')
                    if each_term not in new_terms:
                        new_terms.append(Term(each_term))
            # If there are no *s remaining, must be a single term, so add as usual (if not numeric)
            elif formatted_term not in new_terms:
                new_terms.append(Term(formatted_term))

        rhs_terms = new_terms
        for term in rhs_terms:
            lhs_terms = [each.lhs for each in prev_equations]
            # If term is up to length we're considering
            # and not already in system, add equation for it
            if term not in lhs_terms and str(term) != '\u3009':
                if sum(c.isalpha() for c in str(term)) <= length:
                    term_as_function = term.function()
                    if not closures:
                        next_equation = sym.Eq(sym.Derivative(term_as_function(sym.symbols('t'))),
                                               chain_rule(term, graph, model, closures))
                        if next_equation not in equations:
                            equations.append(next_equation)
                    else:
                        if not can_be_closed(term, graph):
                            next_equation = sym.Eq(sym.Derivative(term_as_function(sym.symbols('t'))),
                                                   chain_rule(term, graph, model, closures))
                            if next_equation not in equations:
                                equations.append(next_equation)
                        else:
                            closure_terms = replace_with_closures(term_as_function, graph)
                            for small_term in closure_terms:
                                if small_term not in lhs_terms:
                                    next_from_closures = sym.Eq(
                                        sym.Function(format_term(str(small_term)))(sym.symbols('t')),
                                        chain_rule(Term(format_term(str(small_term))), graph, model, closures))
                                    if next_from_closures not in equations:
                                        equations.append(next_from_closures)

        # Increase length we are interested in by 1 and recur if length < num vertices in graph
    if length + 1 <= graph.number_of_nodes():
        next_equations = generate_equations(graph, model, length + 1, closures, equations)
        for eqn in next_equations:
            if eqn not in equations:
                equations.append(eqn)
    return equations


def format_term(formatted_term):
    if '*' in str(formatted_term):
        formatted_term = list(str(formatted_term).split('*', 1))[1]
    # Replace / in closures with *, as we deal with them the same later
    if '/' in str(formatted_term):
        formatted_term = str(formatted_term).replace('/', '*')
    return formatted_term


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
            terms.append(transition[2] * Term(list(vertices)).function()(sym.symbols('t')))
    elif transition[0] == Coupling.NEIGHBOUR_EXIT:
        # e.g. v in state S so can exit if any neighbour in I
        other_state_for_neighbours = transition[1].split('*')[1][0]
        for n in neighbours_of_v_not_in_tuple:
            vertices = set(term_clone.vertices)
            vertices.add(v)
            vertices.add(Vertex(other_state_for_neighbours, n))
            terms.append(-transition[2] * Term(list(vertices)).function()(sym.symbols('t')))
    elif transition[0] == Coupling.ISOLATED_ENTER:
        # e.g. v in state R, gets there through recovery after being in i
        other_state_for_v = transition[1][0]
        vertices = set(term_clone.vertices)
        vertices.add(Vertex(other_state_for_v, v.node))
        terms.append(transition[2] * Term(list(vertices)).function()(sym.symbols('t')))
    elif transition[0] == Coupling.ISOLATED_EXIT:
        # e.g. v in state I, transitions out without input of neighbours
        vertices = set(term_clone.vertices)
        vertices.add(v)
        terms.append(-transition[2] * Term(list(vertices)).function()(sym.symbols('t')))
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
        for each_term in terms:
            if not closures:
                all_terms += each_term
            else:
                # Need to only consider the term, not any coefficients or dependencies
                term_as_string = str(each_term)
                if '*' in term_as_string:
                    term_as_string = term_as_string.split('*', 1)[1]
                cleaned = re.sub("[-.+*\u3008\u3009(t)]", "", term_as_string)
                vertices = [Vertex(v[0], int(v[1:])) for v in cleaned.split()]
                actual_term = Term(vertices)
                if not can_be_closed(actual_term, graph):
                    all_terms += each_term
                else:
                    closure_terms = replace_with_closures(each_term, graph)
                    sub_terms = 1.0
                    for each_closure_term in closure_terms:
                        sub_terms *= each_closure_term
                    all_terms += sub_terms
    return all_terms


# d(vw)/dt = (dv/dt)w + v(dw/dt)
def chain_rule(term: Term, graph: Graph, model: CModel, closures=False):
    term_clone = copy.deepcopy(term)

    all_terms = 0
    for v in list(term_clone.vertices):
        term_clone.remove(v)
        terms = derive(v, term_clone, copy.deepcopy(graph), model, closures)
        all_terms += terms
        term_clone.add(v)
    return all_terms
