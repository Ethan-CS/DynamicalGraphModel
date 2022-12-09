import copy
import re

import networkx
import networkx as nx
import sympy as sym
from networkx import Graph

from equation.Term import Vertex, Term
from equation.closing import can_be_closed, replace_with_closures
from model_params.cmodel import CModel
from model_params.helpers import dynamically_relevant, Coupling, coupling_types

t = sym.symbols('t')


def get_single_equations(g, model):
    """
    Returns the equations for single-vertices being in single (dynamically relevant) states.

    :type g: networkx.Graph
    :type model: CModel
    :param g: A networkx graph representing the contact network in the model
    :param model: A specified compartmental model
    :return: The list of single-vertex tuples for which we require equations
    """
    singles_equations = {}
    for i in range(g.number_of_nodes()):
        singles_equations[i+1] = []
    lhs = []
    for state in dynamically_relevant(model):
        for node in g.nodes:
            term = Term([Vertex(state, node)])
            lhs.append(sym.Function(str(term)))
            singles_equations[1].append(sym.Eq(sym.Derivative(sym.Function(str(term))(t)), chain_rule(term, g, model)))
    return singles_equations, lhs


def generate_equations(g, model, length=2, closures=False, prev_equations=None, lhs_terms=None):
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
    lhs_terms : list
        All terms for which we already have equations.

   Returns
   -------
   A dictionary containing the LHS terms mapped to the terms on the RHS
   """
    # If there are no cut vertices, set closures to false as no terms can be closed.
    closures = closures and len(list(nx.articulation_points(g))) > 0
    # If no prev equations provided, get the base case (singles equations)
    if prev_equations is None:
        prev_equations, lhs_terms = get_single_equations(g, model)
    # Get a list of the terms on LHS of previous equations
    if lhs_terms is None:
        lhs_terms = [sym.Integral(each.lhs).doit() for each in set().union(*prev_equations.values())]
    # Start with previous equations as base case and add to that list
    equations = prev_equations
    # Look through RHS terms of previous equations and add eqn for any terms that don't have one yet
    terms = get_rhs_terms(equations, min(length, g.number_of_nodes()), lhs_terms)
    get_specified_equations(terms, equations, lhs_terms, g, model, closures)
    # Increase length we are interested in by 1 and recur if length < num vertices in graph
    if length <= g.number_of_nodes() and len(equations[length]) > 0:
        generate_equations(g, model, length+1, closures, equations, lhs_terms)

    return equations


def get_specified_equations(terms, equations, lhs_terms, g, model, closures):
    for term in terms:
        term = Term(term)
        # If term is up to length we're considering and not already in system, add equation for it
        new_term = term.function()
        t = sym.symbols('t')
        if new_term not in lhs_terms:
            if not closures or (closures and not can_be_closed(term, g)):
                eq_rhs = chain_rule(term, g, model, closures)
                next_equation = sym.Eq(sym.Derivative(term.function()(t)), eq_rhs)
                equations[len(str(term).split(" "))].append(next_equation)
                lhs_terms.append(new_term)
        elif can_be_closed(term, g):
            closure_terms = replace_with_closures(term.function(), g)
            for sub_term in closure_terms:
                if '/' in str(sub_term):
                    sub_term = Term(str(1/sub_term))
                if type(sub_term) is not Term:
                    sub_term = Term(sub_term)
                fn = sub_term.function()
                if fn not in lhs_terms and fn(t) not in lhs_terms:
                    next_from_closures = sym.Eq(sym.Derivative(fn(t)), chain_rule(sub_term, g, model, closures))
                    equations[len(str(sub_term).split(" "))].append(next_from_closures)
                    lhs_terms.append(fn)


def get_rhs_terms(equations, length, lhs):
    if length > 1:
        rhs_expressions = [each.rhs for each in set().union(equations[length], equations[length-1])]
    else:
        rhs_expressions = [each.rhs for each in equations[length]]
    rhs = set()
    for expr in rhs_expressions:
        rhs.update([f for f in expr.atoms(sym.Function) if f not in lhs])
    return list(rhs)


def get_eq_lhs(eq):
    return str(eq.lhs).replace('Derivative', '').replace('(', '').replace(')', '').replace('\u3008', '') \
        .replace('\u3009', '').replace('t', '').replace(',', '')


def format_term(formatted_term):
    formatted_term = re.sub("[-.+\u3008\u3009(t)]", "", str(formatted_term))
    if '*' in str(formatted_term):
        formatted_term = list(str(formatted_term).split('*', 1))[1]
    # Replace / in closures with *, as we deal with them the same later
    if '/' in str(formatted_term):
        formatted_term = str(formatted_term).replace('/', '*')
    return formatted_term


def add_terms(v: Vertex, term: Term, transition: tuple, model: CModel, neighbours_of_v: list):
    term_clone = copy.deepcopy(term)  # make sure we don't amend the original term
    neighbours_of_v_not_in_tuple = list(set(neighbours_of_v) - set(term_clone.node_list()))
    neighbours_of_v_in_tuple = list(set(term_clone.vertices) - {v})
    terms = []
    if transition[0] == Coupling.NEIGHBOUR_ENTER:
        # e.g. v is in state I, so change v to S and each neighbour in turn to I
        other_state_for_v = transition[1][0]  # the state v would be in before transitioning to current state
        # First, look at all external vertices that could lead to entering this state
        for n in neighbours_of_v_not_in_tuple:
            # Make sure new term contains all same terms as original
            vertices = set(term_clone.vertices)
            vertices.add(Vertex(other_state_for_v, v.node))
            # Add the neighbour in the transition-inducing state
            vertices.add(Vertex(v.state, n))
            terms.append(transition[2] * Term(list(vertices)).function()(sym.symbols('t')))
        # Now, try to find any neighbours in the tuple that could lead to this state
        for n in neighbours_of_v_in_tuple:
            if coupling_types(model)[n.state][0][0] == Coupling.NEIGHBOUR_ENTER:
                vertices = set(term_clone.vertices)
                vertices.update({Vertex(other_state_for_v, v.node), n})
                terms.append(coupling_types(model)[n.state][0][2] * Term(list(vertices)).function()(sym.symbols('t')))
    elif transition[0] == Coupling.NEIGHBOUR_EXIT:
        # e.g. v in state S so can exit if any neighbour in I
        other_state_for_neighbours = transition[1].split('*')[1][0]
        for n in neighbours_of_v_not_in_tuple:
            vertices = set(term_clone.vertices)
            vertices.add(v)
            vertices.add(Vertex(other_state_for_neighbours, n))
            terms.append(-transition[2] * Term(list(vertices)).function()(sym.symbols('t')))
        for n in neighbours_of_v_in_tuple:
            if coupling_types(model)[n.state][0][0] == Coupling.NEIGHBOUR_ENTER:
                vertices = set(term_clone.vertices)
                vertices.update({n, v})
                terms.append(-coupling_types(model)[n.state][0][2] * Term(list(vertices)).function()(sym.symbols('t')))
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
    all_expressions = 0
    all_terms = []
    for transition in transitions[v.state]:
        terms = add_terms(v, term_without_v, transition, model, neighbours_of_v)
        for each_term in terms:
            if not closures:
                all_expressions += each_term
                all_terms.append(each_term)
            else:
                # Need to only consider the term, not any coefficients or dependencies
                term_as_string = str(each_term)
                if '*' in term_as_string:
                    term_as_string = term_as_string.split('*', 1)[1]
                cleaned = re.sub("[-.+*\u3008\u3009(t)]", "", term_as_string)
                vertices = [Vertex(v[0], int(v[1:])) for v in cleaned.split()]
                actual_term = Term(vertices)
                if not can_be_closed(actual_term, graph):
                    all_expressions += each_term
                    all_terms.append(each_term)
                else:
                    closure_terms = replace_with_closures(each_term, graph)
                    sub_terms = 1.0
                    for each_closure_term in closure_terms:
                        sub_terms *= each_closure_term
                    all_expressions += sub_terms
                    all_terms.append(sub_terms)
    return all_expressions, all_terms


# d(vw)/dt = (dv/dt)w + v(dw/dt)
def chain_rule(term: Term, graph: Graph, model: CModel, closures=False):
    term_clone = copy.deepcopy(term)

    expression = 0
    for v in list(term_clone.vertices):
        term_clone.remove(v)
        expr, atoms = derive(v, term_clone, graph, model, closures)
        expression += expr
        term_clone.add(v)
    return expression
