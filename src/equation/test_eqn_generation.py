from equation.Term import Vertex, Term
import networkx as nx

from equation.eqn_generation import get_single_equations
from model_params.cmodel import CModel


def test_get_single_equations():
    expected_terms = []
    for i in range(0, 100):
        expected_terms.append(Term([Vertex('S', i)]))
        expected_terms.append(Term([Vertex('I', i)]))

    SIR = CModel('SIR')
    SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
    SIR.set_coupling_rate('I:I=>R', 3, name='gamma')  # Recovery rate

    actual_terms = get_single_equations(nx.erdos_renyi_graph(n=100, p=0.2), SIR).keys()

    for term in actual_terms:
        assert term in expected_terms, f'Expected {str(term)} in actual terms, was not there'

    for term in expected_terms:
        assert term in actual_terms, f'Found {str(term)} in actual terms, didn\'t expect to find it'
