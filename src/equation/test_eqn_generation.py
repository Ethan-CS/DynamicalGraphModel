import networkx

from equation import eqn_generation
from equation.Term import Vertex, Term
import networkx as nx

from equation.eqn_generation import get_single_equations
from model_params.cmodel import CModel

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')  # Recovery rate


def test_get_single_equations():
    expected_terms = []
    for i in range(0, 50):
        expected_terms.append(str(Term([Vertex('S', i)])))
        expected_terms.append(str(Term([Vertex('I', i)])))

    actual_terms = get_single_equations(nx.erdos_renyi_graph(n=50, p=0.2), SIR)
    lhs_terms = [str(each.lhs) for each in actual_terms]
    for term in lhs_terms:
        assert term in expected_terms, f'Expected {str(term)} in actual terms, was not there'

    for term in expected_terms:
        assert term in lhs_terms, f'Found {str(term)} in actual terms, didn\'t expect to find it'


def test_path_equations():
    for i in range(3, 10):
        path = networkx.path_graph(i)
        singles_equations = eqn_generation.get_single_equations(path, SIR)
        equations = eqn_generation.generate_equations(path, SIR, prev_equations=singles_equations)
        closed_equations = eqn_generation.generate_equations(path, SIR, True, singles_equations)

        assert len(equations) == int((3 * i * i - i + 2) / 2), f'incorrect number of equations for full system for ' \
                                                               f'path on {i} vertices'
        assert len(closed_equations) == 5 * i - 3, 'incorrect number of equations for closed system for ' \
                                                   f'path on {i} vertices'
