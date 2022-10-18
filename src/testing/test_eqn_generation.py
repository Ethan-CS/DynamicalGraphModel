import networkx
import sympy as sym

from equation import generation
from equation.Term import Vertex, Term
import networkx as nx

from equation.generation import get_single_equations, generate_equations
from model_params.cmodel import CModel

tau, gamma = 1, 0.1

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', tau, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', gamma, name='gamma')  # Recovery rate

t = sym.symbols('t')

S1 = sym.Function('\u3008S0\u3009')
S2 = sym.Function('\u3008S1\u3009')
S3 = sym.Function('\u3008S2\u3009')
I1 = sym.Function('\u3008I0\u3009')
I2 = sym.Function('\u3008I1\u3009')
I3 = sym.Function('\u3008I2\u3009')

S1I2 = sym.Function('\u3008S0 I1\u3009')
I1S2 = sym.Function('\u3008I0 S1\u3009')
S1I3 = sym.Function('\u3008S0 I2\u3009')
I1S3 = sym.Function('\u3008I0 S2\u3009')
S2I3 = sym.Function('\u3008S1 I2\u3009')
I2S3 = sym.Function('\u3008I1 S2\u3009')

S1S2I3 = sym.Function('\u3008S0 S1 I2\u3009')
S1I2S3 = sym.Function('\u3008S0 I1 S2\u3009')
S1I2I3 = sym.Function('\u3008S0 I1 I2\u3009')
I1S2S3 = sym.Function('\u3008I0 S1 S2\u3009')
I1I2S3 = sym.Function('\u3008I0 I1 S2\u3009')
I1S2I3 = sym.Function('\u3008I0 S1 I2\u3009')


def test_get_single_equations():
    expected_terms = []
    size = 10
    for i in range(0, size):
        expected_terms.append(sym.Derivative(Term([Vertex('S', i)]).function()(t)))
        expected_terms.append(sym.Derivative(Term([Vertex('I', i)]).function()(t)))

    actual_terms = get_single_equations(nx.erdos_renyi_graph(n=size, p=0.2), SIR)
    lhs_terms = [each.lhs for each in actual_terms]
    for term in lhs_terms:
        assert term in expected_terms, f'Expected {str(term)} in actual terms, was not there'

    for term in expected_terms:
        assert term in lhs_terms, f'Found {str(term)} in actual terms, didn\'t expect to find it'


def test_path_equations():
    for i in range(1, 10):
        path = networkx.path_graph(i)
        equations = eqn_generation.generate_equations(path, SIR)
        closed_equations = eqn_generation.generate_equations(path, SIR, closures=True)

        assert len(equations) == int((3 * i * i - i + 2) / 2), f'incorrect number of equations for full system for ' \
                                                               f'path on {i} vertices'
        if i > 3:
            assert len(closed_equations) == 5 * i - 3, 'incorrect number of equations for closed system for ' \
                                                       f'path on {i} vertices:\n{[each.lhs for each in closed_equations]}'


def test_triangle_equations():
    triangle_equations_generated = generate_equations(networkx.cycle_graph(3), SIR)

    print(*triangle_equations_generated, sep='\n')

    assert 18 == len(triangle_equations_generated), f'There were {len(triangle_equations_generated)} equations. ' \
                                                    f'We expected 18.'

    triangle_equations_actual = [
        sym.Eq(sym.Derivative(S1(t)), -tau * S1I2(t) - tau * S1I3(t)),
        sym.Eq(sym.Derivative(S2(t)), -tau * I1S2(t) - tau * S2I3(t)),
        sym.Eq(sym.Derivative(S3(t)), -tau * I2S3(t) - tau * I1S3(t)),
        sym.Eq(sym.Derivative(I1(t)), tau * S1I2(t) + tau * S1I3(t) - gamma * I1(t)),
        sym.Eq(sym.Derivative(I2(t)), tau * I1S2(t) + tau * S2I3(t) - gamma * I2(t)),
        sym.Eq(sym.Derivative(I3(t)), tau * I2S3(t) + tau * I1S3(t) - gamma * I3(t)),

        sym.Eq(sym.Derivative(I1S2(t)), -(tau + gamma) * I1S2(t) - tau * I1S2I3(t) + tau * S1S2I3(t)),
        sym.Eq(sym.Derivative(S1I2(t)), -(tau + gamma) * S1I2(t) + tau * S1S2I3(t) - tau * S1I2I3(t)),
        sym.Eq(sym.Derivative(I2S3(t)), -(tau + gamma) * I2S3(t) + tau * I1S2S3(t) - tau * I1I2S3(t)),
        sym.Eq(sym.Derivative(S2I3(t)), -(tau + gamma) * S2I3(t) - tau * I1S2I3(t) + tau * I1S2S3(t)),
        sym.Eq(sym.Derivative(I1S3(t)), -(tau + gamma) * I1S3(t) - tau * I1I2S3(t) + tau * S1I2S3(t)),
        sym.Eq(sym.Derivative(S1I3(t)), -(tau + gamma) * S1I3(t) - tau * S1I2I3(t) + tau * S1I2S3(t)),

        sym.Eq(sym.Derivative(I1S2I3(t)), -2 * (tau + gamma) * I1S2I3(t) + tau * I1S2S3(t) + tau * S1S2I3(t)),
        sym.Eq(sym.Derivative(S1I2I3(t)), -2 * (tau + gamma) * S1I2I3(t) + tau * S1S2I3(t) + tau * S1I2S3(t)),
        sym.Eq(sym.Derivative(I1I2S3(t)), -2 * (tau + gamma) * I1I2S3(t) + tau * I1S2S3(t) + tau * S1I2S3(t)),
        sym.Eq(sym.Derivative(S1I2S3(t)), -(2 * tau + gamma) * S1I2S3(t)),
        sym.Eq(sym.Derivative(S1S2I3(t)), -(2 * tau + gamma) * S1S2I3(t)),
        sym.Eq(sym.Derivative(I1S2S3(t)), -(2 * tau + gamma) * I1S2S3(t))
    ]

    for t_actual in triangle_equations_actual:
        assert t_actual in triangle_equations_generated, 'generated equations missing this equation:' + str(t_actual)
    for t_generated in triangle_equations_generated:
        assert t_generated in triangle_equations_actual, 'generated equations has an extra equation:' + str(t_generated)
