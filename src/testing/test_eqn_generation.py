import networkx
import sympy as sym

from equation import generation
from equation.Term import Vertex, Term
import networkx as nx

from equation.generation import get_single_equations, generate_equations
from equation.solving import initial_conditions, solve_equations
from model_params.cmodel import get_SIR

tau, gamma = 1, 0.1

SIR = get_SIR(tau, gamma)

t = sym.symbols('t')

S1 = sym.Function('\u3008S_0\u3009')
S2 = sym.Function('\u3008S_1\u3009')
S3 = sym.Function('\u3008S_2\u3009')
I1 = sym.Function('\u3008I_0\u3009')
I2 = sym.Function('\u3008I_1\u3009')
I3 = sym.Function('\u3008I_2\u3009')

S1I2 = sym.Function('\u3008S_0 I_1\u3009')
I1S2 = sym.Function('\u3008I_0 S_1\u3009')
S1I3 = sym.Function('\u3008S_0 I_2\u3009')
I1S3 = sym.Function('\u3008I_0 S_2\u3009')
S2I3 = sym.Function('\u3008S_1 I_2\u3009')
I2S3 = sym.Function('\u3008I_1 S_2\u3009')

S1S2I3 = sym.Function('\u3008S_0 S_1 I_2\u3009')
S1I2S3 = sym.Function('\u3008S_0 I_1 S_2\u3009')
S1I2I3 = sym.Function('\u3008S_0 I_1 I_2\u3009')
I1S2S3 = sym.Function('\u3008I_0 S_1 S_2\u3009')
I1I2S3 = sym.Function('\u3008I_0 I_1 S_2\u3009')
I1S2I3 = sym.Function('\u3008I_0 S_1 I_2\u3009')


def flatten_equations(equations):
    return [eq for eq_list in equations.values() for eq in eq_list]


def test_get_single_equations():
    expected_terms = []
    size = 10
    for i in range(0, size):
        expected_terms.append(Term([Vertex('S', i)]).function())
        expected_terms.append(Term([Vertex('I', i)]).function())

    actual_terms, lhs_terms = get_single_equations(nx.erdos_renyi_graph(n=size, p=0.2), SIR)
    for term in lhs_terms:
        assert term in expected_terms, f'Expected {str(term)} in actual terms, was not there'

    for term in expected_terms:
        assert term in lhs_terms, f'Found {str(term)} in actual terms, didn\'t expect to find it'


def test_path_equations():
    for i in range(1, 10):
        path = networkx.path_graph(i)
        equations = generation.generate_equations(path, SIR)
        all_equations = []
        for each_len in equations:
            for eqn in equations[each_len]:
                all_equations.append(eqn)
        print(*all_equations, sep='\n')
        closed_equations = generation.generate_equations(path, SIR, closures=True)
        closed_equation_count = sum(len(eqn_list) for eqn_list in closed_equations.values())

        assert len(all_equations) == int((3 * i * i - i + 2) / 2), \
            f'incorrect number of equations for full system for path on {i} vertices.\nGot {len(all_equations)}, ' \
            f'expected {int((3 * i * i - i + 2) / 2)}.'
        if i > 3:
            assert closed_equation_count == 5 * i - 3, 'incorrect number of equations for closed system for ' \
                                                       f'path on {i}\n{[sym.Integral(eq.lhs).doit() for sublist in closed_equations.values() for eq in sublist]}'


def test_triangle_equations():
    triangle_equations_generated = generate_equations(networkx.cycle_graph(3), SIR)
    triangle_equations_list = [eqn for eq_list in triangle_equations_generated.values() for eqn in eq_list]

    print(*triangle_equations_list, sep='\n')

    assert 18 == len(triangle_equations_list), f'There were {len(triangle_equations_list)} equations. ' \
                                               f'We expected 18.'

    lhs_terms = {Term(sym.Integral(eq.lhs).doit()) for eq in triangle_equations_list}
    assert len(lhs_terms) == len(triangle_equations_list)

    expected_terms = {
        Term([Vertex('S', 0)]),
        Term([Vertex('S', 1)]),
        Term([Vertex('S', 2)]),
        Term([Vertex('I', 0)]),
        Term([Vertex('I', 1)]),
        Term([Vertex('I', 2)]),
        Term([Vertex('S', 0), Vertex('I', 1)]),
        Term([Vertex('S', 0), Vertex('I', 2)]),
        Term([Vertex('S', 1), Vertex('I', 2)]),
        Term([Vertex('I', 0), Vertex('S', 1)]),
        Term([Vertex('I', 0), Vertex('S', 2)]),
        Term([Vertex('I', 1), Vertex('S', 2)]),
        Term([Vertex('S', 0), Vertex('S', 1), Vertex('I', 2)]),
        Term([Vertex('S', 0), Vertex('I', 1), Vertex('S', 2)]),
        Term([Vertex('S', 0), Vertex('I', 1), Vertex('I', 2)]),
        Term([Vertex('I', 0), Vertex('S', 1), Vertex('S', 2)]),
        Term([Vertex('I', 0), Vertex('I', 1), Vertex('S', 2)]),
        Term([Vertex('I', 0), Vertex('S', 1), Vertex('I', 2)])
    }

    assert lhs_terms == expected_terms, 'triangle graph should yield all singles, pairs, and triples'


def test_term_cap_limits_equations():
    graph = nx.path_graph(4)
    capped_equations = generate_equations(graph, SIR, term_cap=2)

    lhs_terms = []
    for eq_list in capped_equations.values():
        for eqn in eq_list:
            lhs_terms.append(Term(sym.Integral(eqn.lhs).doit()))

    assert lhs_terms, 'Capped system should include at least single-vertex equations'
    assert all(len(term.vertices) <= 2 for term in lhs_terms), 'No equation should exceed the specified cap'
    assert any(len(term.vertices) == 2 for term in lhs_terms), 'Pair terms should remain when cap >= 2'

    triple_term = Term([Vertex('S', 0), Vertex('I', 1), Vertex('I', 2)])
    assert all(term != triple_term for term in lhs_terms), 'Triple terms must not receive equations when capped at 2'

    rhs_contains_higher_order = False
    for eq_list in capped_equations.values():
        for eqn in eq_list:
            for func in eqn.rhs.atoms(sym.Function):
                try:
                    rhs_term = Term(func)
                except AssertionError:
                    continue
                if len(rhs_term.vertices) >= 3:
                    rhs_contains_higher_order = True
                    break
            if rhs_contains_higher_order:
                break
        if rhs_contains_higher_order:
            break

    assert rhs_contains_higher_order, 'Higher-order terms should still appear on the RHS for approximation fidelity'


def test_full_system_matches_reference_path_two():
    graph = nx.path_graph(2)
    generated = {str(eq) for eq in flatten_equations(generate_equations(graph, SIR, closures=False))}

    expected = {
        'Eq(Derivative(〈S_0〉(t), t), -〈S_0 I_1〉(t))',
        'Eq(Derivative(〈S_1〉(t), t), -〈I_0 S_1〉(t))',
        'Eq(Derivative(〈I_0〉(t), t), -0.1*〈I_0〉(t) + 〈S_0 I_1〉(t))',
        'Eq(Derivative(〈I_1〉(t), t), 〈I_0 S_1〉(t) - 0.1*〈I_1〉(t))',
        'Eq(Derivative(〈S_0 I_1〉(t), t), -1.1*〈S_0 I_1〉(t))',
        'Eq(Derivative(〈I_0 S_1〉(t), t), -1.1*〈I_0 S_1〉(t))',
    }

    assert generated == expected


def test_closed_system_matches_reference_path_three():
    graph = nx.path_graph(3)
    generated = {str(eq) for eq in flatten_equations(generate_equations(graph, SIR, closures=True))}

    expected = {
        'Eq(Derivative(〈S_0〉(t), t), -〈S_0 I_1〉(t))',
        'Eq(Derivative(〈S_1〉(t), t), -〈I_0 S_1〉(t) - 〈S_1 I_2〉(t))',
        'Eq(Derivative(〈S_2〉(t), t), -〈I_1 S_2〉(t))',
        'Eq(Derivative(〈I_0〉(t), t), -0.1*〈I_0〉(t) + 〈S_0 I_1〉(t))',
        'Eq(Derivative(〈I_1〉(t), t), 〈I_0 S_1〉(t) - 0.1*〈I_1〉(t) + 〈S_1 I_2〉(t))',
        'Eq(Derivative(〈I_2〉(t), t), 〈I_1 S_2〉(t) - 0.1*〈I_2〉(t))',
        'Eq(Derivative(〈I_1 S_2〉(t), t), 〈I_0 S_1〉(t)*〈S_1 S_2〉(t)/〈S_1〉(t) - 1.1*〈I_1 S_2〉(t))',
        'Eq(Derivative(〈S_0 I_1〉(t), t), -1.1*〈S_0 I_1〉(t) + 〈S_0 S_1〉(t)*〈S_1 I_2〉(t)/〈S_1〉(t))',
        'Eq(Derivative(〈I_0 S_1〉(t), t), 〈I_0 S_1〉(t)*〈S_1 I_2〉(t)/〈S_1〉(t) - 1.1*〈I_0 S_1〉(t))',
        'Eq(Derivative(〈S_1 I_2〉(t), t), 〈I_0 S_1〉(t)*〈S_1 I_2〉(t)/〈S_1〉(t) - 1.1*〈S_1 I_2〉(t))',
        'Eq(Derivative(〈S_0 S_1〉(t), t), 〈S_0 S_1〉(t)*〈S_1 I_2〉(t)/〈S_1〉(t))',
        'Eq(Derivative(〈S_1 S_2〉(t), t), 〈I_0 S_1〉(t)*〈S_1 S_2〉(t)/〈S_1〉(t))',
    }

    assert generated == expected


def test_cycle_graph_has_no_closure_difference():
    graph = nx.cycle_graph(4)
    full_eq = flatten_equations(generate_equations(graph, SIR, closures=False))
    closed_eq = flatten_equations(generate_equations(graph, SIR, closures=True))

    assert {sym.srepr(eq) for eq in full_eq} == {sym.srepr(eq) for eq in closed_eq}


def test_generated_equations_can_be_solved():
    graph = nx.path_graph(3)
    equations = generate_equations(graph, SIR, closures=False)
    lhs_functions = [sym.Integral(eq.lhs).doit() for eq in flatten_equations(equations)]
    init_cond = initial_conditions(
        list(graph.nodes),
        lhs_functions,
        choice=[0],
        num_initial_infected=1,
        symbol=0,
        yes=1,
        no=0,
    )

    solution = solve_equations(equations, init_cond, graph, t_max=2)

    assert solution.success, 'ODE solver should converge on the generated system'
    assert solution.t[-1] >= 1.9, 'Solution should be evaluated up to requested horizon'
    assert solution.y.shape[0] == len(lhs_functions), 'One trajectory per equation is expected'


def run_all():
    test_get_single_equations()
    test_path_equations()
    test_triangle_equations()
    test_term_cap_limits_equations()
    test_full_system_matches_reference_path_two()
    test_closed_system_matches_reference_path_three()
    test_cycle_graph_has_no_closure_difference()
    test_generated_equations_can_be_solved()


if __name__ == '__main__':
    run_all()
