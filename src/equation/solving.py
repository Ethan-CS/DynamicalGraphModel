import random
from time import time

import sympy as sym
from scipy.integrate import solve_ivp

from equation.Term import Term, Vertex
from equation.closing import can_be_closed, replace_with_closures
from equation.generation import format_term


def initial_conditions(nodes, functions, choice=None, num_initial_infected=1, symbol=0, yes=1, no=0):
    """
    Defines initial conditions as specified by the solver used (LSODA).

    :param nodes: list of vertices in the graph
    :param functions: list of LHS of equations in the system
    :param choice: specified choice of initial conditions, which are randomised if unspecified.
    :param num_initial_infected: number of initial infected vertices, defaults to one.
    :param symbol: symbol used to denote time (such as `t`) for printing, does not print if left as 0.
    :param yes: probability with which a vertex is initially infected (1=certain yes, 0=certain no).
    :param no: probability with which a vertex is initially susceptible.
    :return: set of initial conditions used for solving system of equations.
    """
    initial_values = {}
    for node in list(nodes):
        initial_values[sym.Function(str(Vertex('S', node)))(symbol)] = yes
        initial_values[sym.Function(str(Vertex('I', node)))(symbol)] = no
    if choice is not None and type(choice) is list:
        num_initial_infected = len(choice)

    for i in range(num_initial_infected):
        if choice is None:
            initial_infected = random.choice(nodes)
        else:
            initial_infected = choice[i]
        initial_values[sym.Function(str(Vertex('S', initial_infected)))(symbol)] = no
        initial_values[sym.Function(str(Vertex('I', initial_infected)))(symbol)] = yes

    for f in list(functions):
        f = f.subs(sym.symbols('t'), symbol)
        formatted = format_term(str(f).split('\u3009')[0])
        split = formatted.split(" ")
        split = [x for x in split if x != '']
        if len(split) > 1:
            formatted = sym.Function(str('\u3008' + split[0] + '\u3009'))(symbol)
            initial_values[f] = initial_values[formatted]
            for i in range(1, len(split)):
                initial_values[f] *=\
                    initial_values[sym.Function(str('\u3008' + split[i] + '\u3009'))(symbol)]

    return initial_values


def solve_equations(full_equations, init_conditions, graph, t_max):
    """
    Solves the system of equations for the initial conditions given.

    :param full_equations: system of equations.
    :param init_conditions: initial conditions for which to solve the equations.
    :param graph: underlying graph in the model.
    :param t_max: maximum timestep for which the solver should solve.
    :return: the solution of the system of equations for these initial conditions.
    """
    LHS = []
    for list_of_eqn in full_equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())

    return solve(full_equations, graph, init_cond=init_conditions, t_max=t_max, step=5e-1, rtol=1e-1, atol=1e-2,
                 print_option='none')


MIN_STATE_VALUE = 1e-9
TIME_SYMBOL = sym.symbols('t')


def _coerce_numeric_symbols(expr):
    replacements = {}
    for symbol in expr.atoms(sym.Symbol):
        try:
            replacements[symbol] = float(str(symbol))
        except ValueError:
            continue
    return expr.xreplace(replacements) if replacements else expr


def _stabilise_value(value: float) -> float:
    if abs(value) < MIN_STATE_VALUE:
        return MIN_STATE_VALUE if value >= 0 else -MIN_STATE_VALUE
    return value


def solve(full_equations, g, init_cond=None, t_max=10, step=1e-2, rtol=1e-2, atol=1e-4, print_option='none'):
    # Solves a system of equations given parameters (equations, tolerances, max timestep, initial conditions)
    LHS, RHS = [], []
    for list_of_eqn in full_equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())
            RHS.append(each_eqn.rhs)
    if print_option == 'full':
        print(' *** Equations to solve ***')
        print(*list(zip(LHS, RHS)), sep='\n')

    closure_cache = {}

    def rhs(_, y_vec):
        # y_vec just contains values. We need to replace
        # terms in equations with corresponding values
        substitutions = {}  # dict with symbols as keys and corresponding y_vec at pos. as values
        j = 0
        for lhs in LHS:
            substitutions[lhs] = _stabilise_value(float(y_vec[j]))
            j += 1

        def _resolve_function(func):
            if func in substitutions:
                return substitutions[func]
            if func in closure_cache:
                return closure_cache[func]
            term = Term(func)
            if not can_be_closed(term, g):
                # Fall back to mean-field factorisation when closures are unavailable.
                estimate = 1.0
                for vertex in term.vertices:
                    single_term = Term([vertex])
                    single_func = single_term.function()(TIME_SYMBOL)
                    if single_func not in substitutions:
                        raise KeyError(f"No equation available for base term {single_term}")
                    estimate *= substitutions[single_func]
                closure_cache[func] = estimate
                return estimate
            product = sym.Integer(1)
            for component in replace_with_closures(term, g):
                product *= component
            value = _ensure_numeric(product)
            closure_cache[func] = value
            return value

        def _ensure_numeric(expr):
            expr_sub = expr
            for func in expr_sub.atoms(sym.Function):
                if func not in substitutions:
                    substitutions[func] = _resolve_function(func)
            expr_sub = expr_sub.xreplace(substitutions)
            expr_sub = _coerce_numeric_symbols(expr_sub)
            try:
                return float(expr_sub)
            except TypeError as exc:  # pragma: no cover - diagnostic path
                symbols = expr_sub.atoms(sym.Symbol)
                raise TypeError(
                    f"Cannot convert '{expr_sub}' to float; residual symbols: {symbols}"
                ) from exc

        def _augment_substitutions(expr):
            for func in expr.atoms(sym.Function):
                if func not in substitutions:
                    substitutions[func] = _resolve_function(func)
        # print(substitutions)
        # Make the substitutions and store the results in a list
        derivatives = []
        for r in list(RHS):
            _augment_substitutions(r)
            try:
                derivatives.append(_ensure_numeric(r))
            except Exception as exc:  # pylint: disable=broad-except
                raise TypeError(
                    f"Unable to convert expression '{r}' after substitutions to float ({type(exc).__name__}: {exc})."
                ) from exc
        return derivatives

    y0 = []
    if init_cond is None:
        init_cond = initial_conditions(list(g.nodes), list(LHS))

    for each in LHS:  # another bug found here - discrepancy between indexing in init conditions and other lists
        y0.append(init_cond[each.subs(sym.symbols('t'), 0)])

    st = time()
    y_out = solve_ivp(rhs, (0, t_max), y0, method="LSODA", max_step=step, rtol=rtol, atol=atol)
    if print_option == 'full':
        print(f'solved in {time() - st}s')
    return y_out

