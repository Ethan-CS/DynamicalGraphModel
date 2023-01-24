import random
from time import time

import sympy as sym
from scipy.integrate import solve_ivp

from equation.Term import Vertex
from equation.generation import format_term


def initial_conditions(nodes, functions, choice=None, num_initial_infected=1, symbol=0, yes=1, no=0):
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
    LHS = []
    for list_of_eqn in full_equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())

    functions = [sym.Function(str(type(f)))(sym.symbols('t')) for f in list(LHS)]

    return solve(full_equations, graph, init_cond=init_conditions, t_max=t_max, step=5e-1, rtol=1e-1, atol=1e-2,
                 print_option='none')


def solve(full_equations, g, init_cond=None, t_max=10, step=1e-2, rtol=1e-4, atol=1e-6, print_option='none'):
    LHS, RHS = [], []
    for list_of_eqn in full_equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())
            RHS.append(each_eqn.rhs)
    if print_option == 'full':
        print(' *** Equations to solve ***')
        print(*list(zip(LHS, RHS)), sep='\n')

    def rhs(_, y_vec):
        # y_vec just contains values. We need to replace
        # terms in equations with corresponding values
        substitutions = {}  # dict with symbols as keys and corresponding y_vec at pos. as values
        j = 0
        for lhs in LHS:
            substitutions[lhs] = y_vec[j]
            j += 1
        # print(substitutions)
        # Make the substitutions and store the results in a list
        derivatives = []
        for r in list(RHS):
            r_sub = r.xreplace(substitutions)
            derivatives.append(r_sub)  # indices should be consistent over LHS, y_vec and derivatives
        return derivatives

    y0 = []
    if init_cond is None:
        init_cond = initial_conditions(list(g.nodes), list(LHS))

    for each in LHS:  # another bug found here - discrepancy between indexing in init conditions and other lists
        y0.append(init_cond[each.subs(sym.symbols('t'), 0)])

    st = time()
    y_out = solve_ivp(rhs, (0, t_max), y0, method="RK45", max_step=step, rtol=rtol, atol=atol)
    if print_option == 'full':
        print(f'solved in {time() - st}s')
    return y_out


# def scipy_solve():
#     print('setting up')
#     st = time()
#     g = nx.random_tree(4)
#     full_equations = generate_equations(g, get_SIR(), closures=False)
#     print(f'{len(full_equations)} equations generated in {time() - st}s')
#     solve(full_equations, g)
#     print('solved')


def scipy_integration_summary(info):
    s = ''
    s += '='*32 + '\n'
    for i in info.keys():
        msg = i
        if i == 'hu':
            msg = 'vector of step sizes successfully used for each time step'
        elif i == 'tcur':
            msg = 'vector with value of t reached for each time step'
        elif i == 'tolsf':
            msg = 'vector of tolerance scale factors (>1.0) ' \
                  'computed when a request for too much accuracy was detected'
        elif i == 'tsw':
            msg = 'value of t at time of the last method switch for each timestep'
        elif i == 'nst':
            msg = 'cumulative number of time steps'
        elif i == 'nfe':
            msg = 'cumulative number of function evaluations for each time step'
        elif i == 'nje':
            msg = 'cumulative number of jacobian evaluations for each time step'
        elif i == 'nqu':
            msg = 'a vector of method orders for each successful step'
        elif i == 'imxer':
            msg = 'index of the component of largest magnitude in the ' \
                  'weighted local error vector (e / ewt) on an error return, -1 ' \
                  'otherwise'
        elif i == 'lenrw':
            msg = 'the length of the double work array required'
        elif i == 'leniw':
            msg = 'the length of integer work array required'
        elif i == 'mused':
            msg = 'a vector of method indicators for each successful time step: ' \
                  '1: adams (nonstiff), 2: bdf (stiff)'
        s += f'{msg}:\n{info[i]}'
    s += '\n=' * 32
