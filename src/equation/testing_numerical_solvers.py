from time import time

import matplotlib

import networkx as nx
import sympy as sym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import odeint, solve_ivp

from equation import generate_equations, initial_conditions
from model_params.cmodel import get_SIR

# Set seaborn as default and set resolution and style defaults
sns.set()
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")

matplotlib.use('TkAgg')  # Avoids an annoying error on macOS


def scipy_solve():
    print('setting up')
    st = time()
    g = nx.random_tree(10)
    full_equations = generate_equations(g, get_SIR(), closures=False)
    print(f'{len(full_equations)} equations generated in {time() - st}s')
    solve(full_equations, g)


def solve(full_equations, g, init_cond=None, beta=0.75, t_max=10, step=0.1, rtol=0.1, atol=0.1, print_option='none'):
    LHS, RHS = [], []
    for list_of_eqn in full_equations.values():
        for each_eqn in list_of_eqn:
            LHS.append(sym.Integral(each_eqn.lhs).doit())
            RHS.append(each_eqn.rhs)
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

    if init_cond is None:
        # st = time()
        y0 = list(initial_conditions(list(g.nodes), list(LHS), symbol=sym.symbols('t')).values())
        # print(f'got initial conditions in {time() - st}s')
    else:
        y0 = list(init_cond.values())

    st = time()
    y_out = solve_ivp(rhs, (0, t_max), y0, method="Radau", max_step=step, rtol=rtol, atol=atol)
    if print_option == 'full':
        print(f'solved in {time() - st}s')
    return y_out


def integration_summary(info):
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
