from time import time

import matplotlib

import networkx as nx
import sympy as sym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import odeint

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
    print(f'{len(full_equations)} equations generated in {time() - st}s - printing...')
    LHS = [sym.Integral(each.lhs).doit() for each in list(full_equations)]
    RHS = [each.rhs for each in list(full_equations)]
    for i in range(len(LHS)):
        print(f'{LHS[i]}/dt = {RHS[i]}')

    def rhs(y_vec, _):
        derivatives = []
        substitutions = {}
        j = 0
        for lhs in list(LHS):
            substitutions[lhs] = y_vec[j]
            j += 1
        for r in list(RHS):
            derivatives.append(r.xreplace(substitutions))

        return derivatives

    st = time()
    y0 = list(initial_conditions(list(g.nodes), list(LHS), symbol=sym.symbols('t')).values())
    print(f'got initial conditions in {time()-st}s')
    t = np.arange(start=0, stop=2, step=0.01)
    st = time()
    y_out, info = odeint(rhs, y0, t, rtol=0.00001, atol=0.000001, full_output=True)
    print(f'done in {time()-st}s!')
    print_integration_summary(info)

    for i in range(len(LHS)):
        print(f'{LHS[i]}/dt = {RHS[i]}')

    plt.plot(t, y_out)
    plt.xlabel('t')
    plt.ylabel('soln')
    plt.savefig('test.png')
    plt.show()


def print_integration_summary(info):
    print('='*32)
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
        print(f'{msg}:\n{info[i]}')
    print('=' * 32)


scipy_solve()
