from time import time

import networkx as nx
import sympy as sym
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from equation import generate_equations, get_SIR, Term, Vertex, solve, initial_conditions

matplotlib.use('TkAgg')

sns.set()
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")

t = sym.symbols('t')
beta, gamma = 0.7, 0.1
# Initial conditions
no = 0.0
yes = 1.0

graph = nx.cycle_graph(3)
# 0 is infected, others susceptible

full_equations = generate_equations(graph, get_SIR(beta, gamma))
equations = []
for list_of_eqn in full_equations.values():
    for eqn in list_of_eqn:
        print(eqn)
        equations.append(eqn)
t_max = 10

LHS = []
for list_of_eqn in full_equations.values():
    for each_eqn in list_of_eqn:
        LHS.append(sym.Integral(each_eqn.lhs).doit())

IV = initial_conditions(graph.nodes, LHS, [0])
print(IV)
print(len(IV))


def get_numerical_sol_from_generated():
    plt.clf()
    print('\npassing equations and ICs into numerical solver...')
    sol = solve(full_equations, graph, init_cond=IV, t_max=t_max, step=1e-1, atol=1e-8, rtol=1e-9,
                print_option='full')
    print(sol['message'])
    final_vals = []
    for i in sol.y:
        final_vals.append(i[-1])
    print([round(y, 3) for y in final_vals])

    plot_numerical(sol)


def plot_numerical(sol):
    plt.clf()
    plt.plot(sol.t, sol.y[0],
             label=f'${sym.Integral(equations[0].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009', ']'))
    for i in range(1, len(equations)):
        plt.plot(sol.t, sol.y[i],
                 label=f'${sym.Integral(equations[i].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009', ']'))
    plt.title("Numerical solution to $C_5$ system")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.axis([0, t_max, 0, 1])
    # plt.ylim([0, 1])
    plt.size = (10, 8)
    plt.tight_layout()
    plt.savefig('test-num.png')
    plt.show()
    plt.clf()


# Part 1 of plan to get ode solver working on generated equations:
# Get the solver working with hard-coded equation. If can’t, do a
# careful comparison of approach versus original examples to see why.
def get_analytic_sol_from_generated():
    plt.clf()
    print(f'\npassing {len(equations)} equations and ICs into analytic solver...')
    start = time()
    sol = sym.solvers.ode.systems.dsolve_system(eqs=equations, funcs=LHS, t=t, ics=IV)[0]
    print(f'solved in {time() - start}s!')
    print(sol)
    at_ten = []
    for eq in sol:
        at_ten.append(eq.subs(t, t_max))
    print(at_ten)
    print('analytic solution:\n', [round(y.rhs, 3) for y in at_ten])
    plot_analytic(sol)


def plot_analytic(sol):
    t_range = (t, 0, 10)
    p = sym.plot(sol[0].rhs, t_range,
                 label=f'${sym.Integral(equations[0].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009', ']'),
                 legend=True, show=False)
    for i in range(1, len(equations)):
        p.extend(sym.plot(sol[i].rhs, t_range,
                          label=f'${sym.Integral(equations[i].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009',
                                                                                                            ']'),
                          legend=True, show=False))
    p.title = "Analytic solution to $C_5$ system"
    p.x_label = "Time"
    p.y_label = "Probability"
    p.size = (10, 8)
    p.save('test-exact.png')
    p.show()
    plt.axis([0, t_max, 0, 1])
    plt.tight_layout()
    plt.clf()


get_analytic_sol_from_generated()
get_numerical_sol_from_generated()
