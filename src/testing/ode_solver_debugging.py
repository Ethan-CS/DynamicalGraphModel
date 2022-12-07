from time import time

import networkx as nx
import sympy as sym
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from equation import generate_equations, get_SIR, Term, Vertex, solve

matplotlib.use('TkAgg')

sns.set()
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")

t = sym.symbols('t')

# Variables
v_s1, v_s2, v_s3 = Vertex('S', 0), Vertex('S', 1), Vertex('S', 2)
v_i1, v_i2, v_i3 = Vertex('I', 0), Vertex('I', 1), Vertex('I', 2)
# Functions
S1, S2, S3 = sym.Function(str(Term([v_s1]))), sym.Function(str(Term([v_s2]))), sym.Function(str(Term([v_s3])))
I1, I2, I3 = sym.Function(str(Term([v_i1]))), sym.Function(str(Term([v_i2]))), sym.Function(str(Term([v_i3])))
S1I2, I1S2 = sym.Function(str(Term([v_s1, v_i2]))), sym.Function(str(Term([v_i1, v_s2])))
S1I3, I1S3 = sym.Function(str(Term([v_s1, v_i3]))), sym.Function(str(Term([v_i1, v_s3])))
S2I3, I2S3 = sym.Function(str(Term([v_s2, v_i3]))), sym.Function(str(Term([v_i2, v_s3])))
S1S2I3, S1I2S3 = sym.Function(str(Term([v_s1, v_s2, v_i3]))), sym.Function(str(Term([v_s1, v_i2, v_s3])))
S1I2I3, I1S2S3 = sym.Function(str(Term([v_s1, v_i2, v_i3]))), sym.Function(str(Term([v_i1, v_s2, v_s3])))
I1I2S3, I1S2I3 = sym.Function(str(Term([v_i1, v_i2, v_s3]))), sym.Function(str(Term([v_i1, v_s2, v_i3])))
# Initial conditions
no = 0.1
yes = 0.9
IV = {S1(0): no, S2(0): yes, S3(0): yes, I1(0): yes, I2(0): no, I3(0): no}
IV[S1I2(0)] = IV[S1(0)] * IV[S2(0)]
IV[I1S2(0)] = IV[I1(0)] * IV[S2(0)]
IV[S1I3(0)] = IV[S1(0)] * IV[I3(0)]
IV[I1S3(0)] = IV[I1(0)] * IV[S3(0)]
IV[S2I3(0)] = IV[S2(0)] * IV[I3(0)]
IV[I2S3(0)] = IV[I2(0)] * IV[S3(0)]
IV[S1S2I3(0)] = IV[S1(0)] * IV[S2(0)] * IV[I3(0)]
IV[S1I2S3(0)] = IV[S1(0)] * IV[I2(0)] * IV[S3(0)]
IV[S1I2I3(0)] = IV[S1(0)] * IV[I2(0)] * IV[I3(0)]
IV[I1S2S3(0)] = IV[I1(0)] * IV[S2(0)] * IV[S3(0)]
IV[I1I2S3(0)] = IV[I1(0)] * IV[I2(0)] * IV[S3(0)]
IV[I1S2I3(0)] = IV[I1(0)] * IV[S2(0)] * IV[I3(0)]

functions = [S1(t), S2(t), S3(t), I1(t), I2(t), I3(t),
             S1I2(t), I1S2(t), S1I3(t), I1S3(t), S2I3(t), I2S3(t),
             S1S2I3(t), S1I2S3(t), S1I2I3(t), I1S2S3(t), I1I2S3(t), I1S2I3(t)]

beta, gamma = 0.8, 0.2
triangle_generated = generate_equations(nx.cycle_graph(3), get_SIR(beta, gamma))
equations = []
for list_of_eqn in triangle_generated.values():
    for eqn in list_of_eqn:
        print(eqn)
        equations.append(eqn)


def get_numerical_sol_from_generated():
    plt.clf()
    print('\npassing equations and ICs into numerical solver...')
    sol = solve(triangle_generated, nx.cycle_graph(3), init_cond=IV, t_max=10, step=5e-2, atol=1e-4, rtol=1e-4,
                print_option='full')
    print(sol['message'])
    final_vals = []
    for i in sol.y:
        final_vals.append(i[-1])
    print([round(y, 3) for y in final_vals])

    plot_numerical(sol)


def plot_numerical(sol):
    plt.plot(sol.t, sol.y[0],
             label=f'${sym.Integral(equations[0].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009', ']'))
    for i in range(1, len(equations)):
        plt.plot(sol.t, sol.y[i],
                 label=f'${sym.Integral(equations[i].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009', ']'))
    plt.title("Numerical solution to triangle system")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.size = (10, 8)
    plt.ylim([0, 1])
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
    sol = sym.solvers.ode.systems.dsolve_system(eqs=equations, funcs=functions, t=t, ics=IV)[0]
    print(f'solved in {time() - start}s!')
    print(sol)
    at_ten = []
    for eq in sol:
        at_ten.append(eq.subs(t, 10))
    print(at_ten)
    print('analytic solution:\n', sol)
    plot_analytic(sol)


def plot_analytic(sol):
    t_range = (t, 0, 10)
    p = sym.plot(sol[0].rhs, t_range,
                 label=f'${sym.Integral(equations[0].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009', ']'),
                 legend=True, show=False)
    for i in range(1, len(equations)):
        p.extend(sym.plot(sol[i].rhs, t_range,
                          label=f'${sym.Integral(equations[i].lhs).doit()}$'.replace('\u3008', '[').replace('\u3009', ']'),
                          legend=True, show=False))
    p.title = "Analytic solution to triangle system"
    p.x_label = "Time"
    p.y_label = "Probability"
    p.size = (10, 8)
    p.save('test-exact.png')
    p.show()
    plt.tight_layout()
    plt.clf()


get_analytic_sol_from_generated()
get_numerical_sol_from_generated()
