import matplotlib

import networkx as nx
import sympy as sym
from jitcode import y, jitcode
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from equation import generate_equations, initial_conditions
from model_params.cmodel import get_SIR

# Set seaborn as default and set resolution and style defaults
sns.set()
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")

# Avoids an annoying error on macOS
matplotlib.use('TkAgg')

g = nx.random_tree(5)
full_equations = generate_equations(g, get_SIR())
functions = [sym.Integral(each.lhs).doit() for each in full_equations]

print('GENERATING SUBSTITUTIONS')

substitutions = {}
for i in range(len(functions)):
    substitutions[functions[i]] = y(i)
print(substitutions)

print('PERFORMING SUBSTITUTIONS')

full_y_equations = {}
for equation in full_equations:
    equation_lhs = sym.Integral(equation.lhs).doit().subs(substitutions)
    equation_rhs = equation.rhs.subs(substitutions)
    full_y_equations[equation_lhs] = equation_rhs

print(full_y_equations)

print('SETTING UP ODE INTEGRATOR')

ODE = jitcode(list(full_y_equations.values()))
ODE.set_integrator("dopri5")
ODE.set_initial_value(list(initial_conditions(list(g.nodes), functions, symbol=sym.symbols('t')).values()), 0)

times = np.arange(start=0, stop=25, step=0.2)
values = {}
for fun in list(substitutions.values()):
    values[fun] = []

for time in times:
    ODE.integrate(time)
    for fun in list(substitutions.values()):
        values[fun].append(ODE.y_dict[fun])

sns.lineplot(data=values)
plt.savefig('test.png')
