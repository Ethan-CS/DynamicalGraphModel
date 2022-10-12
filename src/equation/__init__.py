from datetime import datetime

import networkx
import pandas as pd
import sympy as sym

from equation import eqn_generation, closures
from equation.Term import Term, Vertex
from equation.eqn_generation import get_single_equations, generate_equations
from equation.eqn_solving import initial_conditions
from model_params.cmodel import CModel

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')  # Recovery rate


def get_SIR():
    return SIR


def main():
    # full_results = {}
    # closed_results = {}
    # num_eqns = {}
    # for i in range(1, 11):
    #     full_results[i] = []
    #     closed_results[i] = []
    #     num_eqns[i] = {'full': int((3 * i * i - i + 2) / 2), 'closed': 5 * i - 3}
    #
    #     path = networkx.path_graph(i)
    #     init_conditions = initial_conditions(path.nodes)
    #     for _ in range(0, 10):
    #         full_start = datetime.now()
    #         full_equations = eqn_generation.generate_equations(path, SIR)
    #         assert len(full_equations) == int((3 * i * i - i + 2) / 2)
    #         full_time_taken = datetime.now() - full_start
    #
    #         closed_start = datetime.now()
    #         closed_equations = eqn_generation.generate_equations(path, SIR, closures=True)
    #         if i > 3:
    #             assert len(closed_equations) == (5 * i - 3), 'something wrong with this set of equations:\n' + \
    #                                                          str([str(eq.lhs) for eq in closed_equations])
    #         closed_time_taken = datetime.now() - closed_start
    #         print(
    #             f'{i},{len(full_equations)},{full_time_taken.seconds}.{full_time_taken.microseconds},'
    #             f'{len(closed_equations)},{closed_time_taken.seconds}.{closed_time_taken.microseconds}')
    #
    #         full_results[i].append(full_time_taken)
    #         closed_results[i].append(closed_time_taken)
    #
    # full_data = pd.DataFrame(full_results)
    # print(full_data)
    # closed_data = pd.DataFrame(closed_results)
    # print(closed_data)
    path = networkx.path_graph(5)
    closed_equations = eqn_generation.generate_equations(path, SIR, closures=False)
    for eq in closed_equations:
        print(eq)


if __name__ == '__main__':
    main()
