import copy
import itertools

import networkx
import numpy as np

from datetime import datetime
from equation import eqn_generation, closures
from equation.eqn_generation import get_single_equations, generate_equations
from model_params.cmodel import CModel

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')   # Recovery rate


def main():
    # for i in range(1, 51):
    #     path = networkx.path_graph(i)
    #     singles_equations = eqn_generation.get_single_equations(path, SIR)
    #
    #     for _ in range(0, 10):
    #         full_start = datetime.now()
    #         equations = eqn_generation.generate_equations(singles_equations, 2, path, SIR)
    #         print("REG:", len(equations), "=", int((3 * i * i - i + 2) / 2))
    #         full_time_taken = datetime.now() - full_start
    #
    #         closed_start = datetime.now()
    #         closed_equations = eqn_generation.generate_equations(singles_equations, 2, path, SIR, True)
    #         print("CLOSED:", len(closed_equations), "=", 5 * i - 3)
    #         closed_time_taken = datetime.now() - closed_start

    path = networkx.cycle_graph(3)
    lollipop_adj = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    print(lollipop_adj)
    lollipop = networkx.from_numpy_matrix(lollipop_adj)

    all_equations = generate_equations(lollipop, SIR)
    print('EQUATIONS:')
    for e in all_equations:
        print(e)
    print('There were', len(all_equations), 'equations')

    closed_equations = generate_equations(lollipop, SIR, closures=True)
    print('EQUATIONS:')
    for e in closed_equations:
        print(e)
    print('There were', len(closed_equations), 'equations')


if __name__ == '__main__':
    main()
