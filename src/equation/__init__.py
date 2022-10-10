from datetime import datetime

import networkx
import numpy as np
import sympy as sym

from equation import eqn_generation, closures
from equation.Term import Term, Vertex
from equation.eqn_generation import get_single_equations, generate_equations
from model_params.cmodel import CModel

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')   # Recovery rate


def main():
    for i in range(1, 20):
        path = networkx.path_graph(i)
        print(f'{i} VERTICES')
        for _ in range(0, 10):
            full_start = datetime.now()
            equations = eqn_generation.generate_equations(path, SIR)
            print("REG:", len(equations), "=", int((3 * i * i - i + 2) / 2))
            full_time_taken = datetime.now() - full_start
            print('FULL TIME TAKEN:', full_time_taken)

            closed_start = datetime.now()
            closed_equations = eqn_generation.generate_equations(path, SIR, closures=True)
            print("CLOSED:", len(closed_equations), "=", 5 * i - 3)
            closed_time_taken = datetime.now() - closed_start
            print('CLOSURES TIME TAKEN:', closed_time_taken)


if __name__ == '__main__':
    main()
