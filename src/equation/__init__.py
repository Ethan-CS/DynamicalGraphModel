import sys
from datetime import datetime

import networkx

from equation import generation, closing
from equation.Term import Term, Vertex
from equation.generation import get_single_equations, generate_equations
from equation.solving import initial_conditions, solve_equations
from model_params.cmodel import CModel

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')  # Recovery rate


def get_SIR():
    return SIR


def main():
    for g in ['path', 'cycle', 'tree']:
        print(f'generating equations for {g}')
        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(f'data/{g}_data.csv', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('number of vertices,number equations (full),time (full),num equations (closed),time (closed)')
            sys.stdout = original_stdout  # Reset the standard output to its original value
            for i in range(1, 25):
                print(f'i={i}')
                for _ in range(0, 10):
                    print(f'iter {_+1} of 10')
                    if g == 'path':
                        graph = networkx.path_graph(i)
                    elif g == 'cycle':
                        graph = networkx.cycle_graph(i)
                    else:
                        graph = networkx.random_tree(i)

                    init_conditions = initial_conditions(list(graph.nodes))

                    full_start = datetime.now()
                    full_equations = eqn_generation.generate_equations(graph, SIR)
                    solve_equations(full_equations, init_conditions)
                    full_time_taken = datetime.now() - full_start

                    closed_start = datetime.now()
                    closed_equations = eqn_generation.generate_equations(graph, SIR, closures=True)
                    solve_equations(closed_equations, init_conditions)
                    closed_time_taken = datetime.now() - closed_start

                    sys.stdout = f  # Change the standard output to the file we created.
                    print(f'{i},{len(full_equations)},{full_time_taken.seconds}.{full_time_taken.microseconds},'
                          f'{len(closed_equations)},{closed_time_taken.seconds}.{closed_time_taken.microseconds}')
                    sys.stdout = original_stdout  # Reset the standard output to its original value


if __name__ == '__main__':
    main()
