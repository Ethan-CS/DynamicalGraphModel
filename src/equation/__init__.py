import networkx

from equation import eqn_generation, closures
from model_params.cmodel import CModel

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')   # Recovery rate


def main():
    p6 = networkx.path_graph(6)
    singles_equations = eqn_generation.get_single_equations(p6, SIR)
    equations = eqn_generation.generate_equations(singles_equations, 2, p6, SIR)

    to_print = ''
    for term in equations.keys():
        to_print += f'\n{term}='
        for t in equations[term]:
            to_print += str(t[1]) + str(t[0])

    print(to_print)

    print('***** CLOSURES *****')
    closed_equations = eqn_generation.generate_equations(singles_equations, 2, p6, SIR, True)

    to_print = ''
    for term in closed_equations.keys():
        to_print += f'\n{term}='
        for t in closed_equations[term]:
            to_print += str(t[1]) + str(t[0])

    print(to_print)


if __name__ == '__main__':
    main()
