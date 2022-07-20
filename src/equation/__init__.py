import networkx
from cpyment import CModel

from equation import eqn_generation

SIR = CModel('SIR')
SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
SIR.set_coupling_rate('I:I=>R', 3, name='gamma')  # Recovery rate


def main():
    singles_equations = eqn_generation.get_singles_terms(networkx.path_graph(6), SIR)
    for term in singles_equations.keys():
        print(f'{term}={[str(t[1]) + str(t[0]) for t in singles_equations[term]]}')


if __name__ == '__main__':
    main()
