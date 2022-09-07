import networkx as nx
import sympy as sym

from equation import eqn_generation
from model_params.cmodel import CModel


def main():
    SIR = CModel('SIR')
    SIR.set_coupling_rate('S*I:S=>I', 1, name='beta')  # Infection rate
    SIR.set_coupling_rate('I:I=>R', 3, name='gamma')  # Recovery rate

    path = nx.path_graph(5)
    singles_equations = eqn_generation.get_single_equations(path, SIR)
    term = list(singles_equations.keys())[0]
    expression = 0
    for t in singles_equations[term]:
        expression += float(t[1]) * sym.Symbol(str(t[0]))

    eqn = sym.Eq(sym.Symbol(str(term)), expression)
    print(str(eqn))


if __name__ == '__main__':
    main()
