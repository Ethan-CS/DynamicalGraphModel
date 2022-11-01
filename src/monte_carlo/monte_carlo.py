import networkx as nx
import numpy as np

from model_params.cmodel import CModel


def monte_carlo_sim(graph: nx.Graph, model: CModel, init_state: dict, t_max: int):
    state = dict(init_state)
    for _ in range(t_max):
        next_timestep = dict(state)
        for v in graph.nodes:
            if next_timestep[v] == 'I':
                # Get list of vertices that could now become infected
                neighbours = nx.neighbors(graph, v)
                for n in neighbours:
                    if state[n] == 'S':
                        if np.random.random(1) < model.couplings['beta'][1]:
                            next_timestep[n] = 'I'
                if np.random.random(1) < model.couplings['gamma'][1]:
                    next_timestep[v] = 'R'
        state = next_timestep
    return state


def example_monte_carlo():
    tree = nx.random_tree(10)
    SIR = CModel('SIR')
    SIR.set_coupling_rate('S*I:S=>I', 0.6, name='beta')  # Infection rate
    SIR.set_coupling_rate('I:I=>R', 0.15, name='gamma')  # Recovery rate
    initial_state = dict()
    print(str(SIR.states))
    for node in tree.nodes:
        initial_state[node] = 'S'
    initial_state[np.random.choice(tree.nodes)] = 'I'
    print(initial_state)
    print('result:', monte_carlo_sim(tree, SIR, initial_state, 10))


example_monte_carlo()
