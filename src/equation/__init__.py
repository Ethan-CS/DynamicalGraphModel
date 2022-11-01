import networkx

from equation import generation, closing
from equation.Term import Term, Vertex
from equation.generation import get_single_equations, generate_equations
from equation.solving import initial_conditions, solve_equations
from equation_MC_comparison import measure_generation_runtimes


def main():
    with open(f'data/path_data.csv', 'w') as f:
        for i in range(1, 11):
            print(f'i={i}')
            measure_generation_runtimes(g=networkx.path_graph(i), num_iter=10, timeout=100, f=f)
    with open(f'data/random_tree_data.csv', 'w') as f:
        for i in range(1, 11):
            print(f'i={i}')
            measure_generation_runtimes(g=networkx.random_tree(i), num_iter=10, timeout=100, f=f)
    with open(f'data/cycle_data.csv', 'w') as f:
        for i in range(1, 11):
            print(f'i={i}')
            measure_generation_runtimes(g=networkx.cycle_graph(i), num_iter=10, timeout=100, f=f)


if __name__ == '__main__':
    main()
