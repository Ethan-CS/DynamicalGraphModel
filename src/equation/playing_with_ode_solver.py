import networkx as nx
import sympy as sym
import matplotlib

from equation import generate_equations, get_SIR

matplotlib.use('TkAgg')  # Stops an annoying bug


def main():
    # Equations for the triangle network
    t = sym.symbols('t')

    tau = 0.3
    gamma = 0.1

    S1 = sym.Function('S1')
    S2 = sym.Function('S2')
    S3 = sym.Function('S3')
    I1 = sym.Function('I1')
    I2 = sym.Function('I2')
    I3 = sym.Function('I3')

    S1I2 = sym.Function('S1 I2')
    I1S2 = sym.Function('I1 S2')
    S1I3 = sym.Function('S1 I3')
    I1S3 = sym.Function('I1 S3')
    S2I3 = sym.Function('S2 I3')
    I2S3 = sym.Function('I2 S3')

    S1S2I3 = sym.Function('S1 S2 I3')
    S1I2S3 = sym.Function('S1 I2 S3')
    S1I2I3 = sym.Function('S1 I2 I3')
    I1S2S3 = sym.Function('I1 S2 S3')
    I1I2S3 = sym.Function('I1 I2 S3')
    I1S2I3 = sym.Function('I1 S2 I3')

    print('setting up equations...')
    triangle_generated = generate_equations(nx.cycle_graph(3), get_SIR())

    functions = [S1(t), S2(t), S3(t), I1(t), I2(t), I3(t),
                 S1I2(t), I1S2(t), S1I3(t), I1S3(t), S2I3(t), I2S3(t),
                 S1S2I3(t), S1I2S3(t), S1I2I3(t), I1S2S3(t), I1I2S3(t), I1S2I3(t)]

    print('setting up initial conditions...')

    IV = {S1(0): 0.8, S2(0): 0.5, S3(0): 0.2, I1(0): 0.2, I2(0): 0.5, I3(0): 0.8}

    IV[S1I2(0)] = IV[S1(0)] * IV[S2(0)]
    IV[I1S2(0)] = IV[I1(0)] * IV[S2(0)]
    IV[S1I3(0)] = IV[S1(0)] * IV[I3(0)]
    IV[I1S3(0)] = IV[I1(0)] * IV[S3(0)]
    IV[S2I3(0)] = IV[S2(0)] * IV[I3(0)]
    IV[I2S3(0)] = IV[I2(0)] * IV[S3(0)]
    IV[S1S2I3(0)] = IV[S1(0)] * IV[S2(0)] * IV[I3(0)]
    IV[S1I2S3(0)] = IV[S1(0)] * IV[I2(0)] * IV[S3(0)]
    IV[S1I2I3(0)] = IV[S1(0)] * IV[I2(0)] * IV[I3(0)]
    IV[I1S2S3(0)] = IV[I1(0)] * IV[S2(0)] * IV[S3(0)]
    IV[I1I2S3(0)] = IV[I1(0)] * IV[I2(0)] * IV[S3(0)]
    IV[I1S2I3(0)] = IV[I1(0)] * IV[S2(0)] * IV[I3(0)]

    print('passing equations and I.C.s into solver...')
    sol = sym.solvers.ode.systems.dsolve_system(eqs=triangle_generated, funcs=functions,
                                                t=t, ics=IV)
    print('solved!')

    t_range = (t, 0, 25)
    p = sym.plot(sol[0][0].rhs, t_range, label=str(sol[0][0].lhs), show=False, legend=True)
    for i in range(1, len(sol[0])):
        p.extend(sym.plot(sol[0][i].rhs, t_range, label=str(sol[0][i].lhs), show=False))

    p.title = "E.g. solution to triangle system"
    p.x_label = "Time"
    p.y_label = "Probability"
    p.size = (10, 8)

    p.save('test.png')
    p.show()


if __name__ == '__main__':
    main()
