# DynamicalGraphModel project

Welcome! This project provides Python code for generating and solving systems of equations describing a specified compartmental model on a specified graph (network).

This project is used in a publication[^1] that assesses more broadly this approach to compartmental modelling on graphs. There is documentation throughout the Python code, but to get started you should read the next section.


## Getting started

 - Go to `equation.__init__.py` to see an example of how equations can be generated (and printed), initial conditions can be specified and equations solved for these conditions.
   - Models can be specified with a graph (we use `networkx`) and a compartmental model (we use a simplified version of `CModel`).
   - Compartmental models can be as simple as SIR (`model_params.CModel` contains a getter method for the usual form of this) 
   - Graphs can be generated using `networkx` methods for simple graph classes or user-specified
   - Initial conditions can be generated using helper methods - see example usage in the `__init__` file
 - If you'd like to compare to a Monte Carlo simulation, go to `monte_carlo.equation_MC_comparison.py` for examples used in the associated publication


------
[^1]: Ethan Hunter, Jessica Enright, Alice Miller, _Feasibility assessments of a dynamical approach to compartmental modelling on graphs: Scaling limits and performance analysis,
Theoretical Computer Science_, Volume 980 (2023), [doi.org/10.1016/j.tcs.2023.114247](https://www.sciencedirect.com/science/article/pii/S0304397523005601)
