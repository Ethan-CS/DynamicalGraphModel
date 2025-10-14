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

## Large-scale experiments

- Use `PYTHONPATH=src python -m experiments.run_experiments --help` to discover the CLI for running larger runtimes, including 100-vertex graphs, multiple graph families, and long timeouts. Experiments can be described ad-hoc via flags or via a JSON file (see `experiments/config.py`).
- If `--config` is omitted the runner automatically consumes `src/experiments/default_config.json`, which spans Erdős–Rényi, Barabási–Albert, Watts–Strogatz, random regular, and random geometric graphs at 100 vertices with extended timeouts.
- Results are appended to `data/experiment_results.csv` by default; override with `--output`.

## Remote deployment on fatanodes

- Prepare a JSON configuration with the experiments to run (e.g. `experiments.json`).
- Stage and launch the code on the remote machines with `python -m deployment.remote_launcher --gateway <gateway-host> --local-config experiments.json`.
- The launcher syncs the repository to `~/DynamicalGraphModel` on the head node, uploads the JSON config, and starts partitioned runs across the compute nodes via `nohup`, writing logs to `~/DynamicalGraphModel/logs` and results to `~/DynamicalGraphModel/data/remote_results.csv`.
- Pass `--dry-run` to inspect the exact SSH/rsync commands without executing them, and adjust `--python` or `--compute-nodes` if a custom environment is required.


------
[^1]: Ethan Hunter, Jessica Enright, Alice Miller, _Feasibility assessments of a dynamical approach to compartmental modelling on graphs: Scaling limits and performance analysis,
Theoretical Computer Science_, Volume 980 (2023), [doi.org/10.1016/j.tcs.2023.114247](https://www.sciencedirect.com/science/article/pii/S0304397523005601)
