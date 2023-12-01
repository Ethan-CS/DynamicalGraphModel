---
title: 'Implementation of an exact, deterministic dynamical approach to compartmental models of disease'
tags:
- Python
- ODE
- compartmental model
- master equation
- moment closure
authors:
- name: Ethan Hunter
  email: e.hunter.2@research.gla.ac.uk
  orcid: 0000-0002-4309-6861
  affiliation: 1
- name: Jessica Enright
  orcid: 0000-0002-0266-3292
  affiliation: 1
- name: Alice Miller
  orcid: 0000-0002-0941-1717
  affiliation: 1
affiliations:
- name: University of Glasgow, Glasgow, G12 8QQ, United Kingdom
  index: 1
  date: 30 November 2023
bibliography: paper.bib
---

# Summary

We provide an implementation of an approach to modelling disease that has been studied by several authors. This
approach was described in [@sharkey:2011] and significantly improved in [@kiss:2017] by the introduction of
a result reducing the number of equations required for an exact representation of the full system dynamics. The
provision of our codebase answers an open problem in [@kiss:2017].

This code heavily extends the procedure described in the literature by opening the
approach up to any (statically defined) compartmental model, whereas its use in literature has so far been restricted to 
$SIR$ (susceptible-infected-recovered) models. Any graph can be defined as the
contact network using the Networkx package [@networkx]. The code generates equations exactly describing
deterministically the dynamics of a compartmental model on a network. These equations can then be solved for given sets
of initial conditions (which nodes are initially infected) and the codebase contains code that plots these results.

This code was used to produce experimental results for an associated publication [@hunter:2023], which outlines broadly
the algorithmic procedure used and discusses the feasibility of this approach for modelling diseases in real-world
scenarios. In particular, in [@hunter:2023] we compare the performance of this code to simulation. Code for testing the
key functionality of this codebase (i.e., generating and solving equations from specified compartmental models and
contact graphs for particular initial conditions) is provided in the `testing` directory.

# Statement of need

Our code answers an open question from [@kiss:2017] by giving procedures to generate and solve equations
for compartmental models with underlying contact networks. Others, such as in [@sharkey:2011], have written
their own code to generate equations for systems up to a certain length of terms (usually terms on one and two 
vertices), which approximates the system dynamics. Instead, our code generates and solves equations up to
the full system size, yielding full, complete and deterministic representations of  model dynamics.

The more usual approach to obtaining modelling results for compartmental models on contact networks is to use a
simulation
approach, which we have included an example of in the `monte_carlo` directory. Using this dynamical approach means that
we can obtain exact, deterministic modelling results (whereas simulation relies on stochasticity, so cannot guarantee
convergence to the correct answer). While in [@hunter:2023] we show that simulation is always faster than generating and
solving equations (and explain the computational theory to address why this is the case), this implementation is
intended for use in situations where epidemiologists require certainty in the correctness of the outputted answer.

# Usage

An example is given in the `__init__.py` file in the `equation` package. Broadly, steps there are as follows:

- Define a compartmental model using the `CModel` class, adapted from [@compyrtment].
    - Define a name, e.g. `model = CModel('SEIRV')`
    - Define transitions and rates, e.g. `model.set_coupling_rate('S*I:S=>E', 1, name='\\beta_1')` for the rate of
      infection for susceptible contacts contracting from infected contacts
- Define a network using [@networkx], e.g. `graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)])`
- To generate equations, use the `generate_equations` method from `equation/generation.py`, e.g.
  `enerate_equations(graph, model, closures=True)`
    - `closures=True` ensures that moment closures are implemented to reduce the number of equations required if
      possible.
      See [kiss:2017, hunter:2023] for explanations of this closure result and the algorithmic procedure respectively.
- In `__init__`, we provide a method that counts the number of equations in a system (and prints out if `verbose` is
  set to `True`)

# References