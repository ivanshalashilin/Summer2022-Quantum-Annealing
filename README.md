# Dynamics of Quantum Annealing - Ivan Shalashilin Summer 2022

All of the code made over the course of my summer research project investigating the dynamics of quantum annealing. Namely, introducing a catalyst and investigating the affects of it on a 5-spin system.

Data for the plots can be found at [this dropbox]()


# Classes

## hamv2.py

- class for doing annealing process, plotting and obtaining data for static and dynamic plots

> some methods require `evals` or `evecs` input

## bacon.py

- class for generating $H_d$ and $H_p$ using Kronecker products
- `wmis/bacon_demo.py` provides a demo of generating hamiltonians


# wmis

- `wmis_template.py` demonstrates use of input wmis hamiltonians into hamv2 Classes
- `2djxx/2d_get_data.py` loops through desired range of $J_{xx}$ and $T$, obtaining all anneal data and writing to a `.pkl`

- `2djxx/2d_get_data.py` plots the data in multiple forms once read from a `.pkl` file


# open systems

- a few trial runs with HOQST `OpenQuatnumTools.jl`, mostly edits to [An Intro to HOQST - Lindblad equation](https://uscqserver.github.io/HOQSTTutorials.jl/html/introduction/02-lindblad_equation.html)
- Guide to code follows [HOQST docs](https://docs.juliahub.com/OpenQuantumTools/iRrSZ/0.6.2/index.html)

- `python.json` included for completeness, some nice code snippets to use for reading and writing `pkl` data, and a few plots.

