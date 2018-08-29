# pynosh

[![Build Status](https://travis-ci.org/nschloe/pynosh.png?branch=master)](https://travis-ci.org/nschloe/pynosh)
[![Coverage Status](https://img.shields.io/coveralls/nschloe/pynosh.svg)](https://coveralls.io/r/nschloe/pynosh?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pynosh/badge/?version=latest)](https://readthedocs.org/projects/pynosh/?badge=latest)
[![doi](https://zenodo.org/badge/doi/10.5281/zenodo.10341.png)](https://zenodo.org/record/10341)
[![pypi](https://img.shields.io/pypi/v/pynosh.svg)](https://pypi.python.org/pypi/pynosh)

pynosh is a solver package for nonlinear Schrödinger equations. It contains the
respective model evaluators along with an implementation of Newton's method and optional
preconditioner for its linearization.

pynosh uses [KryPy](https://github.com/andrenarchy/krypy) for the solution of linear
equation systems and employs its deflation capabilities. The package
[meshplex](https://github.com/nschloe/meshplex) is used to construct the finite-volume
discretization.


# Usage

### Documentation
The documentation is hosted at
[pynosh.readthedocs.org](http://pynosh.readthedocs.org).

### Example
![Ginzburg-Landau solution abs](https://nschloe.github.io/pynosh/solution-abs.png)
![Ginzburg-Landau solution arg](https://nschloe.github.io/pynosh/solution-arg.png)

Absolute value and complex argument of a solution of the _Ginzburg-Landau equations_, a
particular instance of nonlinear Schrödinger equations. The number of nodes in the
discretization is 72166 for this example.

# Development
pynosh is currently maintained by [Nico Schlömer](https://github.com/nschloe). Feel free
to contact Nico. Please submit feature requests and bugs as GitHub issues.

# License
pynosh is free software licensed under the GPL3 License.

# References
pynosh was used to conduct the numerical experiments in the paper

* [Preconditioned Recycling Krylov subspace methods for self-adjoint problems, A. Gaul and N. Schlömer, arxiv: 1208.0264, 2012](http://arxiv.org/abs/1208.0264).
