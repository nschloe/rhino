# PyNosh

[![Build Status](https://travis-ci.org/nschloe/pynosh.png?branch=master)](https://travis-ci.org/nschloe/pynosh)
[![Coverage Status](https://coveralls.io/repos/nschloe/pynosh/badge.png?branch=master)](https://coveralls.io/r/nschloe/pynosh?branch=master)
[![Code Health](https://landscape.io/github/nschloe/pynosh/master/landscape.png)](https://landscape.io/github/nschloe/pynosh/master)
[![Documentation Status](https://readthedocs.org/projects/pynosh/badge/?version=latest)](https://readthedocs.org/projects/pynosh/?badge=latest)
[![doi](https://zenodo.org/badge/doi/10.5281/zenodo.10341.png)](https://zenodo.org/record/10341)

PyNosh is a solver package for nonlinear Schrödinger equations. It contains the respective model evaluators along with an implementation of Newton's method and optional preconditioner for its linearization.

PyNosh uses [KryPy](https://github.com/andrenarchy/krypy) for the solution of linear equation systems and employs its deflation capabilities. The package [VoroPy](https://github.com/nschloe/voropy) is used to construct the finite-volume discrezation.


# Usage

### Documentation
The documentation is hosted at
[pynosh.readthedocs.org](http://pynosh.readthedocs.org).

### Example
![Ginzburg-Landau solution](figures/solution-abs.png)
![Ginzburg-Landau solution](figures/solution-arg.png)

Absolute value and complex argument of a solution of the _Ginzburg-Landau equations_, a particular instance of nonlinear Schrödinger equations. The number of nodes in the discretization is 72166 for this example.

# Development
PyNosh is currently maintained by [Nico Schlömer](https://github.com/nschloe). Feel free to contact Nico. Please submit feature requests and bugs as GitHub issues.

PyNosh is developed with continuous integration. Current status: [![Build Status](https://travis-ci.org/nschloe/pynosh.png?branch=master)](https://travis-ci.org/nschloe/pynosh)

# License
PyNosh is free software licensed under the GPL3 License.

# References
PyNosh was used to conduct the numerical experiments in the paper

* [Preconditioned Recycling Krylov subspace methods for self-adjoint problems, A. Gaul and N. Schlömer, arxiv: 1208.0264, 2012](http://arxiv.org/abs/1208.0264).
