<p align="center">
  <a href="https://github.com/nschloe/rhino"><img alt="logo" src="logo/logo.svg" width="40%"></a>
</p>

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/rhino/master.svg)](https://circleci.com/gh/nschloe/rhino/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/rhino.svg)](https://codecov.io/gh/nschloe/rhino)

rhino is a solver package for nonlinear Schrödinger equations. It contains the
respective model evaluators along with an implementation of Newton's method and
optional preconditioner for its linearization.

rhino uses [KryPy](https://github.com/andrenarchy/krypy) for the solution of
linear equation systems and employs its deflation capabilities. The package
[meshplex](https://github.com/nschloe/meshplex) is used to construct the
finite-volume discretization.

# Usage

### Documentation

The documentation is hosted at
[rhino.readthedocs.org](http://rhino.readthedocs.org).

### Example

![Ginzburg-Landau solution abs](https://nschloe.github.io/rhino/solution-abs.png)
![Ginzburg-Landau solution arg](https://nschloe.github.io/rhino/solution-arg.png)

Absolute value and complex argument of a solution of the _Ginzburg-Landau equations_, a
particular instance of nonlinear Schrödinger equations. The number of nodes in the
discretization is 72166 for this example.

# Development

rhino is currently maintained by [Nico Schlömer](https://github.com/nschloe). Feel free
to contact Nico. Please submit feature requests and bugs as GitHub issues.

# References

rhino was used to conduct the numerical experiments in the paper

- [Preconditioned Recycling Krylov subspace methods for self-adjoint problems, A. Gaul and N. Schlömer, arxiv: 1208.0264, 2012](http://arxiv.org/abs/1208.0264).

### License

rhino is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
