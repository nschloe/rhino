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

### License

rhino is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
