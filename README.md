<p align="center">
  <a href="https://github.com/nschloe/rhino"><img alt="logo" src="logo/logo.svg" width="40%"></a>
</p>

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/rhino/ci?style=flat-square)](https://github.com/nschloe/rhino/actions?query=workflow%3Atests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

rhino is a solver package for nonlinear Schrödinger equations. It contains the
respective model evaluators along with an implementation of Newton's method and
optional preconditioner for its linearization.

rhino uses [KryPy](https://github.com/andrenarchy/krypy) for the solution of
linear equation systems and employs its deflation capabilities. The package
[meshplex](https://github.com/nschloe/meshplex) is used to construct the
finite-volume discretization.

### License

rhino is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
