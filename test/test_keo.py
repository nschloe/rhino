# -*- coding: utf-8 -*-
#
import os

import meshplex
import numpy
import pytest

from pynosh import modelevaluator_nls


@pytest.mark.parametrize(
    "filename, control_values",
    [
        ("rectanglesmall.e", [0.0063121712308067401, 10.224658806561596]),
        ("pacman.e", [0.37044264296585938, 10.000520856079092]),
        ("cubesmall.e", [8.3541623155714007e-05, 10.058364522531498]),
        ("brick-w-hole.e", [0.16763276012920181, 15.131119904340618]),
    ],
)
def test(filename, control_values):
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, filename)
    mu = 1.0e-2

    # read the mesh
    mesh, point_data, field_data, _ = meshplex.read(filename)

    # build the model evaluator
    modeleval = modelevaluator_nls.NlsModelEvaluator(
        mesh, V=point_data["V"], A=point_data["A"]
    )

    # Assemble the KEO.
    keo = modeleval._get_keo(mu)

    tol = 1.0e-13

    # Check that the matrix is Hermitian.
    KK = keo - keo.H
    assert numpy.all(numpy.abs(KK.sum()) < tol)

    # Check the matrix sum.
    assert numpy.all(numpy.abs(control_values[0] - keo.sum()) < tol)

    # Check the 1-norm of the matrix |Re(K)| + |Im(K)|.
    # This equals the 1-norm of the matrix defined by the block
    # structure
    #   Re(K) -Im(K)
    #   Im(K)  Re(K).
    K = abs(keo.real) + abs(keo.imag)
    assert numpy.all(numpy.abs(control_values[1] - numpy.max(K.sum(0))) < tol)
    return
