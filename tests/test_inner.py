#
import os

import meshplex
import numpy
import pytest

from rhino import modelevaluator_nls


@pytest.mark.skip
@pytest.mark.parametrize(
    "filename, control_values",
    [
        ("rectanglesmall.e", [10.0, 0.0, 250.76609861896702]),
        ("pacman.e", [302.52270072101049, 8.8458601556211267, 1261.5908800348018]),
        ("cubesmall.e", [10.0, 0.0, 237.99535357630012]),
        (
            "brick-w-hole.e",
            [388.68629169464111, 30.434181122856277, -24.459076553128803],
        ),
    ],
)
def test(filename, control_values):
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, filename)

    # read the mesh
    mesh, point_data, field_data, _ = meshplex.read(filename)

    # build the model evaluator
    modeleval = modelevaluator_nls.NlsModelEvaluator(
        mesh, V=point_data["V"], A=point_data["A"]
    )
    tol = 1.0e-12

    # For C++ Nosh compatibility:
    # Compute 1-norm of vector (Re(psi[0]), Im(psi[0]), Re(psi[1]), ... )
    N = len(mesh.node_coords)
    phi0 = 1.0 * numpy.ones((N, 1), dtype=complex)
    phi1 = 1.0 * numpy.ones((N, 1), dtype=complex)
    alpha = modeleval.inner_product(phi0, phi1)[0][0]
    assert abs(control_values[0] - alpha) < tol

    phi0 = numpy.empty((N, 1), dtype=complex)
    phi1 = numpy.empty((N, 1), dtype=complex)
    for k, node in enumerate(mesh.node_coords):
        phi0[k] = numpy.cos(numpy.pi * node[0]) + 1j * numpy.sin(numpy.pi * node[1])
        phi1[k] = numpy.sin(numpy.pi * node[0]) + 1j * numpy.cos(numpy.pi * node[1])
    alpha = modeleval.inner_product(phi0, phi1)[0][0]
    assert abs(control_values[1] - alpha) < tol

    phi0 = numpy.empty((N, 1), dtype=complex)
    phi1 = numpy.empty((N, 1), dtype=complex)
    for k, node in enumerate(mesh.node_coords):
        phi0[k] = numpy.dot(node, node)
        phi1[k] = numpy.exp(1j * numpy.dot(node, node))
    alpha = modeleval.inner_product(phi0, phi1)[0][0]
    assert abs(control_values[2] - alpha) < tol
    return
