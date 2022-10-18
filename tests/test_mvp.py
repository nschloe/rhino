import os

import meshplex
import numpy

from rhino import magnetic_vector_potentials as mvp


def _run(filename, control_values):
    """Test $\int_{\Omega} A^2$."""

    # read the mesh
    mesh, _, _, _ = meshplex.read(filename)
    if mesh.control_volumes is None:
        mesh.compute_control_volumes()

    tol = 1.0e-10

    A = mvp.constant_field(mesh.node_coords, numpy.array([0, 0, 1]))
    integral = numpy.sum(mesh.control_volumes * numpy.sum(A**2, axis=1))
    assert numpy.all(numpy.abs(control_values["z"] - integral) < tol)

    # If this is a 2D mesh, append the z-component 0 to each node
    # to make sure that the magnetic vector potentials can be
    # calculated.
    points = mesh.node_coords.copy()
    if points.shape[1] == 2:
        points = numpy.column_stack((points, numpy.zeros(len(points))))

    A = mvp.magnetic_dipole(
        points, x0=numpy.array([0, 0, 10]), m=numpy.array([0, 0, 1])
    )
    integral = numpy.sum(mesh.control_volumes * numpy.sum(A**2, axis=1))
    assert numpy.all(numpy.abs(control_values["dipole"] - integral) < tol)

    # import time
    # start = time.time()
    A = mvp.magnetic_dot(mesh.node_coords, 2.0, [10.0, 11.0])
    # A = numpy.empty((len(points), 3), dtype=float)
    # for k, node in enumerate(points):
    # A[k] = mvp.magnetic_dot(node[0], node[1], 2.0, 10.0, 11.0)
    # end = time.time()
    # print end-start
    integral = numpy.sum(mesh.control_volumes * numpy.sum(A**2, axis=1))
    assert numpy.all(numpy.abs(control_values["dot"] - integral) < tol)
    return


def test_rectanglesmall():
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, "rectanglesmall.e")
    control_values = {
        "z": 63.125,
        "dipole": 0.00012850741240854054,
        "dot": 0.015062118041804408,
    }
    _run(filename, control_values)
    return


def test_pacman():
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, "pacman.e")
    control_values = {
        "z": 3730.2268660993054,
        "dipole": 0.0037630906971841487,
        "dot": 0.46680832033437036,
    }
    _run(filename, control_values)
    return


def test_cubesmall():
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, "cubesmall.e")
    control_values = {
        "z": 1.25,
        "dipole": 0.00015098959555300608,
        "dot": 0.00052723843169109191,
    }
    _run(filename, control_values)
    return


def test_brick():
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, "brick-w-hole.e")
    control_values = {
        "z": 1687.6928071551067,
        "dipole": 0.014339810567783946,
        "dot": 0.4275090788990229,
    }
    _run(filename, control_values)
    return


if __name__ == "__main__":
    test_pacman()
