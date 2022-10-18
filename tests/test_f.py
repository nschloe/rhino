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
        (
            "rectanglesmall.e",
            [0.50126061034211067, 0.24749434381636057, 0.12373710977782607],
        ),
        ("pacman.e", [0.71366475047893463, 0.12552206259336218, 0.055859319123267033]),
        # Geometric ce_ratios
        (
            "cubesmall.e",
            [0.00012499993489764605, 4.419415080700124e-05, 1.5624991863028015e-05],
        ),
        (
            "brick-w-hole.e",
            [1.8317481239998066, 0.15696030933066502, 0.029179895038465554],
        ),
        # Algebraic ce_ratios
        # (
        #     "cubesmall.e",
        #     [8.3541623156163313e-05, 2.9536515963905867e-05, 1.0468744547749431e-05],
        # ),
        # (
        #     "brick-w-hole.e",
        #     [1.8084716102419285, 0.15654267585120338, 0.03074423493622647],
        # ),
    ],
)
def test(filename, control_values):
    # read the mesh
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, filename)
    mesh = meshplex.read(filename)
    mu = 1.0e-2

    # build the model evaluator
    modeleval = modelevaluator_nls.NlsModelEvaluator(
        mesh, V=mesh.point_data["V"], A=mesh.point_data["A"]
    )

    # compute the Ginzburg-Landau residual
    psi = point_data["psi"][:, 0] + 1j * point_data["psi"][:, 1]
    r = modeleval.compute_f(psi, mu, 1.0)

    # scale with D for compliance with the Nosh (C++) tests
    if mesh.control_volumes is None:
        mesh.compute_control_volumes()
    r *= mesh.control_volumes.reshape(r.shape)

    tol = 1.0e-13
    # For C++ Nosh compatibility:
    # Compute 1-norm of vector (Re(psi[0]), Im(psi[0]), Re(psi[1]), ... )
    alpha = numpy.linalg.norm(r.real, ord=1) + numpy.linalg.norm(r.imag, ord=1)
    assert abs(control_values[0] - alpha) < tol
    assert abs(control_values[1] - numpy.linalg.norm(r, ord=2)) < tol
    # For C++ Nosh compatibility:
    # Compute inf-norm of vector (Re(psi[0]), Im(psi[0]), Re(psi[1]), ... )
    alpha = max(
        numpy.linalg.norm(r.real, ord=numpy.inf),
        numpy.linalg.norm(r.imag, ord=numpy.inf),
    )
    assert abs(control_values[2] - alpha) < tol
    return
