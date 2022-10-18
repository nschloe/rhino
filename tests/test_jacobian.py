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
        ("rectanglesmall.e", [20.0126243424616, 20.0063121712308, 0.00631217123080606]),
        ("pacman.e", [605.78628672795264, 605.41584408498682, 0.37044264296586299]),
        # Geometric ce_ratios:
        (
            "cubesmall.e",
            [20.000249999869794, 20.000124999934897, 0.00012499993489734074],
        ),
        ("brick-w-hole.e", [777.7072988143686, 777.5399411018254, 0.16735771254316]),
        (
            "tetrahedron.e",
            [128.3145663425826, 128.3073117169993, 0.0072546255832996644],
        ),
        ("tet.e", [128.316760714389, 128.30840983471703, 0.008350879671951375]),
        # Algebraic ce_ratios:
        # (
        #     "cubesmall.e",
        #     [20.000167083246311, 20.000083541623155, 8.3541623155658495e-05],
        # ),
        # (
        #     "brick-w-hole.e",
        #     [777.70784890954064, 777.54021614941144, 0.16763276012921419],
        # ),
        # (
        #     "tetrahedron.e",
        #     [128.31647020288861, 128.3082636471523, 0.0082065557362998032],
        # ),
        # ("tet.e", [128.31899139655067, 128.30952517579789, 0.0094662207527960365]),
    ],
)
def test(filename, control_values):
    # read the mesh
    this_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(this_path, filename)
    mu = 1.0e-2

    mesh, point_data, field_data, _ = meshplex.read(filename)

    psi = point_data["psi"][:, 0] + 1j * point_data["psi"][:, 1]
    num_unknowns = len(psi)
    psi = psi.reshape(num_unknowns, 1)

    # build the model evaluator
    modeleval = modelevaluator_nls.NlsModelEvaluator(
        mesh, V=point_data["V"], A=point_data["A"]
    )

    # Get the Jacobian
    J = modeleval.get_jacobian(psi, mu, 1.0)

    tol = 1.0e-12

    # [1+i, 1+i, 1+i, ... ]
    phi = (1 + 1j) * numpy.ones((num_unknowns, 1), dtype=complex)
    val = numpy.vdot(phi, mesh.control_volumes.reshape(phi.shape) * (J * phi)).real
    assert abs(control_values[0] - val) < tol

    # [1, 1, 1, ... ]
    phi = numpy.ones((num_unknowns, 1), dtype=complex)
    val = numpy.vdot(phi, mesh.control_volumes[:, None] * (J * phi)).real
    assert abs(control_values[1] - val) < tol

    # [i, i, i, ... ]
    phi = 1j * numpy.ones((num_unknowns, 1), dtype=complex)
    val = numpy.vdot(phi, mesh.control_volumes[:, None] * (J * phi)).real
    assert abs(control_values[2] - val) < tol
    return
