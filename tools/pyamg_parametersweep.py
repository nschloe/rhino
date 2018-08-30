# -*- coding: utf-8 -*-
#
"""
Solve the linearized Ginzburg--Landau problem.
"""

from solver_diagnostics import solver_diagnostics  # pyamg
import numpy as np

import meshplex
import pynosh.modelevaluator_nls


def _main():
    """Main function.
    """
    args = _parse_input_arguments()

    # read the mesh
    # print "Reading the mesh...",
    mesh, point_data, _ = meshplex.reader.read(args.filename, timestep=args.timestep)
    psi0 = point_data["psi"][:, 0] + 1j * point_data["psi"][:, 1]

    # build the model evaluator
    mu = 1.0e-1
    g = 1.0
    num_nodes = len(mesh.node_coords)
    V = -np.ones(num_nodes)
    modeleval = pynosh.modelevaluator_nls.NlsModelEvaluator(
        mesh, V=V, A=point_data["A"]
    )

    # Get the preconditioner for in matrix form
    keo = modeleval._get_keo(mu)
    if g > 0.0:
        if modeleval.mesh.control_volumes is None:
            modeleval.mesh.compute_control_volumes(variant=modeleval.cv_variant)
        alpha = (
            g
            * 2.0
            * (psi0.real ** 2 + psi0.imag ** 2)
            * modeleval.mesh.control_volumes.reshape(psi0.shape)
        )
        num_unknowns = len(psi0)
        from scipy import sparse

        prec = keo + sparse.spdiags(alpha, [0], num_unknowns, num_unknowns)
    else:
        prec = keo

    # https://code.google.com/p/pyamg/source/browse/trunk/Examples/SolverDiagnostics/solver_diagnostics.py
    solver_diagnostics(
        prec, fname="solver_diagnostic", definiteness="positive", symmetry="hermitian"
    )
    return


def _parse_input_arguments():
    """Parse input arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Does a parameter sweep for PyAMG on K."
    )

    parser.add_argument(
        "filename",
        metavar="FILE",
        type=str,
        help="file containing the geometry and initial state",
    )

    parser.add_argument(
        "--timestep",
        "-t",
        metavar="TIMESTEP",
        dest="timestep",
        nargs="?",
        type=int,
        const=0,
        default=0,
        help="read a particular time step (default: 0)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    _main()
