# -*- coding: utf-8 -*-
#
import time

import matplotlib2tikz
import matplotlib.pyplot as pp

# from lobpcg import lobpcg as my_lobpcg
import meshplex
import numpy as np
import pynosh.modelevaluator_nls
from scipy.linalg import eig, eigh, norm
from scipy.sparse import spdiags


def _main():
    """Main function."""
    args = _parse_input_arguments()

    # read the mesh
    mesh, point_data, field_data = meshplex.read(args.filename, timestep=args.timestep)

    # build the model evaluator
    modeleval = pynosh.modelevaluator_nls.NlsModelEvaluator(
        mesh=mesh, A=point_data["A"], V=point_data["V"], preconditioner_type="exact"
    )

    # set the range of parameters
    steps = 10
    mus = np.linspace(0.5, 5.5, steps)

    num_unknowns = len(mesh.node_coords)

    # initial guess for the eigenvectors
    # psi = np.random.rand(num_unknowns) + 1j * np.random.rand(num_unknowns)
    psi = np.ones(num_unknowns)  # + 1j * np.ones(num_unknowns)
    psi *= 0.5
    # psi = 4.0 * 1.0j * np.ones(num_unknowns)
    print(num_unknowns)
    eigenvals_list = []

    g = 10.0
    for mu in mus:
        if args.operator == "k":
            # build dense KEO
            A = modeleval._get_keo(mu).toarray()
            B = None
        elif args.operator == "p":
            # build dense preconditioner
            P = modeleval.get_preconditioner(psi, mu, g)
            A = P.toarray()
            B = None
        elif args.operator == "j":
            # build dense jacobian
            J1, J2 = modeleval.get_jacobian_blocks(psi, mu, g)
            A = _build_stacked_operator(J1.toarray(), J2.toarray())
            B = None
        elif args.operator == "kj":
            J1, J2 = modeleval.get_jacobian_blocks(psi)
            A = _build_stacked_operator(J1.toarray(), J2.toarray())

            modeleval._assemble_keo()
            K = _modeleval._keo
            B = _build_stacked_operator(K.toarray())
        elif args.operator == "pj":
            J1, J2 = modeleval.get_jacobian_blocks(psi)
            A = _build_stacked_operator(J1.toarray(), J2.toarray())

            P = modeleval.get_preconditioner(psi)
            B = _build_stacked_operator(P.toarray())
        else:
            raise ValueError("Unknown operator '", args.operator, "'.")

        print("Compute eigenvalues for mu =", mu, "...")
        # get smallesteigenvalues
        start_time = time.clock()
        # use eig as the problem is not symmetric (but it is self-adjoint)
        eigenvals, U = eig(
            A,
            b=B,
            # lower = True,
        )
        end_time = time.clock()
        print("done. (", end_time - start_time, "s).")

        # sort by ascending eigenvalues
        assert norm(eigenvals.imag, np.inf) < 1.0e-14
        eigenvals = eigenvals.real
        sort_indices = np.argsort(eigenvals.real)
        eigenvals = eigenvals[sort_indices]
        U = U[:, sort_indices]

        # rebuild complex-valued U
        U_complex = _build_complex_vector(U)
        # normalize
        for k in range(U_complex.shape[1]):
            norm_Uk = np.sqrt(
                modeleval.inner_product(U_complex[:, [k]], U_complex[:, [k]])
            )
            U_complex[:, [k]] /= norm_Uk

        ## Compare the different expressions for the eigenvalues.
        # for k in xrange(len(eigenvals)):
        #    JU_complex = J1 * U_complex[:,[k]] + J2 * U_complex[:,[k]].conj()
        #    uJu = modeleval.inner_product(U_complex[:,k], JU_complex)[0]

        #    PU_complex = P * U_complex[:,[k]]
        #    uPu = modeleval.inner_product(U_complex[:,k], PU_complex)[0]

        #    KU_complex = modeleval._keo * U_complex[:,[k]]
        #    uKu = modeleval.inner_product(U_complex[:,k], KU_complex)[0]

        #    # expression 1
        #    lambd = uJu / uPu
        #    assert abs(eigenvals[k] - lambd) < 1.0e-10, abs(eigenvals[k] - lambd)

        #    # expression 2
        #    alpha = modeleval.inner_product(U_complex[:,k]**2, psi**2)
        #    lambd = uJu / (-uJu + 1.0 - alpha)
        #    assert abs(eigenvals[k] - lambd) < 1.0e-10

        #    # expression 3
        #    alpha = modeleval.inner_product(U_complex[:,k]**2, psi**2)
        #    beta = modeleval.inner_product(abs(U_complex[:,k])**2, abs(psi)**2)
        #    lambd = -1.0 + (1.0-alpha) / (uKu + 2*beta)
        #    assert abs(eigenvals[k] - lambd) < 1.0e-10

        #    # overwrite for plotting
        #    eigenvals[k] = 1- alpha

        eigenvals_list.append(eigenvals)

    # plot the eigenvalues
    # _plot_eigenvalue_series( mus, eigenvals_list )
    for ev in eigenvals_list:
        pp.plot(ev, ".")

    # pp.plot( mus,
    # small_eigenvals_approx,
    # '--'
    # )
    # pp.legend()
    pp.title("eigenvalues of %s" % args.operator)

    # pp.ylim( ymin = 0.0 )

    # pp.xlabel( '$\mu$' )

    pp.show()

    # matplotlib2tikz.save('eigenvalues.tikz',
    # figurewidth = '\\figurewidth',
    # figureheight = '\\figureheight'
    # )
    return


def _build_stacked_operator(A, B=None):
    """Build the block operator.
    [ A.real+B.real, -A.imag+B.imag ]
    [ A.imag+B.imag,  A.real-B.real ]
    """
    out = np.empty((2 * A.shape[0], 2 * A.shape[1]), dtype=float)
    out[0::2, 0::2] = A.real
    out[0::2, 1::2] = -A.imag
    out[1::2, 0::2] = A.imag
    out[1::2, 1::2] = A.real

    if B is not None:
        assert A.shape == B.shape
        out[0::2, 0::2] += B.real
        out[0::2, 1::2] += B.imag
        out[1::2, 0::2] += B.imag
        out[1::2, 1::2] -= B.real

    return out


def _build_complex_vector(x):
    """Build complex vector."""
    xreal = x[0::2, :]
    ximag = x[1::2, :]
    return xreal + 1j * ximag


def _build_real_vector(x):
    """Build complex vector."""
    xx = np.empty((2 * len(x), 1))
    xx[0::2, :] = x.real
    xx[1::2, :] = x.imag
    return xx


def _parse_input_arguments():
    """Parse input arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute all eigenvalues of a specified."
    )

    parser.add_argument(
        "filename",
        metavar="FILE",
        type=str,
        help="ExodusII file containing the geometry and initial state",
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

    parser.add_argument(
        "--operator",
        "-o",
        metavar="OPERATOR",
        dest="operator",
        nargs="?",
        type=str,
        const="k",
        default="k",
        help="operator to compute the eigenvalues of (k, p, j, pj; default: k)",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    _main()
