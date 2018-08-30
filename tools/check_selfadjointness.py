# -*- coding: utf-8 -*-
#
import numpy as np

import voropy
import pynosh.modelevaluator_nls
import pynosh.modelevaluator_bordering_constant

# import pynosh.numerical_methods as nm


def _main():
    """Main function.
    """
    args = _parse_input_arguments()

    mesh, point_data, field_data = voropy.read(args.filename, timestep=args.timestep)
    N = len(mesh.node_coords)
    # build the model evaluator
    nls_modeleval = pynosh.modelevaluator_nls.NlsModelEvaluator(
        mesh,
        V=-np.ones(N),
        A=point_data["A"],
        preconditioner_type=args.preconditioner_type,
        num_amg_cycles=args.num_amg_cycles,
    )

    current_psi = np.random.rand(N, 1) + 1j * np.random.rand(N, 1)

    if args.bordering:
        modeleval = pynosh.bordered_modelevaluator.BorderedModelEvaluator(nls_modeleval)
        # right hand side
        x = np.empty((N + 1, 1), dtype=complex)
        x[0:N] = current_psi
        x[N] = 1.0
    else:
        modeleval = nls_modeleval
        x = current_psi

    print(("machine eps = %g" % np.finfo(np.complex).eps))

    mu = args.mu
    g = 1.0

    # check the jacobian operator
    J = modeleval.get_jacobian(x, mu, g)
    print(
        (
            "max(|<v,Ju> - <Jv,u>|) = %g"
            % _check_selfadjointness(J, modeleval.inner_product)
        )
    )

    if args.preconditioner_type != "none":
        # check the preconditioner
        P = modeleval.get_preconditioner(x, mu, g)
        print(
            (
                "max(|<v,Pu> - <Pv,u>|) = %g"
                % _check_selfadjointness(P, modeleval.inner_product)
            )
        )
        # Check positive definiteness of P.
        print(
            (
                "min(<u,Pu>) = %g"
                % _check_positivedefiniteness(P, modeleval.inner_product)
            )
        )

        # check the inverse preconditioner
        # Pinv = modeleval.get_preconditioner_inverse(x, mu, g)
        # print('max(|<v,P^{-1}u> - <P^{-1}v,u>|) = %g'
        #      % _check_selfadjointness(Pinv, modeleval.inner_product
        #      )
        # check positive definiteness of P^{-1}
        # print('min(<u,P^{-1}u>) = %g'
        #      % _check_positivedefiniteness(Pinv, inner_product)
        #      )
    return


def _check_selfadjointness(operator, inner_product):
    N = operator.shape[0]
    num_samples = 100
    max_discrepancy = 0.0
    for k in range(num_samples):
        # Make the last one unconditionally real. This is for bordering.
        u = np.random.rand(N) + 1j * np.random.rand(N)
        u[-1] = u[-1].real
        # Make the last one unconditionally real. This is for bordering.
        v = np.random.rand(N) + 1j * np.random.rand(N)
        v[-1] = v[-1].real
        alpha = inner_product(v, operator * u)
        beta = inner_product(operator * v, u)
        max_discrepancy = max(max_discrepancy, abs(alpha - beta))
    return max_discrepancy


def _check_positivedefiniteness(operator, inner_product):
    N = operator.shape[0]
    num_samples = 1000
    min_val = np.inf
    for k in range(num_samples):
        u = np.random.rand(N, 1) + 1j * np.random.rand(N, 1)
        alpha = inner_product(u, operator * u)[0, 0]
        if abs(alpha.imag) > 1.0e-13:
            raise ValueError("Operator not self-adjoint? <u,Lu> = %e" % repr(alpha))
        min_val = min(min_val, alpha.real)
    return min_val


def _parse_input_arguments():
    """Parse input arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve the linearized Ginzburg--Landau problem."
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
    parser.add_argument("--mu", "-m", required=True, type=float, help="value of mu")
    parser.add_argument(
        "--preconditioner-type",
        "-p",
        choices=["none", "exact", "cycles"],
        default="none",
        help="preconditioner type (default: none)",
    )
    parser.add_argument(
        "--num-amg-cycles",
        "-a",
        type=int,
        default=1,
        help="number of AMG cycles (default: 1)",
    )
    parser.add_argument(
        "--bordering",
        "-b",
        action="store_true",
        default=False,
        help="use bordering (default: False)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    _main()
