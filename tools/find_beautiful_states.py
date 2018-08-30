# -*- coding: utf-8 -*-
#
"""Solve the Ginzburg--Landau equation.
"""
import numpy as np

import meshplex
import krypy

import pynosh.numerical_methods as nm
import pynosh.modelevaluator_nls as gm


def _main():
    args = _parse_input_arguments()

    # read the mesh
    print("Reading the mesh...")
    mesh, _, _, _ = meshplex.read(args.filename)
    print("done.")

    # Hardcode V, A
    n = mesh.node_coords.shape[0]
    X = mesh.node_coords.T
    A = 0.5 * np.column_stack([-X[1], X[0], np.zeros(n)])
    point_data = {"V": -np.ones(n), "A": A}

    # build the model evaluator
    modeleval = gm.NlsModelEvaluator(mesh, V=point_data["V"], A=point_data["A"])

    mu_range = np.linspace(args.mu_range[0], args.mu_range[1], args.num_parameter_steps)
    print("Looking for solutions for mu in")
    print(mu_range)
    print()
    find_beautiful_states(modeleval, mu_range, args.forcing_term)
    return


def find_beautiful_states(modeleval, mu_range, forcing_term, save_doubles=True):
    """Loop through a set of parameters/initial states and try to find
    starting points that (quickly) lead to "interesting looking" solutions.
    Such solutions are filtered out only by their energy at the moment.
    """

    # Define search space.
    # Don't use Mu=0 as the preconditioner is singular for mu=0, psi=0.
    Alpha = np.linspace(0.2, 1.0, 5)
    Frequencies = [0.0, 0.5, 1.0, 2.0]

    # Compile the search space.
    # If all nodes sit in x-y-plane, the frequency loop in z-direction can be omitted.
    if modeleval.mesh.node_coords.shape[1] == 2 or np.all(np.abs(modeleval.mesh.node_coords[:, 2]) < 1.0e-13):
        search_space_k = [(a, b) for a in Frequencies for b in Frequencies]
    elif modeleval.mesh.node_coords.shape[1] == 3:
        search_space_k = [
            (a, b, c) for a in Frequencies for b in Frequencies for c in Frequencies
        ]
    search_space = [(a, k) for a in reversed(Alpha) for k in search_space_k]

    solution_id = 0
    for mu in mu_range:
        # Reset the solutions each time the problem parameters change.
        found_solutions = []
        # Loop over initial states.
        for alpha, k in search_space:
            print("mu = {}; alpha = {}; k = {}".format(mu, alpha, k))
            # Set the intitial guess for Newton.
            if len(k) == 2:
                psi0 = (
                    alpha
                    * np.cos(k[0] * np.pi * modeleval.mesh.node_coords[:, 0])
                    * np.cos(k[1] * np.pi * modeleval.mesh.node_coords[:, 1])
                    + 1j * 0
                )
            elif len(k) == 3:
                psi0 = (
                    alpha
                    * np.cos(k[0] * np.pi * modeleval.mesh.node_coords[:, 0])
                    * np.cos(k[1] * np.pi * modeleval.mesh.node_coords[:, 1])
                    * np.cos(k[2] * np.pi * modeleval.mesh.node_coords[:, 2])
                    + 1j * 0
                )
            else:
                raise RuntimeError("Illegal k.")

            print("Performing Newton iteration...")
            linsolve_maxiter = 500  # 2*len(psi0)
            try:
                newton_out = nm.newton(
                    psi0[:, None],
                    modeleval,
                    nonlinear_tol=1.0e-10,
                    newton_maxiter=50,
                    compute_f_extra_args={"mu": mu, "g": 1.0},
                    eta0=1.0e-10,
                    forcing_term=forcing_term,
                )
            except krypy.utils.ConvergenceError:
                print("Krylov convergence failure. Skip.\n")
                continue
            print(" done.")

            num_krylov_iters = [
                len(resvec) for resvec in newton_out["linear relresvecs"]
            ]
            print("Num Krylov iterations:", num_krylov_iters)
            print("Newton residuals:", newton_out["Newton residuals"])
            if newton_out["info"] == 0:
                num_newton_iters = len(newton_out["linear relresvecs"])
                psi = newton_out["x"]
                # Use the energy as a measure for ruling out boring states such as
                # psi==0 or psi==1 overall.
                energy = modeleval.energy(psi)
                print("Energy of solution state: %g." % energy)
                if energy > -0.999 and energy < -0.001:
                    # Store the file as VTU such that ParaView can loop through and
                    # display them at once. For this, also be sure to keep the file name
                    # in the format 'interesting<-krylovfails03>-01414.vtu'.
                    filename = "interesting-"
                    num_krylov_fails = num_krylov_iters.count(linsolve_maxiter)
                    if num_krylov_fails > 0:
                        filename += "krylovfails{:02d}-".format(num_krylov_fails)
                    filename += "{:02d}{:03d}.vtu".format(num_newton_iters, solution_id)
                    print("Interesting state found for mu={}!".format(mu))
                    # Check if we already stored that one.
                    already_found = False
                    for state in found_solutions:
                        # Synchronize the complex argument.
                        state *= np.exp(1j * (np.angle(psi[0]) - np.angle(state[0])))
                        diff = psi - state
                        if modeleval.inner_product(diff, diff) < 1.0e-10:
                            already_found = True
                            break
                    if already_found and not save_doubles:
                        print("-- But we already have that one.")
                    else:
                        found_solutions.append(psi)
                        print("Storing in {}.".format(filename))
                        # if len(k) == 2:
                        #     function_string = (
                        #         "psi0(X) = %g * cos(%g*pi*x) * cos(%g*pi*y)"
                        #         % (alpha, k[0], k[1])
                        #     )
                        # elif len(k) == 3:
                        #     function_string = (
                        #         "psi0(X) = %g * cos(%g*pi*x) * cos(%g*pi*y) * cos(%g*pi*z)"
                        #         % (alpha, k[0], k[1], k[2])
                        #     )
                        # else:
                        #     raise RuntimeError("Illegal k.")
                        modeleval.mesh.write(
                            filename,
                            point_data={
                                "psi": np.column_stack([psi.real, psi.imag]),
                                "psi0": np.column_stack([psi0.real, psi0.imag]),
                                "V": modeleval._V,
                                "A": modeleval._raw_magnetic_vector_potential,
                            },
                            field_data={
                                "g": np.array(1.0),
                                "mu": np.array(mu),
                                # "psi0(X)": function_string,
                            },
                        )
                        solution_id += 1
            print()
    return


def _parse_input_arguments():
    """Parse input arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Find solutions to a nonlinear Schrödinger equation."
    )

    parser.add_argument(
        "filename",
        metavar="FILE",
        type=str,
        help="File containing the geometry and initial state",
    )

    parser.add_argument(
        "--mu-range",
        "-r",
        required=True,
        nargs=2,
        type=float,
        help="Range for mu sweep",
    )

    parser.add_argument(
        "--num-parameter-steps",
        "-n",
        metavar="NUMSTEPS",
        required=True,
        type=int,
        help="Number of steps in the parameter range",
    )

    parser.add_argument(
        "--forcing-term",
        "-f",
        required=True,
        choices=["constant", "type 1", "type 2"],
        type=str,
        help="Forcing term for Newton" "s method",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _main()
