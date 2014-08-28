#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of PyNosh.
#
#  PyNosh is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PyNosh is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyNosh.  If not, see <http://www.gnu.org/licenses/>.
'''Solve the Ginzburg--Landau equation.
'''

import numpy as np

import pynosh.numerical_methods as nm
import pynosh.modelevaluator_nls as gm
import voropy


def _main():
    args = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    mesh, point_data, field_data = voropy.read(args.filename)
    print 'done.'

    # build the model evaluator
    modeleval = gm.NlsModelEvaluator(mesh,
                                     g=field_data['g'],
                                     V=point_data['V'],
                                     A=point_data['A'],
                                     mu=field_data['mu']
                                     )

    param_range = np.linspace(args.parameter_range[0],
                              args.parameter_range[1],
                              args.num_parameter_steps)
    print 'Looking for solutions for %s in' % args.parameter
    print param_range
    print
    find_beautiful_states(modeleval, args.parameter,
                          param_range, args.forcing_term
                          )
    return


def find_beautiful_states(modeleval,
                          param_name,
                          param_range,
                          forcing_term,
                          save_doubles=True
                          ):
    '''Loop through a set of parameters/initial states and try to find
    starting points that (quickly) lead to "interesting looking" solutions.
    Such solutions are filtered out only by their energy at the moment.
    '''

    # Define search space.
    # Don't use Mu=0 as the preconditioner is singular for mu=0, psi=0.
    Alpha = np.linspace(0.2, 1.0, 5)
    Frequencies = [0.0, 0.5, 1.0, 2.0]

    # Compile the search space.
    # If all nodes sit in x-y-plane, the frequency loop in z-direction can be
    # omitted.
    if modeleval.mesh.node_coords.shape[1] == 2:
        search_space_k = [(a, b) for a in Frequencies for b in Frequencies]
    elif modeleval.mesh.node_coords.shape[1] == 3:
        search_space_k = [(a, b, c)
                          for a in Frequencies
                          for b in Frequencies
                          for c in Frequencies
                          ]
    search_space = [(a, k) for a in reversed(Alpha) for k in search_space_k]

    solution_id = 0
    for p in param_range:
        # Reset the solutions each time the problem parameters change.
        found_solutions = []
        # Loop over initial states.
        for alpha, k in search_space:
            modeleval.set_parameter(param_name, p)
            print '%s = %g; alpha = %g; k = %s' % (param_name, p, alpha, k,)
            # Set the intitial guess for Newton.
            if len(k) == 2:
                psi0 = alpha \
                    * np.cos(k[0] * np.pi * modeleval.mesh.node_coords[:, 0]) \
                    * np.cos(k[1] * np.pi * modeleval.mesh.node_coords[:, 1]) \
                    + 1j * 0
            elif len(k) == 3:
                psi0 = alpha \
                    * np.cos(k[0] * np.pi * modeleval.mesh.node_coords[:, 0]) \
                    * np.cos(k[1] * np.pi * modeleval.mesh.node_coords[:, 1]) \
                    * np.cos(k[2] * np.pi * modeleval.mesh.node_coords[:, 2]) \
                    + 1j * 0
            else:
                raise RuntimeError('Illegal k.')

            print 'Performing Newton iteration...'
            linsolve_maxiter = 500  # 2*len(psi0)
            # perform newton iteration
            newton_out = nm.newton(psi0[:, None],
                                   modeleval,
                                   linear_solver=nm.minres,
                                   linear_solver_maxiter=linsolve_maxiter,
                                   linear_solver_extra_args={},
                                   nonlinear_tol=1.0e-10,
                                   forcing_term=forcing_term,
                                   eta0=1.0e-10,
                                   use_preconditioner=True,
                                   deflation_generators=[lambda x: 1j*x],
                                   num_deflation_vectors=0,
                                   debug=True,
                                   newton_maxiter=50
                                   )
            print ' done.'

            num_krylov_iters = \
                [len(resvec) for resvec in newton_out['linear relresvecs']]
            print 'Num Krylov iterations:', num_krylov_iters
            print 'Newton residuals:', newton_out['Newton residuals']
            if newton_out['info'] == 0:
                num_newton_iters = len(newton_out['linear relresvecs'])
                psi = newton_out['x']
                # Use the energy as a measure for ruling out boring states
                # such as psi==0 or psi==1 overall.
                energy = modeleval.energy(psi)
                print 'Energy of solution state: %g.' % energy
                if energy > -0.999 and energy < -0.001:
                    # Store the file as VTU such that ParaView can loop through
                    # and display them at once. For this, also be sure to keep
                    # the file name in the format
                    # 'interesting<-krylovfails03>-01414.vtu'.
                    filename = 'interesting-'
                    num_krylov_fails = num_krylov_iters.count(linsolve_maxiter)
                    if num_krylov_fails > 0:
                        filename += 'krylovfails%s-' \
                            % repr(num_krylov_fails).rjust(2, '0')
                    filename += repr(num_newton_iters).rjust(2, '0') \
                        + repr(solution_id).rjust(3, '0') \
                        + '.vtu'
                    print('Interesting state found for %s=%g!'
                          % (param_name, p),
                          )
                    # Check if we already stored that one.
                    already_found = False
                    for state in found_solutions:
                        # Synchronize the complex argument.
                        state *= np.exp(1j *
                            (np.angle(psi[0]) - np.angle(state[0]))
                            )
                        diff = psi - state
                        if modeleval.inner_product(diff, diff) < 1.0e-10:
                            already_found = True
                            break
                    if already_found and not save_doubles:
                        print '-- But we already have that one.'
                    else:
                        found_solutions.append(psi)
                        print 'Storing in %s.' % filename
                        if len(k) == 2:
                            function_string = 'psi0(X) = %g * cos(%g*pi*x) * cos(%g*pi*y)' % (alpha, k[0], k[1])
                        elif len(k) == 3:
                            function_string = 'psi0(X) = %g * cos(%g*pi*x) * cos(%g*pi*y) * cos(%g*pi*z)' % (alpha, k[0], k[1], k[2])
                        else:
                            raise RuntimeError('Illegal k.')
                        modeleval.mesh.write(
                            filename,
                            point_data={'psi': psi,
                                        'psi0': psi0,
                                        'V': modeleval._V,
                                        'A': modeleval._raw_magnetic_vector_potential
                                        },
                            field_data={'g': modeleval._g,
                                        'mu': modeleval.mu,
                                        'psi0(X)': function_string
                                        }
                            )
                        solution_id += 1
            print
    return


def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Find solutions to a nonlinear Schrödinger equation.')

    parser.add_argument('filename',
                        metavar='FILE',
                        type=str,
                        help='File containing the geometry and initial state'
                        )

    parser.add_argument('--parameter', '-p',
                        required=True,
                        choices=['mu', 'g'],
                        type=str,
                        help='Which parameter to sweep'
                        )

    parser.add_argument('--parameter-range', '-r',
                        required=True,
                        nargs=2,
                        type=float,
                        help='Range for parameter sweep'
                        )

    parser.add_argument('--num-parameter-steps', '-n',
                        metavar='NUMSTEPS',
                        required=True,
                        type=int,
                        help='Number of steps in the parameter range'
                        )

    parser.add_argument('--forcing-term', '-f',
                        required=True,
                        choices=['constant', 'type 1', 'type 2'],
                        type=str,
                        help='Forcing term for Newton''s method'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    _main()
