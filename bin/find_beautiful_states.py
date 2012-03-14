#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve the Ginzburg--Landau equation.
'''
# ==============================================================================
import numpy as np

import pyginla.numerical_methods as nm
import pyginla.ginla_modelevaluator as gm
import voropy
# ==============================================================================
def _main():
    filename = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    mesh, point_data, field_data = voropy.read( filename )
    print 'done.'

    # build the model evaluator
    mu = 0.0
    ginla_modelval = gm.GinlaModelEvaluator(mesh, point_data['A'], mu)

    find_beautiful_states(ginla_modelval)
    return
# ==============================================================================
def find_beautiful_states( ginla_modelval ):
    '''Loop through a set of parameters/initial states and try to find
    starting points that (quickly) lead to "interesting looking" solutions.
    Such solutions are filtered out only by their energy at the moment.'''

    # Define search space.
    # Don't use Mu=0 as the preconditioner is singular for mu=0, psi=0.
    Mu = np.linspace(0.1, 1.0, 10)
    scalePsi0 = np.linspace(0.1, 1.0, 10)

    # initial guess
    num_nodes = len(ginla_modelval.mesh.node_coords)

    interesting_solutions = []

    k = 0
    for mu in Mu:
        ginla_modelval.set_parameter(mu)
        for alpha in scalePsi0:
            print 'mu = %g, alpha = %g' % (mu, alpha)
            psi0 = alpha * np.ones((num_nodes,1), dtype=complex)
            newton_out = newton(ginla_modelval, psi0, debug=False)
            if newton_out['info'] == 0:
                psi = newton_out['x']
                # Use the energy as a measure for ruling out boring states,
                # e.g., psi==0 or psi==1 overall.
                energy = ginla_modelval.energy( psi )
                print 'Energy of solution state: %g.' % energy
                if energy > -0.99 and energy < -0.01:
                    # Store the file as VTU such that ParaView can loop through
                    # and display them at once.
                    filename = 'sol' + repr(k).rjust(3,'0') + '.vtu'
                    print 'Interesting!',
                    # Check if we already stored that one.
                    already_found = False
                    for state in interesting_solutions:
                        # Synchronize the complex argument.
                        state *= np.exp(1j * (np.angle(psi[0]) - np.angle(state[0])))
                        diff = psi - state
                        if ginla_modelval.inner_product(diff, diff) < 1.0e-10:
                            already_found = True
                            break
                    if already_found:
                        print '-- But we already have that one.'
                    else:
                        interesting_solutions.append(psi)
                        print 'Storing in %s.' % filename
                        ginla_modelval.mesh.write(filename,
                                                  point_data={'psi': psi},
                                                  field_data={'mu': mu})
                        k += 1
            print

    return
# ==============================================================================
def newton(ginla_modelval, psi0, debug=True):
    '''Solve with Newton.
    '''

    print 'Performing Newton iteration...'
    # perform newton iteration
    newton_out = nm.newton(psi0,
                           ginla_modelval,
                           linear_solver = nm.minres,
                           linear_solver_maxiter = 500, #2*len(psi0),
                           linear_solver_extra_args = {},
                           nonlinear_tol = 1.0e-10,
                           forcing_term = 'constant', #'constant', #'type 2'
                           eta0 = 1.0e-10,
                           use_preconditioner = True,
                           deflate_ix = False,
                           num_deflation_vectors = 0,
                           debug=debug,
                           newton_maxiter = 10
                           )
    print ' done.'
    print 'Newton residuals:', newton_out['Newton residuals']

    return newton_out
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Find solutions to the Ginzburg--Landau equation.' )

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'ExodusII file containing the geometry and initial state'
                       )

    args = parser.parse_args()

    return args.filename
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
