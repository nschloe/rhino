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
    ginla_modeleval = gm.GinlaModelEvaluator(mesh, point_data['A'], mu)

    find_beautiful_states(ginla_modeleval)
    return
# ==============================================================================
def find_beautiful_states( ginla_modeleval ):
    '''Loop through a set of parameters/initial states and try to find
    starting points that (quickly) lead to "interesting looking" solutions.
    Such solutions are filtered out only by their energy at the moment.'''

    # Define search space.
    # Don't use Mu=0 as the preconditioner is singular for mu=0, psi=0.
    Mu = np.linspace(0.1, 1.0, 10)
    Alpha = np.linspace(0.1, 1.0, 10)
    Wavelengths = [0, 0.25, 0.5, 1, 2, 4]

    # initial guess
    num_nodes = len(ginla_modeleval.mesh.node_coords)

    interesting_solutions = []

    solution_id = 0
    psi0 = np.empty((num_nodes,1), dtype=complex)
    for mu, alpha, k1, k2 in ((a,b,c,d) for a in Mu for b in Alpha for c in Wavelengths for d in Wavelengths):
        ginla_modeleval.set_parameter(mu)
        print 'mu = %g, alpha = %g, k1 = %g, k2 = %g' % (mu, alpha, k1, k2)
        # Set the intitial guess for Newton.
        #psi0 = alpha * np.ones((num_nodes,1), dtype=complex)
        for i, node in enumerate(ginla_modeleval.mesh.node_coords):
            psi0[i] = alpha * np.cos(k1 * node[0]) * np.cos(k2 * node[1])
        newton_out = newton(ginla_modeleval, psi0, debug=False)
        print 'Num MINRES iterations:', [len(resvec) for resvec in newton_out['linear relresvecs']]
        print 'Newton residuals:', newton_out['Newton residuals']
        if newton_out['info'] == 0:
            num_newton_iters = len(newton_out['linear relresvecs'])
            psi = newton_out['x']
            # Use the energy as a measure for ruling out boring states,
            # e.g., psi==0 or psi==1 overall.
            energy = ginla_modeleval.energy( psi )
            print 'Energy of solution state: %g.' % energy
            if energy > -0.999 and energy < -0.001:
                # Store the file as VTU such that ParaView can loop through
                # and display them at once.
                filename = 'interesting-newton' + repr(num_newton_iters).rjust(2,'0') \
                         + '-id' + repr(solution_id).rjust(3,'0') \
                         + '.e'
                print 'Interesting!',
                # Check if we already stored that one.
                already_found = False
                for state in interesting_solutions:
                    # Synchronize the complex argument.
                    state *= np.exp(1j * (np.angle(psi[0]) - np.angle(state[0])))
                    diff = psi - state
                    if ginla_modeleval.inner_product(diff, diff) < 1.0e-10:
                        already_found = True
                        break
                if already_found:
                    print '-- But we already have that one.'
                else:
                    interesting_solutions.append(psi)
                    print 'Storing in %s.' % filename
                    ginla_modeleval.mesh.write(filename,
                                              point_data={'psi': psi, 'A': ginla_modeleval._raw_magnetic_vector_potential},
                                              field_data={'mu': mu, 'alpha': alpha, 'k1': k1, 'k2': k2})
                    solution_id += 1
        print

    return
# ==============================================================================
def newton(ginla_modeleval, psi0, debug=True):
    '''Solve with Newton.
    '''

    print 'Performing Newton iteration...'
    # perform newton iteration
    newton_out = nm.newton(psi0,
                           ginla_modeleval,
                           linear_solver = nm.minres,
                           linear_solver_maxiter = 500, #2*len(psi0),
                           linear_solver_extra_args = {},
                           nonlinear_tol = 1.0e-10,
                           forcing_term = 'constant', #'constant', #'type 2'
                           eta0 = 1.0e-10,
                           use_preconditioner = True,
                           deflation_generators = [ lambda x: 1j*x ],
                           num_deflation_vectors = 0,
                           debug=debug,
                           newton_maxiter = 50
                           )
    print ' done.'

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
