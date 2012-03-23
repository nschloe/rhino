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
    point_data['psi'] = point_data['psi'][:,0] \
                      + 1j * point_data['psi'][:,1]
    print 'done.'

    # build the model evaluator
    mu = 0.4
    ginla_modelval = gm.GinlaModelEvaluator(mesh, point_data['A'], mu)

    # initial guess
    num_nodes = len(mesh.node_coords)
    #psi0 = 1.0 * np.ones((num_nodes,1), dtype=complex)
    psi0 = np.reshape(point_data['psi'], (num_nodes,1))
    newton_out = newton(ginla_modelval, psi0)
    # write the solution to a file
    ginla_modelval.mesh.write('solution.e', {'psi': newton_out['x']})
    # energy of the state
    print 'Energy of the final state: %g.' % ginla_modelval.energy( newton_out['x'] )

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
                           deflation_generators = [ lambda x: 1j*x ],
                           num_deflation_vectors = 0,
                           debug=debug,
                           newton_maxiter = 10
                           )
    print ' done.'
    print 'Newton residuals:', newton_out['Newton residuals']
    #assert( newton_out['info'] == 0 )

    import matplotlib.pyplot as pp
    multiplot_data_series( newton_out['linear relresvecs'] )
    #pp.xlim([0,45])
    pp.show()
    #import matplotlib2tikz
    #matplotlib2tikz.save('minres-prec-defl.tex')

    #print 'Performing Poor man's continuation...'
    #nm.poor_mans_continuation( psi0,
                               #ginla_modelval,
                               #mu,
                               #initial_step_size = 1.0e-2,
                               #nonlinear_tol = 1.0e-8
                             #)

    return newton_out
# ==============================================================================
def multiplot_data_series( list_of_data_vectors ):
    '''Plot a list of data vectors with increasing black value.'''
    import matplotlib.pyplot as pp
    num_plots = len( list_of_data_vectors )
    for k, relresvec in enumerate(list_of_data_vectors):
        pp.semilogy(relresvec, color=str(1.0 - float(k+1)/num_plots))
    pp.xlabel('MINRES step')
    pp.ylabel('||r||/||b||')
    return
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

    #import cProfile
    #cProfile.run( '_main()', 'pfvm_profile.dat' )
# ==============================================================================
