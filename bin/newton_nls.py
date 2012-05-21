#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve the Ginzburg--Landau equation.
'''
# ==============================================================================
import numpy as np

import pyginla.numerical_methods as nm
import pyginla.nls_modelevaluator as nls
import voropy
# ==============================================================================
def _main():
    args = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    mesh, point_data, field_data = voropy.read( args.filename )
    print 'done.'

    # build the model evaluator
    if args.kappa is not None:
        kappa = args.kappa
        print 'Using parameter  kappa = %g.' % kappa
    elif 'mu' in field_data:
        kappa = field_data['kappa']
        print 'Using  kappa = %g  as found in file.' % kappa
    else:
        raise RuntimeError('Parameter ''kappa'' not found in file or command line.')

    model_eval = nls.NlsModelEvaluator(mesh, kappa)

    # initial guess
    num_nodes = len(mesh.node_coords)
    if 'psi' in point_data:
        point_data['psi'] = point_data['psi'][:,0] \
                          + 1j * point_data['psi'][:,1]
        psi0 = np.reshape(point_data['psi'], (num_nodes,1))
    else:
        psi0 = 1.0 * np.ones((num_nodes,1), dtype=complex)
    newton_out = newton(model_eval, psi0)
    print 'Newton residuals:', newton_out['Newton residuals']

    if args.show:
        import matplotlib.pyplot as pp
        multiplot_data_series( newton_out['linear relresvecs'] )
        #pp.xlim([0,45])
        pp.show()

    #import matplotlib2tikz
    #matplotlib2tikz.save('minres-prec-defl.tex')

    # write the solution to a file
    model_eval.mesh.write('solution.e', {'psi': newton_out['x']})
    # energy of the state
    print 'Energy of the final state: %g.' % model_eval.energy( newton_out['x'] )

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
                           linear_solver_maxiter = 1000, #2*len(psi0),
                           linear_solver_extra_args = {},
                           nonlinear_tol = 1.0e-10,
                           forcing_term = 'constant', #'constant', 'type1', 'type 2'
                           eta0 = 1.0e-10,
                           use_preconditioner = True,
                           deflation_generators = [ lambda x: 1j*x ],
                           num_deflation_vectors = 0,
                           debug=debug,
                           newton_maxiter = 30
                           )
    print ' done.'
    #assert( newton_out['info'] == 0 )

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

    parser.add_argument('filename',
                        metavar = 'FILE',
                        type    = str,
                        help    = 'ExodusII file containing the geometry and initial state'
                        )

    parser.add_argument('--show', '-s',
                        action = 'store_true',
                        default = False,
                        help    = 'show the relative residuals of each linear iteration (default: False)'
                        )

    parser.add_argument('--kappa', '-k',
                        default = None,
                        type = float,
                        help = 'override value for kappa from FILE (default: None)'
                        )


    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
