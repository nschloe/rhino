#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve nonlinear Schr\"odinger equations.
'''
# ==============================================================================
import numpy as np

import pyginla.numerical_methods as nm
import pyginla.gp_modelevaluator as gpm
import voropy
import matplotlib.pyplot as pp
import matplotlib2tikz
# ==============================================================================
def _main():
    args = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    mesh, point_data, field_data = voropy.read( args.filename )
    print 'done.'

    # build the model evaluator
    if args.mu is not None:
        mu = args.mu
        print 'Using parameter  mu = %g.' % mu
    elif 'mu' in field_data:
        mu = field_data['mu']
        print 'Using  mu = %g  as found in file.' % mu
    else:
        raise RuntimeError('Parameter ''mu'' not found in file or command line.')

    modeleval = gpm.GrossPitaevskiiModelEvaluator(mesh,
                                                  g = field_data['g'],
                                                  V = point_data['V'],
                                                  A = point_data['A'],
                                                  mu = mu)
    #nls_modeleval = gpm.GrossPitaevskiiModelEvaluator(mesh, g=1.0)

    # initial guess
    num_nodes = len(mesh.node_coords)
    psi0Name = 'psi0'
    if psi0Name in point_data:
        psi0 = np.reshape(point_data[psi0Name][:,0] + 1j * point_data[psi0Name][:,1],
                          (num_nodes,1))
    else:
        psi0 = 1.0 * np.ones((num_nodes,1), dtype=complex)
        #alpha = 0.3
        #kx = 2
        #ky = 0.5
        #for i, node in enumerate(mesh.node_coords):
            #psi0[i] = alpha * np.cos(kx * node[0]) * np.cos(ky * node[1])
    newton_out = my_newton(args, modeleval, psi0)
    print 'Newton residuals:', newton_out['Newton residuals']

    # Get output.
    multiplot_data_series( newton_out['linear relresvecs'] )
    pp.title('Krylov: %s    Prec: %r    Defl: %r' %
             (args.krylov_method, args.use_preconditioner, args.use_deflation)
             )
    #pp.xlim([0,45])
    matplotlib2tikz.save(args.tikz)
    if args.show:
        pp.show()

    # write the solution to a file
    modeleval.mesh.write('solution.e', {'psi': newton_out['x']})
    # energy of the state
    print 'Energy of the final state: %g.' % modeleval.energy( newton_out['x'] )

    return
# ==============================================================================
def my_newton(args, modeleval, psi0, debug=True):
    '''Solve with Newton.
    '''

    if args.krylov_method == 'cg':
        lin_solve = nm.cg
    elif args.krylov_method == 'minres':
        lin_solve = nm.minres
    elif args.krylov_method == 'gmres':
        lin_solve = nm.gmres
    else:
        raise ValueError('Unknown Krylov solver ''%s''.' % args.krylov_method)

    defl = []
    if args.use_deflation:
        defl.append( lambda x: 1j*x )

    print 'Performing Newton iteration (dim=%d)...' % (2 * len(psi0))
    # perform newton iteration
    newton_out = nm.newton(psi0,
                           modeleval,
                           linear_solver = lin_solve,
                           linear_solver_maxiter = 1000, #2*len(psi0),
                           linear_solver_extra_args = {'explicit_residual': True},
                           nonlinear_tol = 1.0e-10,
                           forcing_term = 'constant', #'constant', 'type1', 'type 2'
                           eta0 = 1.0e-15,
                           use_preconditioner = args.use_preconditioner,
                           deflation_generators = defl,
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

    parser = argparse.ArgumentParser( description = 'Find solutions to nonlinear Schrödinger equations.' )

    parser.add_argument('filename',
                        metavar = 'FILE',
                        type    = str,
                        help    = 'Mesh file containing the geometry and initial state'
                        )

    parser.add_argument('--tikz', '-t',
                        metavar = 'TIKZFILE',
                        required = True,
                        default = None,
                        const = None,
                        type = str,
                        help    = 'TikZ file to store the results'
                        )

    parser.add_argument('--krylov-method', '-k',
                        choices = ['cg', 'minres', 'gmres'],
                        default = 'minres',
                        help    = 'which Krylov method to use (default: False)'
                        )

    parser.add_argument('--use-preconditioner', '-p',
                        action = 'store_true',
                        default = False,
                        help    = 'use preconditioner (default: False)'
                        )

    parser.add_argument('--use-deflation', '-d',
                        action = 'store_true',
                        default = False,
                        help    = 'use deflation (default: False)'
                        )

    parser.add_argument('--show', '-s',
                        action = 'store_true',
                        default = False,
                        help    = 'show the relative residuals of each linear iteration (default: False)'
                        )

    parser.add_argument('--mu', '-m',
                        default = None,
                        type = float,
                        help = 'override value for mu from FILE (default: None)'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
