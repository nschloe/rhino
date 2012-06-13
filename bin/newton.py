#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve nonlinear Schr\"odinger equations.
'''
# ==============================================================================
import numpy as np

import pyginla.numerical_methods as nm
import pyginla.gp_modelevaluator as gpm
import pyginla.yaml
import voropy
import matplotlib
# Use the AGG backend to make sure that we don't need
# $DISPLAY to plot something (to files).
matplotlib.use('agg')
import matplotlib.pyplot as pp
import matplotlib2tikz
# ==============================================================================
def _main():
    args = _parse_input_arguments()

    # read the mesh
    print '# Reading the mesh...',
    mesh, point_data, field_data = voropy.read( args.filename )
    print 'done.'

    # build the model evaluator
    if args.mu is not None:
        mu = args.mu
    elif 'mu' in field_data:
        mu = field_data['mu'][0]
    else:
        raise RuntimeError('Parameter ''mu'' not found in file or command line.')

    modeleval = gpm.GrossPitaevskiiModelEvaluator(mesh,
                                                  g = field_data['g'],
                                                  V = point_data['V'],
                                                  A = point_data['A'],
                                                  mu = mu,
                                                  preconditioner_type = args.preconditioner_type,
                                                  num_amg_cycles = args.num_amg_cycles)
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
    ye = pyginla.yaml.YamlEmitter()
    ye.begin_doc()
    ye.begin_map()
    ye.add_key('num_unknowns')
    ye.add_value(len(psi0))
    ye.add_key('filename')
    ye.add_value(args.filename)
    ye.add_key('mu')
    ye.add_value(mu)
    ye.add_key('g')
    ye.add_value(field_data['g'][0])

    ye.add_key('krylov')
    ye.add_value(args.krylov_method)
    ye.add_key('preconditioner type')
    ye.add_value(args.preconditioner_type)
    ye.add_key('ix deflation')
    ye.add_value(args.use_deflation)
    ye.add_key('extra deflation')
    ye.add_value(args.num_extra_defl_vectors)
    ye.add_key('explicit residual')
    ye.add_value(args.resexp)

    newton_out = my_newton(args, modeleval, psi0, yaml_emitter=ye)

    ye.end_map()

    # energy of the state
    print '# Energy of the final state: %g.' % modeleval.energy( newton_out['x'] )

    # Get output.
    #pp.subplot(121)
    multiplot_data_series( newton_out['linear relresvecs'] )
    pp.title('Krylov: %s    Prec: %r    ix-defl: %r    extra defl: %r    ExpRes: %r    Newton iters: %d' %
             (args.krylov_method, args.preconditioner_type, args.use_deflation,
              args.num_extra_defl_vectors, args.resexp, len(newton_out['Newton residuals'])-1)
             )
    # Plot Newton residuals.
    #pp.subplot(122)
    #pp.semilogy(newton_out['Newton residuals'])
    #pp.title('Newton residuals')

    # Write the info out to files.
    if args.imgfile:
        pp.savefig(args.imgfile)
    if args.tikzfile:
        matplotlib2tikz.save(args.tikzfile)
    if args.solutionfile:
        modeleval.mesh.write(args.solutionfile, {'psi': newton_out['x']})

    return
# ==============================================================================
def my_newton(args, modeleval, psi0, yaml_emitter=None, debug=True):
    '''Solve with Newton.
    '''

    lin_solve_args = {'explicit_residual': args.resexp}
    if args.krylov_method == 'cg':
        lin_solve = nm.cg
    elif args.krylov_method == 'minres':
        lin_solve = nm.minres
    elif args.krylov_method == 'minresfo':
        lin_solve = nm.minres
        lin_solve_args.update({'full_reortho': True})
    elif args.krylov_method == 'gmres':
        lin_solve = nm.gmres
    else:
        raise ValueError('Unknown Krylov solver ''%s''.' % args.krylov_method)

    defl = []
    if args.use_deflation:
        defl.append( lambda x: 1j*x )

    # perform newton iteration
    yaml_emitter.add_key('Newton results')
    newton_out = nm.newton(psi0,
                           modeleval,
                           linear_solver = lin_solve,
                           linear_solver_maxiter = 500, #2*len(psi0),
                           linear_solver_extra_args = lin_solve_args,
                           nonlinear_tol = 1.0e-10,
                           forcing_term = 'constant', #'constant', 'type1', 'type 2'
                           eta0 = args.eta,
                           deflation_generators = defl,
                           num_deflation_vectors = args.num_extra_defl_vectors,
                           debug=debug,
                           yaml_emitter = yaml_emitter,
                           newton_maxiter = 30
                           )
    yaml_emitter.add_comment('done.')
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

    parser.add_argument('--imgfile', '-i',
                        metavar = 'IMG_FILE',
                        required = True,
                        default = None,
                        const = None,
                        type = str,
                        help = 'Image file to store the results'
                        )

    parser.add_argument('--tikzfile', '-t',
                        metavar = 'TIKZ_FILE',
                        required = True,
                        default = None,
                        const = None,
                        type = str,
                        help = 'TikZ file to store the results'
                        )

    parser.add_argument('--solutionfile', '-s',
                        metavar = 'SOLUTION_FILE',
                        default = None,
                        const = None,
                        type = str,
                        help = 'Mesh file to store the final solution'
                        )

    parser.add_argument('--krylov-method', '-k',
                        choices = ['cg', 'minres', 'minresfo', 'gmres'],
                        default = 'gmres',
                        help    = 'which Krylov method to use (default: gmres)'
                        )

    parser.add_argument('--preconditioner-type', '-p',
                        choices = ['none', 'exact', 'cycles'],
                        default = 'none',
                        help    = 'preconditioner type (default: none)'
                        )

    parser.add_argument('--num-amg-cycles', '-a',
                        type = int,
                        default = 1,
                        help    = 'number of AMG cycles (default: 1)'
                        )

    parser.add_argument('--use-deflation', '-d',
                        action = 'store_true',
                        default = False,
                        help    = 'use deflation (default: False)'
                        )

    parser.add_argument('--mu', '-m',
                        default = None,
                        type = float,
                        help = 'override value for mu from FILE (default: None)'
                        )

    parser.add_argument('--eta', '-e',
                        default = 1e-10,
                        type = float,
                        help = 'override value for linear solver tolerance (default: 1e-10)'
                        )

    parser.add_argument('--resexp', '-r',
                        action = 'store_true',
                        default = False,
                        help = 'compute explicit residual norms (default: False)'
                        )

    parser.add_argument('--num-extra-defl-vectors', '-n',
                        default = 0,
                        type = int,
                        help = 'number of extra deflation vectors (default: 0)'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
