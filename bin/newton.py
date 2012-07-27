#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve nonlinear Schr\"odinger equations.
'''
# ==============================================================================
import numpy as np

import pynosh.numerical_methods as nm
import pynosh.nls_modelevaluator as gpm
import pynosh.bordered_modelevaluator as bme
import pynosh.yaml
import voropy
# ==============================================================================
def _main():
    args = _parse_input_arguments()

    ye = pynosh.yaml.YamlEmitter()
    ye.begin_doc()

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

    num_nodes = len(mesh.node_coords)

    modeleval = gpm.NlsModelEvaluator(mesh,
                                      g = field_data['g'],
                                      V = point_data['V'],
                                      A = point_data['A'],
                                      mu = mu,
                                      preconditioner_type = args.preconditioner_type,
                                      num_amg_cycles = args.num_amg_cycles)

    ## check out self-adjointness
    #n = num_nodes
    #x_test = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    #J = modeleval.get_jacobian(x_test)
    #for k in xrange(5):
    #    phi = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    #    psi = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    #    Jphi = J*phi
    #    Jpsi = J*psi
    #    print modeleval.inner_product(phi,Jpsi) \
    #        - modeleval.inner_product(Jphi, psi)

    #print
    #x_test = np.empty((n+1,1),dtype=complex)
    #x_test[0:n] = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    #x_test[n] = np.random.rand(1)
    #J = bordered_modeleval.get_jacobian(x_test)
    #for k in xrange(5):
    #    phi = np.empty((n+1,1),dtype=complex)
    #    phi[0:n] = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    #    phi[n] = np.random.rand(1)
    #    psi = np.empty((n+1,1),dtype=complex)
    #    psi[0:n] = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    #    psi[n] = np.random.rand(1)
    #    Jphi = J*phi
    #    Jpsi = J*psi
    #    print bordered_modeleval.inner_product(phi, Jpsi) \
    #        - bordered_modeleval.inner_product(Jphi, psi)
    #dasdas

    # initial guess
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

    ye.begin_map()
    import sys, os, datetime
    ye.add_comment('Newton run with newton.py (%r, %s).' % (os.uname()[1], datetime.datetime.now()))
    ye.add_comment(' '.join(sys.argv))
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
    ye.add_key('bordering')
    ye.add_value(args.bordering)

    if args.bordering:
        # Build bordered system.
        x0 = np.empty((num_nodes+1,1), dtype=complex)
        x0[0:num_nodes] = psi0
        x0[-1] = 0.0
        # Use psi0 as initial bordering.
        bordered_modeleval = bme.BorderedModelEvaluator(modeleval, psi0)
        newton_out = my_newton(args, bordered_modeleval, x0, yaml_emitter=ye)
        sol = newton_out['x'][0:num_nodes]
    else:
        newton_out = my_newton(args, modeleval, psi0, yaml_emitter=ye)
        sol = newton_out['x']

    ye.end_map()

    # energy of the state
    print '# Energy of the final state: %g.' % modeleval.energy(sol)

    if args.solutionfile:
        modeleval.mesh.write(args.solutionfile, {'psi': sol})

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
                           linear_solver_maxiter = 1000, #2*len(psi0),
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

    parser.add_argument('--bordering', '-b',
                        default = False,
                        action = 'store_true',
                        help = 'use the bordered formulation to counter the nullspace, does not work with preconditioner (default: false)'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
