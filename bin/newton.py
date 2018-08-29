# -*- coding: utf-8 -*-
#
'''
Solve nonlinear Schrödinger equations.
'''
import numpy as np

import pynosh.numerical_methods as nm
import pynosh.modelevaluator_nls as gpm
import pynosh.modelevaluator_bordering_constant as bme
import pynosh.yaml

import voropy
import krypy


def _main():
    args = _parse_input_arguments()

    ye = pynosh.yaml.YamlEmitter()
    ye.begin_doc()

    # read the mesh
    print('# Reading the mesh...')
    mesh, point_data, field_data = voropy.reader.read(args.filename)
    print('# done.')

    num_nodes = len(mesh.node_coords)

    # build the model evaluator
    if args.mu is not None:
        mu = args.mu
    elif 'mu' in field_data:
        mu = field_data['mu'][0]
    else:
        raise RuntimeError('Parameter ''mu'' not found.')

    if 'g' in field_data:
        g = field_data['g'][0]
    else:
        g = 1.0

    if 'V' in point_data:
        V = point_data['V']
    else:
        V = -np.ones(num_nodes)

    nls_modeleval = \
        gpm.NlsModelEvaluator(mesh,
                              V=V,
                              A=point_data['A'],
                              preconditioner_type=args.preconditioner_type,
                              num_amg_cycles=args.num_amg_cycles
                              )

    # initial guess
    if args.initial_name in point_data:
        psi0 = np.reshape(point_data[args.initial_name][:, 0]
                          + 1j * point_data[args.initial_name][:, 1],
                          (num_nodes, 1)
                          )
    else:
        psi0 = 1.0 * np.ones((num_nodes, 1), dtype=complex)
        #alpha = 0.3
        #kx = 2
        #ky = 0.5
        #for i, node in enumerate(mesh.node_coords):
            #psi0[i] = alpha * np.cos(kx * node[0]) * np.cos(ky * node[1])

    ye.begin_map()
    import sys
    import os
    import datetime
    ye.add_comment('Newton run with newton.py (%r, %s).'
                   % (os.uname()[1], datetime.datetime.now())
                   )
    ye.add_comment(' '.join(sys.argv))
    ye.add_key_value('num_unknowns', len(psi0))
    ye.add_key_value('filename', args.filename)
    ye.add_key_value('mu', mu)
    ye.add_key_value('g', g)

    ye.add_key_value('krylov', args.krylov_method)
    ye.add_key_value('preconditioner type', args.preconditioner_type)
    ye.add_key_value('ix deflation', args.defl_include_ix)
    ye.add_key_value('extra deflation', args.defl_num_ritz_vectors)
    ye.add_key_value('explicit residual', args.resexp)
    ye.add_key_value('bordering', args.bordering)

    if args.bordering:
        # Build bordered system.
        x0 = np.empty((num_nodes+1, 1), dtype=complex)
        x0[0:num_nodes] = psi0
        x0[-1] = 0.0
        # Use psi0 as initial bordering.
        modeleval = bme.BorderedModelEvaluator(nls_modeleval)
    else:
        x0 = psi0
        modeleval = nls_modeleval

    newton_out = my_newton(args, modeleval, x0, g, mu, yaml_emitter=ye)
    sol = newton_out['x'][0:num_nodes]

    ye.end_map()

    # energy of the state
    print('# Energy of the final state: %g.' % nls_modeleval.energy(sol))

    if args.solutionfile:
        modeleval.mesh.write(args.solutionfile,
                             point_data={'psi': sol,
                                         'A': point_data['A'],
                                         'V': V},
                             field_data={'mu': mu, 'g': g}
                             )
    return


class IxFactory(krypy.recycling.factories._DeflationVectorFactory):
    def __init__(self, x):
        self._x = x

    def get(self, solver):
        return 1j*self._x


def my_newton(args, modeleval, psi0, g, mu, yaml_emitter=None, debug=True):
    '''Solve with Newton.
    '''

    recycling_solver_kwargs = {
        'explicit_residual': args.resexp,
        'maxiter': 200
        }
    if args.krylov_method == 'cg':
        RecyclingSolver = krypy.recycling.RecyclingCg
    elif args.krylov_method == 'minres':
        RecyclingSolver = krypy.recycling.RecyclingMinres
    elif args.krylov_method == 'minresfo':
        RecyclingSolver = krypy.recycling.RecyclingMinres
        recycling_solver_kwargs['ortho'] = 'mgs'
    elif args.krylov_method == 'gmres':
        RecyclingSolver = krypy.recycling.RecyclingGmres
    else:
        raise ValueError('Unknown Krylov solver ''%s''.' % args.krylov_method)

    def vector_factory_generator(x):
        '''Generates the vector factory with i*x if requested.'''
        if not args.defl_include_ix and args.defl_num_ritz_vectors == 0:
            # no deflation
            return None
        else:
            ritz_factory = krypy.recycling.factories.RitzFactorySimple(
                n_vectors=args.defl_num_ritz_vectors,
                which='sm'
                )
            if args.defl_include_ix:
                return krypy.recycling.factories.UnionFactory(
                    [ritz_factory, IxFactory(x)])
            return ritz_factory

    # perform newton iteration
    yaml_emitter.add_key('Newton results')
    newton_out = nm.newton(psi0,
                           modeleval,
                           RecyclingSolver=RecyclingSolver,
                           recycling_solver_kwargs=recycling_solver_kwargs,
                           vector_factory_generator=vector_factory_generator,
                           nonlinear_tol=1.0e-10,
                           eta0=args.eta,
                           forcing_term='constant',
                           compute_f_extra_args={'g': g, 'mu': mu},
                           debug=debug,
                           yaml_emitter=yaml_emitter,
                           newton_maxiter=30
                           )
    yaml_emitter.add_comment('done.')
    #assert( newton_out['info'] == 0 )
    return newton_out


def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser(
        description='Find solutions to nonlinear Schrödinger equations.'
        )

    parser.add_argument('filename',
                        metavar='FILE',
                        type=str,
                        help='File containing the geometry and initial state'
                        )

    parser.add_argument('--solutionfile', '-s',
                        metavar='SOLUTION_FILE',
                        default=None,
                        const=None,
                        type=str,
                        help='Mesh file to store the final solution'
                        )

    parser.add_argument('--krylov-method', '-k',
                        choices=['cg', 'minres', 'minresfo', 'gmres'],
                        default='gmres',
                        help='which Krylov method to use (default: gmres)'
                        )

    parser.add_argument('--preconditioner-type', '-p',
                        choices=['none', 'exact', 'cycles'],
                        default='none',
                        help='preconditioner type (default: none)'
                        )

    parser.add_argument('--num-amg-cycles', '-a',
                        type=int,
                        default=1,
                        help='number of AMG cycles (default: 1)'
                        )

    parser.add_argument('--mu', '-m',
                        default=None,
                        type=float,
                        help='override value for mu from FILE (default: None)'
                        )

    parser.add_argument('--eta', '-e',
                        default=1e-10,
                        type=float,
                        help='linear solver tolerance (default: 1e-10)'
                        )

    parser.add_argument('--resexp', '-r',
                        action='store_true',
                        default=False,
                        help='compute explicit residual norms (default: False)'
                        )

    parser.add_argument('--defl-include-ix', '-x',
                        action='store_true',
                        default=False,
                        help='include 1j*x in the set of deflation vectors '
                             '(default: False)'
                        )

    parser.add_argument('--defl-num-ritz-vectors', '-n',
                        default=0,
                        type=int,
                        help='number of Ritz vectors for deflation '
                             '(default: 0)'
                        )

    parser.add_argument('--bordering', '-b',
                        default=False,
                        action='store_true',
                        help='use the bordered formulation to counter the '
                             'nullspace (default: false)'
                        )

    parser.add_argument('--initial-name', '-i',
                        metavar='INITIAL_NAME',
                        default='psi0',
                        type=str,
                        help='name of the initial guess stored in FILE '
                             '(default: psi0)'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    _main()
