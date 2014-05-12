#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of pynosh.
#
#  pynosh is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  pynosh is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with pynosh.  If not, see <http://www.gnu.org/licenses/>.
'''
Solve the linearized Ginzburg--Landau problem.
'''
# ==============================================================================
from scipy.sparse.linalg import LinearOperator
import time
import numpy as np
import cmath
import matplotlib.pyplot as pp
#from matplotlib import rc
#rc( 'text', usetex = True )
#rc( 'font', family = 'serif' )

import voropy
import pynosh.modelevaluator_nls
import pynosh.modelevaluator_bordering_constant
#import pynosh.preconditioners
import pynosh.numerical_methods as nm
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # Setting the default file_mvp.
    if args.file_mvp is None:
        args.file_mvp = args.filename

    # Read the magnetic vector potential.
    mesh, point_data, field_data = voropy.read(args.file_mvp)

    # build the model evaluator
    print('Creating model evaluator...',)
    start = time.time()
    num_coords = len( mesh.node_coords )
    nls_modeleval = pynosh.modelevaluator_nls.NlsModelEvaluator(mesh,
                                                                V=-np.ones(num_coords),
                                                                A=point_data['A'],
                                                                preconditioner_type=args.preconditioner_type,
                                                                num_amg_cycles = args.num_amg_cycles)
    if args.bordering:
        modeleval = pynosh.bordered_modelevaluator.BorderedModelEvaluator(nls_modeleval)
    else:
        modeleval = nls_modeleval
    end = time.time()
    print('done. (%gs)' % (end - start))

    # Run through all time steps.
    assert len(args.timesteps) == len(args.mu), \
           'There must be as many time steps as mus (%d != %d).' \
           % (len(args.timesteps), len(args.mu))
    for (timestep, mu) in zip(args.timesteps, args.mu):
        relresvec = _solve_system(modeleval, args.filename, timestep, mu, args)
        print('relresvec:')
        print(relresvec)
        print('num iters:', len(relresvec)-1)
        if args.show_relres:
            pp.semilogy(relresvec, 'k')
            pp.show()

    return
# ==============================================================================
def _solve_system(modeleval, filename, timestep, mu, args):
    # read the mesh
    print('Reading current psi...',)
    start = time.time()
    mesh, point_data, field_data = voropy.read(filename,
                                               timestep=timestep
                                               )
    total = time.time() - start
    print('done (%gs).' % total)

    num_coords = len( mesh.node_coords )
    # --------------------------------------------------------------------------
    # set psi at which to create the Jacobian
    print('Creating initial guess and right-hand side...',)
    start = time.time()
    current_psi = (point_data['psi'][:,0] + 1j * point_data['psi'][:,1]).reshape(-1,1)

    # Perturb a bit.
    eps = 1.0e-10
    perturbation = eps * (np.random.rand(num_coords) + 1j * np.random.rand(num_coords))
    current_psi += perturbation.reshape(current_psi.shape)

    if args.bordering:
        phi0 = np.zeros((num_coords+1,1), dtype=complex)
        # right hand side
        x = np.empty((num_coords+1,1), dtype=complex)
        x[0:num_coords] = current_psi
        x[num_coords] = 0.0
    else:
        # create right hand side and initial guess
        phi0 = np.zeros( (num_coords,1), dtype=complex )
        x = current_psi
    # right hand side
    #rhs = np.ones( (num_coords,1), dtype=complex )

    #rhs = np.empty( num_coords, dtype = complex )
    #radius = np.random.rand( num_coords )
    #arg    = np.random.rand( num_coords ) * 2.0 * cmath.pi
    #for k in range( num_coords ):
        #rhs[ k ] = cmath.rect(radius[k], arg[k])
    rhs = modeleval.compute_f(x=x, mu=mu, g=1.0)
    end = time.time()
    print('done. (%gs)' % (end - start))
    print('||rhs|| = %g' % np.sqrt(modeleval.inner_product(rhs, rhs)))
    # --------------------------------------------------------------------------
    # create the linear operator
    print('Getting Jacobian...',)
    start_time = time.clock()
    jacobian = modeleval.get_jacobian(x=x, mu=mu, g=1.0)
    end_time = time.clock()
    print('done. (%gs)' % (end_time - start_time))

    # create precondictioner object
    print('Getting preconditioner...',)
    start_time = time.clock()
    prec = modeleval.get_preconditioner_inverse(x=x, mu=mu, g=1.0)
    end_time = time.clock()
    print('done. (%gs)' % (end_time - start_time))

    # --------------------------------------------------------------------------
    # Get reference solution
    #print 'Get reference solution (dim = %d)...' % (2*num_coords),
    #start_time = time.clock()
    #ref_sol, info, relresvec, errorvec = nm.minres_wrap( jacobian, rhs,
                                          #x0 = phi0,
                                          #tol = 1.0e-14,
                                          #M = prec,
                                          #inner_product = modeleval.inner_product,
                                          #explicit_residual = True
                                        #)
    #end_time = time.clock()
    #if info == 0:
        #print 'success!',
    #else:
        #print 'no convergence.',
    #print ' (', end_time - start_time, 's,', len(relresvec)-1 ,' iters).'

    if args.use_deflation:
        W = 1j * x
        AW = jacobian * W
        P, x0new = nm.get_projection(W, AW, rhs, phi0,
                                     inner_product = modeleval.inner_product
                                     )
    else:
        #AW = np.zeros((len(current_psi),1), dtype=np.complex)
        P = None
        x0new = phi0

    if args.krylov_method == 'cg':
        lin_solve = nm.cg
    elif args.krylov_method == 'minres':
        lin_solve = nm.minres
    #elif args.krylov_method == 'minresfo':
        #lin_solve = nm.minres
        #lin_solve_args.update({'full_reortho': True})
    elif args.krylov_method == 'gmres':
        lin_solve = nm.gmres
    else:
        raise ValueError('Unknown Krylov solver ''%s''.' % args.krylov_method)

    print('Solving the system (len(x) = %d, bordering: %r)...' % (len(x), args.bordering),)
    start_time = time.clock()
    timer = False
    out = lin_solve(jacobian, rhs,
                    x0new,
                    tol = args.tolerance,
                    Mr = P,
                    M = prec,
                    #maxiter = 2*num_coords,
                    maxiter = 500,
                    inner_product = modeleval.inner_product,
                    explicit_residual = True,
                    #timer=timer
                    #exact_solution = ref_sol
                    )
    end_time = time.clock()
    print('done. (%gs)' % (end_time - start_time))
    print('(%d,%d)' % (2*num_coords, len(out['relresvec'])-1))


    # compute actual residual
    #res = rhs - jacobian * out['xk']
    #print '||b-Ax|| = %g' % np.sqrt(modeleval.inner_product(res, res))

    if timer:
        # pretty-print timings
        print(' '*22 + 'sum'.rjust(14) + 'mean'.rjust(14) + 'min'.rjust(14) + 'std dev'.rjust(14))
        for key, item in out['times'].items():
            print('\'%s\': %12g  %12g  %12g  %12g'
                 % (key.ljust(20), item.sum(), item.mean(), item.min(), item.std()))

    # Get the number of MG cycles.
    # 'modeleval.num_cycles' contains the number of MG cycles executed
    # for all AMG calls run.
    # In nm.minres, two calls to the precondictioner are done prior to the
    # actual iteration for the normalization of the residuals.
    # With explicit_residual=True, *two* calls to the preconditioner are done
    # in each iteration.
    # What we would like to have here is the number of V-cycles done per loop
    # when explicit_residual=False. Also, forget about the precondictioner
    # calls for the initialization.
    # Hence, cut of the first two and replace it by 0, and out of the
    # remainder take every other one.
    #nc = [0] + modeleval.tot_amg_cycles[2::2]
    #nc_cumsum = np.cumsum(nc)
    #pp.semilogy(nc_cumsum, out['relresvec'], color='0.0')
    #pp.show()
    #import matplotlib2tikz
    #matplotlib2tikz.save('cycle10.tex')

    #matplotlib2tikz.save('inf.tex')

    return out['relresvec']
# ==============================================================================
def _run_one_mu( modeleval,
                 precs,
                 jacobian,
                 rhs,
                 psi0,
                 test_preconditioners
               ):
    # --------------------------------------------------------------------------
    # build the kinetic energy operator
    print('Building the KEO...')
    start_time = time.clock()
    modeleval._assemble_kinetic_energy_operator()
    end_time = time.clock()
    print('done.', end_time - start_time)
    # --------------------------------------------------------------------------
    # Run the preconditioners and gather the relative residuals.
    relresvecs = _run_preconditioners( jacobian,
                                       rhs,
                                       psi0,
                                       test_preconditioners
                                     )

    # Plot the relative residuals.
    _plot_relresvecs( test_preconditioners, relresvecs )
    matplotlib2tikz.save( 'one-mu.tikz',
                          figurewidth = '\\figurewidth',
                          figureheight = '\\figureheight'
                        )
    pp.show()
    return
# ==============================================================================
def _run_along_top( modeleval,
                    precs,
                    jacobian,
                    rhs,
                    psi0,
                    test_preconditioners
                  ):

    num_coords = len( psi0 )

    # prepare the range of mus
    mu_min = 0.0
    mu_max = 5.0
    num_steps = 1001
    mus = np.linspace( mu_min, mu_max, num = num_steps )

    num_iterations = {}
    for prec in test_preconditioners:
        num_iterations[ prec['name'] ] = []

    # run over the mu and solve the equation systems
    for mu in mus:
        print()
        print(' mu =', mu)
        # ----------------------------------------------------------------------
        # build the kinetic energy operator
        modeleval.set_parameter( mu )
        precs.set_parameter( mu )
        print('Building the KEO...')
        start_time = time.clock()
        modeleval._assemble_kinetic_energy_operator()
        end_time = time.clock()
        print('done. (', end_time - start_time, 's).')
        # ----------------------------------------------------------------------
        # Run the preconditioners and gather the relative residuals.
        relresvecs = _run_preconditioners( jacobian,
                                           rhs,
                                           psi0,
                                           test_preconditioners
                                         )
        # ----------------------------------------------------------------------
        # append the number of iterations to the data
        for prec in test_preconditioners:
            num_iterations[ prec['name'] ].append(
                                             len( relresvecs[prec['name']] ) - 1
                                                 )
        # ----------------------------------------------------------------------

    # plot them all
    for name, num_iteration in num_iterations.iteritems():
        pp.plot( mus,
                 num_iteration,
                 label = name
               )

    # add title and so forth
    pp.title( 'CG convergence for $J$' )
    pp.xlabel( '$\mu$' )
    pp.ylabel( 'Number of iterations till $<10^{-10}$' )
    pp.legend()

    matplotlib2tikz.save( 'toprun.tikz',
                          figurewidth = '\\figurewidth',
                          figureheight = '\\figureheight'
                        )
    pp.show()

    return
# ==============================================================================
def _run_different_meshes( modeleval,
                           precs
                         ):
    mesh_files = [
                   'states/rectangle10.vtu',
                   'states/rectangle20.vtu',
                   'states/rectangle30.vtu',
                   'states/rectangle40.vtu',
                   'states/rectangle50.vtu',
                   'states/rectangle60.vtu',
                   'states/rectangle70.vtu',
                   'states/rectangle80.vtu',
                   'states/rectangle90.vtu',
                   'states/rectangle100.vtu',
                   #'states/rectangle110.vtu',
                   #'states/rectangle120.vtu',
                   #'states/rectangle130.vtu',
                   #'states/rectangle140.vtu',
                   #'states/rectangle150.vtu',
                   #'states/rectangle160.vtu',
                   #'states/rectangle170.vtu',
                   #'states/rectangle180.vtu',
                   #'states/rectangle190.vtu',
                   #'states/rectangle200.vtu'
                 ]

    mu = 1.0e-0
    modeleval.set_parameter( mu )
    precs.set_parameter( mu )

    # --------------------------------------------------------------------------
    # loop over the meshes and compute
    nums_unknowns = []

    num_iterations = {}

    for mesh_file in mesh_files:
        # ----------------------------------------------------------------------
        # read and set the mesh
        print()
        print('Reading the mesh...')
        try:
            mesh = vtkio.read_mesh( mesh_file )
        except AttributeError:
            raise IOError( 'Could not read from file ', mesh_file, '.' )
        print(' done.')
        modeleval.set_mesh( mesh )
        precs.set_mesh( mesh )
        # ----------------------------------------------------------------------
        # recreate all the objects necessary to perform the precondictioner run
        num_coords = len( mesh.nodes )

        nums_unknowns.append( num_coords )

        # create the linear operator
        jacobian = LinearOperator( (num_coords, num_coords),
                                         matvec = modeleval.compute_jacobian,
                                         dtype = complex
                                       )

        # set psi at which to create the Jacobian
        # generate random numbers within the unit circle
        radius = np.random.rand( num_coords )
        arg    = np.random.rand( num_coords )
        current_psi = np.zeros( num_coords,
                                dtype = complex
                              )
        for k in range( num_coords ):
            current_psi[ k ] = cmath.rect(radius[k], arg[k])
        modeleval.set_current_psi( current_psi )

        # create right hand side and initial guess
        rhs  =  np.random.rand( num_coords ) \
            + 1j * np.random.rand( num_coords )

        # initial guess for all operations
        psi0 = np.zeros( num_coords,
                         dtype = complex
                       )

        test_preconditioners = _create_preconditioner_list( precs, num_coords )

        # ----------------------------------------------------------------------
        # build the kinetic energy operator
        print('Building the KEO...')
        start_time = time.clock()
        modeleval._assemble_kinetic_energy_operator()
        end_time = time.clock()
        print('done. (', end_time - start_time, 's).')
        # ----------------------------------------------------------------------
        # Run the preconditioners and gather the relative residuals.
        relresvecs = _run_preconditioners( jacobian,
                                           rhs,
                                           psi0,
                                           test_preconditioners
                                         )
        # ----------------------------------------------------------------------
        # append the number of iterations to the data
        for prec in test_preconditioners:
            if prec['name'] not in num_iterations.keys():
                num_iterations[ prec['name'] ] = []
            num_iterations[ prec['name'] ].append(
                                            len( relresvecs[prec['name']] ) - 1
                                                )
        # ----------------------------------------------------------------------

    print(num_iterations)

    # plot them all
    plot_handles = []
    for prec in test_preconditioners:
        pp.semilogy( nums_unknowns,
                     num_iterations[ prec['name'] ],
                     '-o',
                     label = prec['name']
                   )

    # plot legend
    pp.legend()

    # add title and so forth
    pp.title( 'CG convergence for $J$' )
    pp.xlabel( 'Number of unknowns $n$' )
    pp.ylabel( 'Number of iterations till $<10^{-10}$' )

    matplotlib2tikz.save( 'meshrun.tikz',
                          figurewidth = '\\figurewidth',
                          figureheight = '\\figureheight'
                        )
    pp.show()

    return
# ==============================================================================
def _run_preconditioners( linear_operator, rhs, x0, preconditioners ):

    tol = 1.0e-10
    maxiter = 1000

    relresvecs = {}
    for prec in preconditioners:
        print('Solving the system with', prec['name'], '...')
        start_time = time.clock()
        sol, info, relresvec = nm.minres_wrap( linear_operator, rhs,
                                           x0 = x0,
                                           tol = tol,
                                           maxiter = maxiter,
                                           #M = prec['precondictioner'],
                                           #inner_product = prec['inner product']
                                         )
        end_time = time.clock()
        relresvecs[ prec['name'] ] = relresvec
        if info == 0:
            print('success!',)
        else:
            print('no convergence.',)
        print(' (', end_time - start_time, 's,', len(relresvec)-1 ,' iters).')

    return relresvecs
# ==============================================================================
def _plot_relresvecs( test_preconditioners,
                      relresvecs
                    ):
    # plot them all
    for prec in test_preconditioners:
        pp.semilogy( relresvecs[ prec['name'] ],
                     label = prec['name']
                   )

    # add title and so forth
    pp.title( 'CG convergence for $J$, $\mu=1.0$' )
    pp.xlabel( '$k$' )
    pp.ylabel( '$\|r_k\|_M / \|r_0\|_M$' )
    pp.legend()

    return
# ==============================================================================
def _plot_l2_condition_numbers( model_evaluator ):

    # set the range of parameters
    mu_min = 0.0
    mu_max = 5.0
    steps = 2
    mus = np.arange( steps, dtype=float ) \
        / (steps-1) * (mu_max-mu_min) \
        + mu_min

    small_eigenvals = np.zeros( len(mus) )
    large_eigenvals = np.zeros( len(mus) )

    k = 0
    for mu in mus:
        model_evaluator.set_parameter( mu )

        # get the KEO
        if model_evaluator._keo is None:
            model_evaluator._assemble_kinetic_energy_operator()

        print('Smallest..')
        # get smallest and largest eigenvalues
        small_eigenval = eigs( model_evaluator._keo,
                                       k = 1,
                                       sigma = None,
                                       which = 'SM',
                                       return_eigenvectors = False
                                     )
        small_eigenvals[ k ] = small_eigenval[ 0 ]
        print('done.', small_eigenvals[ k ])

        #print 'Largest..'
        #large_eigenval = arpack.eigen( model_evaluator._keo,
                                       #k = 1,
                                       #sigma = None,
                                       #which = 'LM',
                                       #return_eigenvectors = False
                                     #)
        #large_eigenvals[ k ] = large_eigenval[ 0 ]
        #print 'done.', large_eigenvals[ k ]

        print
        k += 1

    print(small_eigenvals)
    print(large_eigenvals)

    pp.plot( mus, small_eigenvals, 'g^' )
    pp.title( 'Smallest magnitude eigenvalues of J' )

    #pp.plot( mus, large_eigenvals, 'gv' )
   #pp.title( 'Largest magnitude eigenvalue of the KEO' )

    pp.xlabel( '$\mu$' )
    pp.show()

    return
# ==============================================================================
def _construct_matrix( linear_operator ):
    shape = linear_operator.shape

    A = np.zeros( shape )

    e = np.zeros( shape[0] )
    for j in range( shape[1] ):
        e[j] = 1.0
        A[:,j] = linear_operator * e
        e[j] = 0.0

    A = np.matrix( A )
    return A
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Solve the linearized Ginzburg--Landau problem.'
                                    )

    parser.add_argument('filename',
                        metavar = 'FILE',
                        type    = str,
                        help    = 'ExodusII file containing the geometry and the state'
                        )

    parser.add_argument('--file-mvp', '-f',
                        metavar = 'MVP_FILE',
                        type    = str,
                        help    = 'file that contains the magnetic vector potential (default: FILE)'
                        )

    parser.add_argument('--timesteps', '-t',
                        metavar='TIMESTEPS',
                        dest='timesteps',
                        nargs='+',
                        type=int,
                        default=[0],
                        help='read a particular time step (default: 0)'
                        )

    parser.add_argument('--mu', '-m',
                        metavar='MU',
                        nargs='+',
                        type=float,
                        default=[0.0],
                        help='values of mu; must be the same number of arguments as TIMESTEPS (default: 0.0)'
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

    parser.add_argument('--tolerance', '-r',
                        type = float,
                        default = 1.0e-10,
                        help = 'relative tolerance (default: 1.0e-10)'
                        )

    parser.add_argument('--show-relres', '-s',
                        dest='show_relres',
                        action='store_true',
                        default=False,
                        help='show the relative residuals (default: False)'
                        )

    parser.add_argument('--deflation-ix', '-d',
                        dest='use_deflation',
                        action='store_true',
                        default=False,
                        help='use deflation with i*x (default: False)'
                        )

    parser.add_argument('--bordering', '-b',
                        action='store_true',
                        default=False,
                        help='use bordering (default: False)'
                        )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
