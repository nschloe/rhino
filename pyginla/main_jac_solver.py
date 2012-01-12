#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Solve the linearized Ginzburg--Landau problem.
'''
# ==============================================================================
import mesh_io
import ginla_modelevaluator
import preconditioners
import numerical_methods as nm
from scipy.sparse.linalg import LinearOperator, arpack
import time

import numpy as np
import cmath

import matplotlib.pyplot as pp
#from matplotlib import rc
#rc( 'text', usetex = True )
#rc( 'font', family = 'serif' )

import matplotlib2tikz
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    print "Reading the mesh...",
    mesh, psi, A, field_data = mesh_io.read_mesh( args.filename,
                                                  timestep=args.timestep
                                                )
    print "done."

    # build the model evaluator
    mu = 1.0e-1
    ginla_modelval = ginla_modelevaluator.GinlaModelEvaluator( mesh, A, mu )

    # initialize the preconditioners
    precs = preconditioners.Preconditioners( ginla_modelval )

    num_unknowns = len( mesh.nodes )

    # create the linear operator
    ginla_jacobian = LinearOperator( (num_unknowns, num_unknowns),
                                     matvec = ginla_modelval.apply_jacobian,
                                     dtype = complex
                                   )

    # create precondictioner obj
    if args.use_preconditioner:
        keo_prec = LinearOperator( (num_unknowns, num_unknowns),
                                    matvec = precs.keo_amg,
                                    dtype = complex
                                 )
    else:
        keo_prec = None

    #_plot_l2_condition_numbers( ginla_modelval )

    # --------------------------------------------------------------------------
    # set psi at which to create the Jacobian
    #current_psi = (1.0-1.0e-2) * np.ones( num_unknowns, dtype = complex )
    current_psi = 1.0 * np.ones( num_unknowns, dtype = complex )

    #current_psi = psi

    # generate random numbers within the unit circle
    #current_psi = np.empty( num_unknowns, dtype = complex )
    #radius = np.random.rand( num_unknowns )
    #arg    = np.random.rand( num_unknowns ) * 2.0 * cmath.pi
    #for k in range( num_unknowns ):
        #current_psi[ k ] = cmath.rect(radius[k], arg[k])

    ginla_modelval.set_current_psi( current_psi )
    # --------------------------------------------------------------------------
    # create right hand side and initial guess
    phi0 = np.zeros( num_unknowns, dtype=complex )

    # right hand side
    rhs = np.ones( num_unknowns, dtype=complex )

    #rhs = np.empty( num_unknowns, dtype = complex )
    #radius = np.random.rand( num_unknowns )
    #arg    = np.random.rand( num_unknowns ) * 2.0 * cmath.pi
    #for k in range( num_unknowns ):
        #rhs[ k ] = cmath.rect(radius[k], arg[k])
    # --------------------------------------------------------------------------
    # Get reference solution
    #print "Get reference solution (dim = %d)..." % (2*num_unknowns),
    #start_time = time.clock()
    #ref_sol, info, relresvec, errorvec = nm.minres_wrap( ginla_jacobian, rhs,
                                           #x0 = phi0,
                                           #tol = 1.0e-14,
                                           #M = keo_prec,
                                           #inner_product = ginla_modelval.inner_product,
                                           #explicit_residual = True
                                         #)
    #end_time = time.clock()
    #if info == 0:
        #print "success!",
    #else:
        #print "no convergence.",
    #print " (", end_time - start_time, "s,", len(relresvec)-1 ," iters)."

    print "Solving the system (dim = %d)..." % (2*num_unknowns),
    start_time = time.clock()
    #sol, info, relresvec, errorvec = nm.minres_wrap( ginla_jacobian, rhs,
                                           #x0 = phi0,
                                           #tol = 1.0e-12,
                                           #M = keo_prec,
                                           #inner_product = ginla_modelval.inner_product,
                                           #explicit_residual = True,
                                           ##exact_solution = ref_sol
                                         #)

    sol, info, relresvec, errorvec = nm.gmres_wrap( ginla_jacobian, rhs,
                                           x0 = phi0,
                                           tol = 1.0e-12,
                                           Mleft = keo_prec,
                                           inner_product = ginla_modelval.inner_product,
                                           explicit_residual = True,
                                           #exact_solution = ref_sol
                                         )
    end_time = time.clock()
    if info == 0:
        print "success!",
    else:
        print "no convergence.",
    print " (", end_time - start_time, "s,", len(relresvec)-1 ," iters)."
    pp.semilogy( relresvec )
    #pp.semilogy( errorvec )
    pp.show()
    # --------------------------------------------------------------------------
    #_run_one_mu( ginla_modelval,
                 #precs,
                 #ginla_jacobian,
                 #rhs,
                 #psi0,
                 #test_preconditioners
               #)
    #_run_along_top( ginla_modelval,
                    #precs,
                    #ginla_jacobian,
                    #rhs,
                    #psi0,
                    #test_preconditioners
                  #)
    #_run_different_meshes( ginla_modelval,
                           #precs
                         #)
    # --------------------------------------------------------------------------
    return
# ==============================================================================
def _create_preconditioner_list( precs, num_unknowns ):

    test_preconditioners = []
    test_preconditioners.append( { 'name':'regular CG',
                                   'precondictioner': None,
                                   'inner product': 'regular'
                                 }
                               )
    test_preconditioners.append( { 'name':'-',
                                   'precondictioner': None,
                                   'inner product': 'real'
                                 }
                               )

    prec_diag = LinearOperator( (num_unknowns, num_unknowns),
                                matvec = precs.diagonal,
                                dtype = complex
                              )
    test_preconditioners.append( { 'name': 'diag',
                                   'precondictioner': prec_diag,
                                   'inner product': 'real'
                                 }
                               )

    prec_keolu = LinearOperator( (num_unknowns, num_unknowns),
                                 matvec = precs.keo_lu,
                                 dtype = complex
                               )
    test_preconditioners.append( { 'name': 'KEO ($LU$)',
                                   'precondictioner': prec_keolu,
                                   'inner product': 'real'
                                 }
                               )

    #prec_keoi = LinearOperator( (num_unknowns, num_unknowns),
                                #matvec = precs.keoi,
                                #dtype = complex
                              #)
    #test_preconditioners.append( { 'name': 'KEO$+\\alpha I$',
                                   #'precondictioner': prec_keoi,
                                   #'inner product': 'real'
                                 #}
                               #)

    #prec_keo_approx = LinearOperator( (num_unknowns, num_unknowns),
                                      #matvec = precs.keo_cgapprox,
                                      #dtype = complex
                                    #)
    #test_preconditioners.append( { 'name': 'KEO CG approx',
                                   #'precondictioner': prec_keo_approx,
                                   #'inner product': 'real'
                                 #}
                               #)

    #prec_keo_ilu4 = LinearOperator( (num_unknowns, num_unknowns),
                                    #matvec = precs.keo_ilu4,
                                    #dtype = complex
                                  #)
    #test_preconditioners.append( { 'name': 'KEO i$LU$4',
                                   #'precondictioner': prec_keo_ilu4,
                                   #'inner product': 'real'
                                 #}
                               #)

    #prec_keo_symilu2 = LinearOperator( (num_unknowns, num_unknowns),
                                       #matvec = precs.keo_symmetric_ilu2,
                                       #dtype = complex
                                     #)
    #test_preconditioners.append( { 'name': 'KEO sym i$LU$2',
                                   #'precondictioner': prec_keo_symilu2,
                                   #'inner product': 'real'
                                 #}
                               #)

    #prec_keo_symilu4 = LinearOperator( (num_unknowns, num_unknowns),
                                       #matvec = precs.keo_symmetric_ilu4,
                                       #dtype = complex
                                     #)
    #test_preconditioners.append( { 'name': 'KEO sym i$LU$4',
                                   #'precondictioner': prec_keo_symilu4,
                                   #'inner product': 'real'
                                 #}
                               #)

    #prec_keo_symilu6 = LinearOperator( (num_unknowns, num_unknowns),
                                      #matvec = precs.keo_symmetric_ilu6,
                                      #dtype = complex
                                    #)
    #test_preconditioners.append( { 'name': 'KEO sym i$LU$6',
                                   #'precondictioner': prec_keo_symilu6,
                                   #'inner product': 'real'
                                 #}
                               #)

    #prec_keo_symilu8 = LinearOperator( (num_unknowns, num_unknowns),
                                      #matvec = precs.keo_symmetric_ilu8,
                                      #dtype = complex
                                    #)
    #test_preconditioners.append( { 'name': 'KEO sym i$LU$8',
                                   #'precondictioner': prec_keo_symilu8,
                                   #'inner product': 'real'
                                 #}
                               #)

    return test_preconditioners
# ==============================================================================
def _run_one_mu( ginla_modelval,
                 precs,
                 ginla_jacobian,
                 rhs,
                 psi0,
                 test_preconditioners
               ):
    # --------------------------------------------------------------------------
    # build the kinetic energy operator
    print "Building the KEO..."
    start_time = time.clock()
    ginla_modelval._assemble_kinetic_energy_operator()
    end_time = time.clock()
    print "done.", end_time - start_time
    # --------------------------------------------------------------------------
    # Run the preconditioners and gather the relative residuals.
    relresvecs = _run_preconditioners( ginla_jacobian,
                                       rhs,
                                       psi0,
                                       test_preconditioners
                                     )

    # Plot the relative residuals.
    _plot_relresvecs( test_preconditioners, relresvecs )
    matplotlib2tikz.save( "one-mu.tikz",
                          figurewidth = "\\figurewidth",
                          figureheight = "\\figureheight"
                        )
    pp.show()
    return
# ==============================================================================
def _run_along_top( ginla_modelval,
                    precs,
                    ginla_jacobian,
                    rhs,
                    psi0,
                    test_preconditioners
                  ):

    num_unknowns = len( psi0 )

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
        print
        print " mu =", mu
        # ----------------------------------------------------------------------
        # build the kinetic energy operator
        ginla_modelval.set_parameter( mu )
        precs.set_parameter( mu )
        print "Building the KEO..."
        start_time = time.clock()
        ginla_modelval._assemble_kinetic_energy_operator()
        end_time = time.clock()
        print "done. (", end_time - start_time, "s)."
        # ----------------------------------------------------------------------
        # Run the preconditioners and gather the relative residuals.
        relresvecs = _run_preconditioners( ginla_jacobian,
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
    pp.ylabel( "Number of iterations till $<10^{-10}$" )
    pp.legend()

    matplotlib2tikz.save( "toprun.tikz",
                          figurewidth = "\\figurewidth",
                          figureheight = "\\figureheight"
                        )
    pp.show()

    return
# ==============================================================================
def _run_different_meshes( ginla_modelval,
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
    ginla_modelval.set_parameter( mu )
    precs.set_parameter( mu )

    # --------------------------------------------------------------------------
    # loop over the meshes and compute
    nums_unknowns = []

    num_iterations = {}

    for mesh_file in mesh_files:
        # ----------------------------------------------------------------------
        # read and set the mesh
        print
        print "Reading the mesh..."
        try:
            mesh = vtkio.read_mesh( mesh_file )
        except AttributeError:
            raise IOError( "Could not read from file ", mesh_file, "." )
        print " done."
        ginla_modelval.set_mesh( mesh )
        precs.set_mesh( mesh )
        # ----------------------------------------------------------------------
        # recreate all the objects necessary to perform the precondictioner run
        num_unknowns = len( mesh.nodes )

        nums_unknowns.append( num_unknowns )

        # create the linear operator
        ginla_jacobian = LinearOperator( (num_unknowns, num_unknowns),
                                         matvec = ginla_modelval.compute_jacobian,
                                         dtype = complex
                                       )

        # set psi at which to create the Jacobian
        # generate random numbers within the unit circle
        radius = np.random.rand( num_unknowns )
        arg    = np.random.rand( num_unknowns )
        current_psi = np.zeros( num_unknowns,
                                dtype = complex
                              )
        for k in range( num_unknowns ):
            current_psi[ k ] = cmath.rect(radius[k], arg[k])
        ginla_modelval.set_current_psi( current_psi )

        # create right hand side and initial guess
        rhs  =  np.random.rand( num_unknowns ) \
            + 1j * np.random.rand( num_unknowns )

        # initial guess for all operations
        psi0 = np.zeros( num_unknowns,
                         dtype = complex
                       )

        test_preconditioners = _create_preconditioner_list( precs, num_unknowns )

        # ----------------------------------------------------------------------
        # build the kinetic energy operator
        print "Building the KEO..."
        start_time = time.clock()
        ginla_modelval._assemble_kinetic_energy_operator()
        end_time = time.clock()
        print "done. (", end_time - start_time, "s)."
        # ----------------------------------------------------------------------
        # Run the preconditioners and gather the relative residuals.
        relresvecs = _run_preconditioners( ginla_jacobian,
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

    print num_iterations

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
    pp.ylabel( "Number of iterations till $<10^{-10}$" )

    matplotlib2tikz.save( "meshrun.tikz",
                          figurewidth = "\\figurewidth",
                          figureheight = "\\figureheight"
                        )
    pp.show()

    return
# ==============================================================================
def _run_preconditioners( linear_operator, rhs, x0, preconditioners ):

    tol = 1.0e-10
    maxiter = 1000

    relresvecs = {}
    for prec in preconditioners:
        print "Solving the system with", prec['name'], "..."
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
            print "success!",
        else:
            print "no convergence.",
        print " (", end_time - start_time, "s,", len(relresvec)-1 ," iters)."

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
    pp.ylabel( "$\|r_k\|_M / \|r_0\|_M$" )
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

        print 'Smallest..'
        # get smallest and largest eigenvalues
        small_eigenval = arpack.eigen( model_evaluator._keo,
                                       k = 1,
                                       sigma = None,
                                       which = 'SM',
                                       return_eigenvectors = False
                                     )
        small_eigenvals[ k ] = small_eigenval[ 0 ]
        print 'done.', small_eigenvals[ k ]

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

    print small_eigenvals
    print large_eigenvals

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

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'ExodusII file containing the geometry and initial state'
                       )

    parser.add_argument( '--timestep', '-t',
                         metavar='TIMESTEP',
                         dest='timestep',
                         nargs='?',
                         type=int,
                         const=0,
                         default=0,
                         help='read a particular time step (default: 0)'
                       )

    parser.add_argument( '--noprec', '-n',
                         dest='use_preconditioner',
                         action='store_const',
                         const=False,
                         default=True,
                         help='don\'t use a preconditioner (default: use keo)')

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()

    #import cProfile
    #cProfile.run( '_main()', 'pfvm_profile.dat' )
# ==============================================================================
