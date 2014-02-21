#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Solve a linear equation system with the kinetic energy operator.
'''
import vtkio
import numerical_methods as nm
import sys
from scipy.sparse.linalg import LinearOperator
import time

import matplotlib.pyplot as pp
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

from model_evaluator import *
from preconditioners import *


def _main():
    '''
    Main function.
    '''

    # create model evaluator interface
    mu = 1.0e-0
    pynosh_modelval = pynosh_model_evaluator(mu)

    # create preconditioners object
    precs = preconditioners(pynosh_modelval)
    precs.set_parameter(mu)

    # run the preconditioners
    _run_different_meshes(pynosh_modelval, precs)

    #print 'Solving the system without preconditioning, scipy cg...'
    #sol, info, relresvec0 = nm.cg_wrap( pynosh_modelval._keo, rhs,
                                        #x0 = psi0,
                                        #tol = 1.0e-10,
                                        #maxiter = 1000,
                                        #M = None
                                      #)
    ##print 'done.'
    ##print info

    ## plot (relative) residuals
    #pp.semilogy( relresvec0, 'ro' )
    ##pp.semilogy( relresvec1, 'g' )
    ##pp.semilogy( relresvec2, 'b' )
    ##pp.semilogy( relresvec3, 'y' )
    #pp.title( 'Convergence history of CG for the KEO, $\mu=0.1$' )
    #pp.xlabel( '$k$' )
    #pp.ylabel( '$\|A_{\mathrm{KEO}}\psi_k-b\|_2$' )
    #pp.show()

    return


def _run_different_meshes(pynosh_modelval,
                          precs
                          ):
    mesh_files = [
        #'states/rectangle10.vtu',
        #'states/rectangle20.vtu',
        #'states/rectangle30.vtu',
        #'states/rectangle40.vtu',
        #'states/rectangle50.vtu',
        #'states/rectangle60.vtu',
        #'states/rectangle70.vtu',
        #'states/rectangle80.vtu',
        #'states/rectangle90.vtu',
        #'states/rectangle100.vtu',
        #'states/rectangle110.vtu',
        #'states/rectangle120.vtu',
        #'states/rectangle130.vtu',
        #'states/rectangle140.vtu',
        #'states/rectangle150.vtu',
        #'states/rectangle160.vtu',
        #'states/rectangle170.vtu',
        #'states/rectangle180.vtu',
        #'states/rectangle190.vtu',
        'states/rectangle200.vtu'
        ]

    # loop over the meshes and compute
    nums_unknowns = []

    num_iterations = {}

    for mesh_file in mesh_files:
        # read and set the mesh
        print
        print 'Reading the mesh...'
        try:
            mesh, psi, field_data = vtkio.read_mesh(mesh_file)
        except AttributeError:
            print 'Could not read from file ', mesh_file, '.'
            sys.exit()
        print ' done.'
        pynosh_modelval.set_mesh(mesh)
        precs.set_mesh(mesh)

        # recreate all the objects necessary to perform the precondictioner run
        num_unknowns = len( mesh.nodes )

        nums_unknowns.append( num_unknowns )

        # set psi at which to create the Jacobian
        # generate random numbers within the unit circle
        radius = np.random.rand( num_unknowns )
        arg    = np.random.rand( num_unknowns )
        current_psi = np.empty( num_unknowns,
                                dtype = complex
                              )
        for k in range( num_unknowns ):
            current_psi[ k ] = cmath.rect(radius[k], arg[k])
        pynosh_modelval.set_current_psi( current_psi )

        # create right hand side and initial guess
        rhs  =  np.random.rand( num_unknowns ) \
            + 1j * np.random.rand( num_unknowns )

        # initial guess for all operations
        psi0 = np.zeros( num_unknowns,
                         dtype = complex
                       )

        test_preconditioners = _create_preconditioner_list(precs,
                                                           num_unknowns
                                                           )

        # build the kinetic energy operator
        print 'Building the KEO...'
        start_time = time.clock()
        pynosh_modelval._assemble_kinetic_energy_operator()
        end_time = time.clock()
        print 'done. (', end_time - start_time, 's).'

        # Run the preconditioners and gather the relative residuals.
        relresvecs = _run_preconditioners(pynosh_modelval._keo,
                                          rhs,
                                          psi0,
                                          test_preconditioners
                                          )

        # append the number of iterations to the data
        for prec in test_preconditioners:
            if prec['name'] not in num_iterations.keys():
                num_iterations[prec['name']] = []
            num_iterations[prec['name']].append(
                len(relresvecs[prec['name']]) - 1
                )

    print num_iterations

    # plot them all
    for prec in test_preconditioners:
        pp.semilogy(nums_unknowns,
                    num_iterations[prec['name']],
                    '-o',
                    label=prec['name']
                    )

    # plot legend
    pp.legend()

    # add title and so forth
    pp.title('CG convergence for $K$')
    pp.xlabel('Number of unknowns $n$')
    pp.ylabel('Number of iterations till $<10^{-10}$')

    matplotlib2tikz.save('meshrun-k.tikz',
                         figurewidth='\\figurewidth',
                         figureheight='\\figureheight'
                         )
    pp.show()
    return


def _run_preconditioners(linear_operator, rhs, x0, preconditioners):
    tol = 1.0e-10
    maxiter = 5000
    relresvecs = {}
    for prec in preconditioners:
        print 'Solving the system with', prec['name'], '...'
        start_time = time.clock()
        sol, info, relresvec = nm.cg_wrap(linear_operator, rhs,
                                          x0=x0,
                                          tol=tol,
                                          maxiter=maxiter,
                                          M=prec['precondictioner']
                                          )
        end_time = time.clock()
        relresvecs[prec['name']] = relresvec
        if info == 0:
            print 'success!',
        else:
            print 'no convergence.',
        print ' (', end_time - start_time, 's,', len(relresvec)-1, ' iters).'
    return relresvecs


def _create_preconditioner_list(precs, num_unknowns):

    test_preconditioners = []

    test_preconditioners.append({'name': '-',
                                 'precondictioner': None
                                 })

    prec_keo_symilu2 = LinearOperator((num_unknowns, num_unknowns),
                                      matvec=precs.keo_symmetric_ilu2,
                                      dtype=complex
                                      )

    test_preconditioners.append({'name': 'sym i$LU$2',
                                 'precondictioner': prec_keo_symilu2
                                 })

    prec_keo_symilu4 = LinearOperator( (num_unknowns, num_unknowns),
                                       matvec = precs.keo_symmetric_ilu4,
                                       dtype = complex
                                     )
    test_preconditioners.append( { 'name': 'sym i$LU$4',
                                   'precondictioner': prec_keo_symilu4
                                 }
                               )

    prec_keo_symilu6 = LinearOperator( (num_unknowns, num_unknowns),
                                      matvec = precs.keo_symmetric_ilu6,
                                      dtype = complex
                                    )
    test_preconditioners.append( { 'name': 'sym i$LU$6',
                                   'precondictioner': prec_keo_symilu6
                                 }
                               )

    prec_keo_symilu8 = LinearOperator( (num_unknowns, num_unknowns),
                                       matvec = precs.keo_symmetric_ilu8,
                                       dtype = complex
                                     )
    test_preconditioners.append( { 'name': 'sym i$LU$8',
                                   'precondictioner': prec_keo_symilu8
                                 }
                               )


    prec_keo_amg = LinearOperator( (num_unknowns, num_unknowns),
                                   matvec = precs.keo_amg,
                                   dtype = complex
                                 )
    test_preconditioners.append( { 'name': 'AMG',
                                   'precondictioner': prec_keo_amg
                                 }
                               )
    return test_preconditioners


def _construct_matrix(linear_operator):
    shape = linear_operator.shape
    A = np.zeros(shape)
    e = np.zeros(shape[0])
    for j in range(shape[1]):
        e[j] = 1.0
        A[:, j] = linear_operator * e
        e[j] = 0.0
    A = np.matrix(A)
    return A


def _parse_input_arguments():
    '''Parse input arguments.
    '''
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-f', '--file',
                      dest='filename',
                      type=str,
                      help='read mesh from VTKFILE',
                      metavar='VTKFILE'
                      )
    #parser.add_option('-q', '--quiet',
                      #action='store_false', dest='verbose', default=True,
                      #help='don't print status messages to stdout')
    (opts, args) = parser.parse_args()
    return opts, args


if __name__ == '__main__':
    _main()

    #import cProfile
    #cProfile.run( '_main()', 'pfvm_profile.dat' )
