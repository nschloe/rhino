# -*- coding: utf-8 -*-
# ==============================================================================
from scipy.linalg import norm, eigh
from scipy.sparse import spdiags
import time
import matplotlib.pyplot as pp
import numpy as np

import matplotlib2tikz

#from lobpcg import lobpcg as my_lobpcg
import mesh.mesh_io
import pyginla.ginla_modelevaluator
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh(args.filename,
                                                  timestep=args.timestep
                                                )

    # build the model evaluator
    mu = 0.0
    ginla_modelval = \
        pyginla.ginla_modelevaluator.GinlaModelEvaluator( pyginlamesh, A, mu )

    # set the range of parameters
    steps = 1
    mus = np.linspace( 0.5, 0.5, steps )

    num_unknowns = len(pyginlamesh.nodes)

    # initial guess for the eigenvectors
    small_eigenvals_approx = []
    num_eigenvalues = 10
    X = np.ones((num_unknowns, 1))
    psi = np.random.rand(num_unknowns) + 1j * np.random.rand(num_unknowns)
    #X[:,0] = 1.0
    print num_unknowns
    eigenvals_list = []
    # --------------------------------------------------------------------------
    for mu in mus:
        ginla_modelval.set_parameter(mu)

        # get scaling matrix
        if ginla_modelval.control_volumes is None:
            ginla_modelval._compute_control_volumes()
        D = spdiags(ginla_modelval.control_volumes.T, [0],
                    num_unknowns, num_unknowns
                    )
        
        if args.operator == 'k':
            # build dense KEO
            ginla_modelval._assemble_keo()
            K = D * ginla_modelval._keo
            A = K.todense()
            B = None
        elif args.operator == 'p':
            # build dense preconditioner
            P = D * ginla_modelval.get_preconditioner(psi)
            A = P.todense()
            B = None
        elif args.operator == 'j':
            # build dense jacobian
            J1, J2 = ginla_modelval.get_jacobian_blocks(psi)
            J1 = D * J1
            J2 = D * J2
            A = _build_stacked_operator(J1.todense(), J2.todense())
            B = None
        elif args.operator == 'kj':
            J1, J2 = ginla_modelval.get_jacobian_blocks(psi)
            J1 = D * J1
            J2 = D * J2
            A = _build_stacked_operator(J1.todense(), J2.todense())

            K = D * ginla_modelval._keo
            B = _build_stacked_operator(K.todense())
        elif args.operator == 'pj':
            J1, J2 = ginla_modelval.get_jacobian_blocks(psi)
            J1 = D * J1
            J2 = D * J2
            A = _build_stacked_operator(J1.todense(), J2.todense())

            P = D * ginla_modelval.get_preconditioner(psi)
            B = _build_stacked_operator(P.todense())
        else:
            raise ValueError('Unknown operator \'', args.operator, '\'.')

        print 'Compute eigenvalues for mu =', mu, '..'
        # get smallesteigenvalues
        start_time = time.clock()
        eigenvals, X = eigh(A, b = B,
                            lower = True,
                            )
        end_time = time.clock()
        print 'done. (', end_time - start_time, 's).'

        # make sure they are real (as they are supposed to be)
        assert all(abs(eigenvals.imag) < 1.0e-10), eigenvals
        eigenvals_list.append( eigenvals.real )
        #small_eigenval, X = my_lobpcg( ginla_modelval._keo,
                                       #X,
                                       #tolerance = 1.0e-5,
                                       #maxiter = len(pyginlamesh.nodes),
                                       #verbosity = 1
                                     #)
        #print 'Calculated values: ', small_eigenval
        #alpha = ginla_modelval.keo_smallest_eigenvalue_approximation()
        #print 'Linear approximation: ', alpha
        #small_eigenvals_approx.append( alpha )
        #print
        eigenvals_list.append( eigenvals )
    # --------------------------------------------------------------------------

    # plot all the eigenvalues as balls
    #_plot_eigenvalue_series( mus, eigenvals_list )
    for ev in eigenvals_list:
        pp.plot(ev, '.')

    #pp.plot( mus,
             #small_eigenvals_approx,
             #'--'
           #)
    #pp.legend()
    pp.title('eigenvalues of %s' % args.operator)

    #pp.ylim( ymin = 0.0 )

    #pp.xlabel( '$\mu$' )

    pp.show()

    #matplotlib2tikz.save('eigenvalues.tikz',
                         #figurewidth = '\\figurewidth',
                         #figureheight = '\\figureheight'
                         #)
    return
# ==============================================================================
def _build_stacked_operator(A, B=None):
    '''Build the operator
    '''
    if B is None:
        return np.r_[ np.c_[A.real, -A.imag],
                      np.c_[A.imag,  A.real] ]
    else:
        return np.r_[ np.c_[A.real+B.real, -A.imag+B.imag],
                      np.c_[A.imag+B.imag,  A.real-B.real] ]
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser(description =
                                      'Compute all eigenvalues of a specified.')

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

    parser.add_argument( '--operator', '-o',
                         metavar='OPERATOR',
                         dest='operator',
                         nargs='?',
                         type=str,
                         const='k',
                         default='k',
                         help='operator to compute the eigenvalues of (k, p, j, pj; default: k)'
                       )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
