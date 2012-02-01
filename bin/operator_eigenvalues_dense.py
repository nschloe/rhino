# -*- coding: utf-8 -*-
# ==============================================================================
from scipy.linalg import norm, eig, eigh
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
    mu = 0.0 # dummy -- reset later
    ginla_modelval = \
        pyginla.ginla_modelevaluator.GinlaModelEvaluator( pyginlamesh, A, mu )

    # set the range of parameters
    steps = 1
    mus = np.linspace( 0.5, 0.5, steps )

    num_unknowns = len(pyginlamesh.nodes)

    # initial guess for the eigenvectors
    #psi = np.random.rand(num_unknowns) + 1j * np.random.rand(num_unknowns)
    psi = np.ones(num_unknowns) #+ 1j * np.ones(num_unknowns)
    psi *= 0.5
    #psi = 4.0 * 1.0j * np.ones(num_unknowns)
    print num_unknowns
    eigenvals_list = []
    # --------------------------------------------------------------------------
    for mu in mus:
        ginla_modelval.set_parameter(mu)
        
        if args.operator == 'k':
            # build dense KEO
            ginla_modelval._assemble_keo()
            K = ginla_modelval._keo
            A = ginla_modelval._keo.todense()
            B = None
        elif args.operator == 'p':
            # build dense preconditioner
            P = ginla_modelval.get_preconditioner(psi)
            A = P.todense()
            B = None
        elif args.operator == 'j':
            # build dense jacobian
            J1, J2 = ginla_modelval.get_jacobian_blocks(psi)
            A = _build_stacked_operator(J1.todense(), J2.todense())
            B = None
        elif args.operator == 'kj':
            J1, J2 = ginla_modelval.get_jacobian_blocks(psi)
            A = _build_stacked_operator(J1.todense(), J2.todense())

            ginla_modelval._assemble_keo()
            K = _ginla_modelval._keo
            B = _build_stacked_operator(K.todense())
        elif args.operator == 'pj':
            J1, J2 = ginla_modelval.get_jacobian_blocks(psi)
            A = _build_stacked_operator(J1.todense(), J2.todense())

            P = ginla_modelval.get_preconditioner(psi)
            B = _build_stacked_operator(P.todense())
        else:
            raise ValueError('Unknown operator \'', args.operator, '\'.')

        print 'Compute eigenvalues for mu =', mu, '..'
        # get smallesteigenvalues
        start_time = time.clock()
        # use eig as the problem is not symmetric (but it is self-adjoint)
        eigenvals, U = eig(A, b = B,
                            #lower = True,
                            )
        end_time = time.clock()
        print 'done. (', end_time - start_time, 's).'

        # sort by ascending eigenvalues
        assert norm(eigenvals.imag, np.inf) < 1.0e-14
        eigenvals = eigenvals.real
        sort_indices = np.argsort(eigenvals.real)
        eigenvals = eigenvals[sort_indices]
        U = U[:,sort_indices]

        # rebuild complex-valued U
        U_complex = _build_complex_vector(U)
        # normalize
        for k in xrange(U_complex.shape[1]):
            norm_Uk = np.sqrt(ginla_modelval.inner_product(U_complex[:,[k]], U_complex[:,[k]]))
            U_complex[:,[k]] /= norm_Uk

        # Compare the different expressions for the eigenvalues.
        for k in xrange(len(eigenvals)):
            JU_complex = J1 * U_complex[:,[k]] + J2 * U_complex[:,[k]].conj()
            uJu = ginla_modelval.inner_product(U_complex[:,k], JU_complex)[0]

            PU_complex = P * U_complex[:,[k]]
            uPu = ginla_modelval.inner_product(U_complex[:,k], PU_complex)[0]

            KU_complex = ginla_modelval._keo * U_complex[:,[k]]
            uKu = ginla_modelval.inner_product(U_complex[:,k], KU_complex)[0]

            # expression 1
            lambd = uJu / uPu
            assert abs(eigenvals[k] - lambd) < 1.0e-10, abs(eigenvals[k] - lambd)

            # expression 2
            alpha = ginla_modelval.inner_product(U_complex[:,k]**2, psi**2)
            lambd = uJu / (-uJu + 1.0 - alpha)
            assert abs(eigenvals[k] - lambd) < 1.0e-10

            # expression 3
            alpha = ginla_modelval.inner_product(U_complex[:,k]**2, psi**2)
            beta = ginla_modelval.inner_product(abs(U_complex[:,k])**2, abs(psi)**2)
            lambd = -1.0 + (1.0-alpha) / (uKu + 2*beta)
            assert abs(eigenvals[k] - lambd) < 1.0e-10

            # overwrite for plotting
            eigenvals[k] = 1- alpha

        eigenvals_list.append( eigenvals )
    # --------------------------------------------------------------------------

    # plot the eigenvalues
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
    '''Build the block operator.
       [ A.real+B.real, -A.imag+B.imag ]
       [ A.imag+B.imag,  A.real-B.real ]
    '''
    out = np.empty((2*A.shape[0], 2*A.shape[1]), dtype=float)
    out[0::2,0::2] = A.real
    out[0::2,1::2] = -A.imag
    out[1::2,0::2] = A.imag
    out[1::2,1::2] = A.real

    if B is not None:
        assert A.shape == B.shape
        out[0::2,0::2] += B.real
        out[0::2,1::2] += B.imag
        out[1::2,0::2] += B.imag
        out[1::2,1::2] -= B.real

    return out
# ==============================================================================
def _build_complex_vector(x):
    '''Build complex vector.
    '''
    xreal = x[0::2,:]
    ximag = x[1::2,:]
    return xreal + 1j * ximag
# ==============================================================================
def _build_real_vector(x):
    '''Build complex vector.
    '''
    xx = np.empty((2*len(x) ,1))
    xx[0::2,:] = x.real
    xx[1::2,:] = x.imag
    return xx
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
