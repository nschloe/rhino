# -*- coding: utf-8 -*-
# ==============================================================================
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
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
    steps = 51
    mus = np.linspace( 0.0, 0.5, steps )

    #small_eigenvals = np.zeros( len(mus) )
    #large_eigenvals = np.zeros( len(mus) )

    # initial guess for the eigenvectors
    small_eigenvals = []
    small_eigenvals_approx = []
    num_eigenvalues = 10
    X = np.ones( (len(pyginlamesh.nodes), 1) )
    #X[:,0] = 1.0
    eigenvals_list = []
    num_unknowns = len(pyginlamesh.nodes)
    # --------------------------------------------------------------------------
    for mu in mus:
        ginla_modelval.set_parameter(mu)

        if args.operator == 'k':
            ginla_modelval._assemble_keo()
            A = ginla_modelval._keo
        elif args.operator == 'p':
            A = ginla_modelval.get_preconditioner(psi)
        elif args.operator == 'j':
            A = ginla_modelval.get_jacobian(psi)
        elif args.operator == 'pj':
            # build preconditioned operator
            prec_inv = ginla_modelval.get_preconditioner_inverse(psi)
            jacobian = ginla_modelval.get_jacobian(psi)
            def _apply_prec_jacobian(phi):
                return prec_inv * (jacobian * phi)
            A = LinearOperator((num_unknowns, num_unknowns),
                               _apply_prec_jacobian,
                               dtype = complex
                               )
        else:
            raise ValueError('Unknown operator \'', args.operator, '\'.')

        print 'Compute smallest eigenvalues for mu =', mu, '..'
        # get smallesteigenvalues
        start_time = time.clock()
        eigenvals, X = eigs(A,
                            k = num_eigenvalues,
                            sigma = None,
                            which = args.eigenvalue_type,
                            v0 = X[:,0],
                            return_eigenvectors = True
                            )
        end_time = time.clock()
        print 'done. (', end_time - start_time, 's).'

        # make sure they are real (as they are supposed to be)
        assert all(abs(eigenvals.imag) < 1.0e-10), eigenvals
        eigenvals = eigenvals.real
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
    _plot_eigenvalue_series( mus, eigenvals_list )

    #pp.plot( mus,
             #small_eigenvals_approx,
             #'--'
           #)
    #pp.legend()
    pp.title('%s eigenvalues of %s' % (args.eigenvalue_type, args.operator))

    #pp.ylim( ymin = 0.0 )

    pp.xlabel( '$\mu$' )

    pp.show()

    #matplotlib2tikz.save('eigenvalues.tikz',
                         #figurewidth = '\\figurewidth',
                         #figureheight = '\\figureheight'
                         #)
    return
# ==============================================================================
def _plot_eigenvalue_series(x, eigenvals_list):
    '''Plotting series of eigenvalues can be hard to make visually appealing.
    The reason for this is that at each data point, the values are mostly
    ordered in some way, not respecting previous calculations. When two
    eigenvalues 'cross' -- and this notion doesn't actually exist -- then the
    colors of the two crossing parts change.
    This function tries to take care of this by guessing which are the
    corresponding values by linear extrapolation.
    '''
    # --------------------------------------------------------------------------
    def _linear_extrapolation(x0, x1, Y0, Y1, x2):
        '''Linear extrapolation of the data sets (x0,Y0), (x1,Y1) to x2.
        '''
        return ( (Y1 - Y0) * x2 + x1*Y0 - Y1*x0 ) \
               / (x1 - x0)
    # --------------------------------------------------------------------------
    def _permutation_match(y, y2):
        '''Returns the permutation of y that best matches y2.
        '''
        n = len(y2)
        assert len(y) == n
        #y = np.array( y ) # to make Boolean indices possible
        y_new = np.empty( n )
        y_masked = np.ma.array(y, mask=np.zeros(n, dtype = bool))
        for k in xrange(n):
            min_index = np.argmin(abs(y_masked - y2[k]))
            y_new[k] = y_masked[min_index]
            # mask the index
            y_masked.mask[min_index] = True
        return y_new
    # --------------------------------------------------------------------------

    len_list = len(eigenvals_list)
    num_eigenvalues = len(eigenvals_list[0])
    # Stuff the reordered eigenvalues into an array so we can easily fill the
    # columns and then plot the rows.
    reordered_eigenvalues = np.zeros((num_eigenvalues, len_list), dtype=float)
    reordered_eigenvalues[:,0] = eigenvals_list[0]
    # use the same values for the first (constant) extrapolation)
    eigenvals_extrapolation = reordered_eigenvalues[:,0]
    for k, eigenvalues in enumerate(eigenvals_list[1:]): # skip the first
        # match the the extrapolation
        reordered_eigenvalues[:,k+1] = _permutation_match(eigenvalues,
                                                          eigenvals_extrapolation
                                                          )
        # linear extrapolation
        if k+2 < len(x):
            eigenvals_extrapolation = _linear_extrapolation(x[k], x[k+1],
                                                            reordered_eigenvalues[:,k],
                                                            reordered_eigenvalues[:,k+1],
                                                            x[k+2]
                                                            )

    # plot it
    for k in xrange(num_eigenvalues):
        pp.plot(x, reordered_eigenvalues[k,:], '-x')

    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Smallest eigenvalues of the KEO.'
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

    parser.add_argument( '--operator', '-o',
                         metavar='OPERATOR',
                         dest='operator',
                         nargs='?',
                         type=str,
                         const='k',
                         default='k',
                         help='operator to compute the eigenvalues of (k, p, j, pj; default: k)'
                       )

    parser.add_argument( '--largest', '-l',
                         dest='eigenvalue_type',
                         action='store_const',
                         const='LM',
                         default='SM',
                         help='get the largest eigenvalues (default: smallest)')

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
