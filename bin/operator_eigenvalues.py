# -*- coding: utf-8 -*-
# ==============================================================================
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
import time
import matplotlib.pyplot as pp
import numpy as np

import matplotlib2tikz
import voropy

#from lobpcg import lobpcg as my_lobpcg
import pyginla.ginla_modelevaluator
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    mesh, point_data, field_data = voropy.read(args.filename,
                                               timestep=args.timestep
                                               )

    if 'mu' in field_data:
        mu = field_data['mu']
    else:
        if args.mu is None:
            raise ValueError('Parameter ''mu'' not found in file. Please provide on command line.')
        else:
            mu = args.mu
            print 'Using mu=%g from command line.' % mu

    # build the model evaluator
    ginla_modeleval = \
        pyginla.ginla_modelevaluator.GinlaModelEvaluator(mesh,
                                                         point_data['A'],
                                                         mu
                                                         )

    if not args.series:
        # compute the eigenvalues once
        psi0 = point_data['psi'][:,0] + 1j * point_data['psi'][:,1]
        eigenvals, X = _compute_eigenvalues(args.operator,
                                            args.eigenvalue_type,
                                            args.num_eigenvalues,
                                            None,
                                            psi0[:,None],
                                            ginla_modeleval
                                            )
        print 'The following eigenvalues were computed:'
        print eigenvals
    else:
        # initial guess for the eigenvectors
        X = np.ones((len(mesh.node_coords), 1))
        # ----------------------------------------------------------------------
        # set the range of parameters
        steps = 51
        mus = np.linspace( 0.0, 0.5, steps )
        eigenvals_list = []
        #small_eigenvals_approx = []
        for mu in mus:
            ginla_modeleval.set_parameter(mu)
            eigenvals, X = _compute_eigenvalues(args.operator,
                                                args.eigenvalue_type,
                                                args.num_eigenvalues,
                                                X[:, 0],
                                                ginla_modeleval
                                                )
            #small_eigenval, X = my_lobpcg( ginla_modeleval._keo,
                                          #X,
                                          #tolerance = 1.0e-5,
                                          #maxiter = len(pyginlamesh.nodes),
                                          #verbosity = 1
                                        #)
            #print 'Calculated values: ', small_eigenval
            #alpha = ginla_modeleval.keo_smallest_eigenvalue_approximation()
            #print 'Linear approximation: ', alpha
            #small_eigenvals_approx.append( alpha )
            #print
            eigenvals_list.append( eigenvals )
        # plot all the eigenvalues as balls
        _plot_eigenvalue_series( mus, eigenvals_list )
        #pp.legend()
        pp.title('%s eigenvalues of %s' % (args.eigenvalue_type, args.operator))
        #pp.ylim( ymin = 0.0 )
        pp.xlabel( '$\mu$' )
        pp.show()
        #matplotlib2tikz.save('eigenvalues.tikz',
                            #figurewidth = '\\figurewidth',
                            #figureheight = '\\figureheight'
                            #)
        # ----------------------------------------------------------------------
    return
# ==============================================================================
def _compute_eigenvalues(operator_type,
                         eigenvalue_type,
                         num_eigenvalues,
                         v0,
                         psi,
                         ginla_modeleval
                         ):
    if operator_type == 'k':
        ginla_modeleval._assemble_keo()
        A = ginla_modeleval._keo
    elif operator_type == 'p':
        A = ginla_modeleval.get_preconditioner(psi)
    elif operator_type == 'j':
        A = ginla_modeleval.get_jacobian(psi)
    elif operator_type == 'pj':
        # build preconditioned operator
        prec_inv = ginla_modeleval.get_preconditioner_inverse(psi)
        jacobian = ginla_modeleval.get_jacobian(psi)
        def _apply_prec_jacobian(phi):
            return prec_inv * (jacobian * phi)
        num_unknowns = len(ginla_modeleval.mesh.node_coords)
        A = LinearOperator((num_unknowns, num_unknowns),
                            _apply_prec_jacobian,
                            dtype = complex
                            )
    else:
        raise ValueError('Unknown operator \'%s\'.' % operator_type)

    print 'Compute the %s %d eigenvalues of %s...' \
          % (eigenvalue_type, num_eigenvalues, operator_type)
    start_time = time.clock()
    eigenvals, X = eigs(A,
                        k = num_eigenvalues,
                        sigma = None,
                        which = eigenvalue_type,
                        v0 = v0,
                        return_eigenvectors = True
                        )
    end_time = time.clock()
    print 'done. (', end_time - start_time, 's).'

    # make sure they are real (as they are supposed to be)
    assert all(abs(eigenvals.imag) < 1.0e-10), eigenvals

    return eigenvals.real, X
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

    parser = argparse.ArgumentParser(description = \
                                     'Compute a few eigenvalues of a specified operator.')

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'ExodusII file containing the geometry and initial state'
                       )

    parser.add_argument( '--timestep', '-t',
                         metavar='TIMESTEP',
                         dest='timestep',
                         type=int,
                         default=0,
                         help='read a particular time step (default: 0)'
                       )

    parser.add_argument( '--operator', '-o',
                         metavar='OPERATOR',
                         required = True,
                         dest='operator',
                         choices = ['k', 'p', 'j', 'pj'],
                         help='operator to compute the eigenvalues of (default: k)'
                       )

    parser.add_argument('--numeigenvalues', '-k',
                        dest='num_eigenvalues',
                        type=int,
                        default=6,
                        help='the number of eigenvalues to compute (default: 6)'
                        )

    parser.add_argument('--series', '-s',
                        dest='series',
                        action='store_true',
                        default=False,
                        help='compute a series of eigenvalues for different mu (default: False)'
                        )

    parser.add_argument('--type', '-y',
                        dest='eigenvalue_type',
                        default='SM',
                        help='the type of eigenvalues to compute (default: SM (smallest magnitude))'
                        )

    parser.add_argument('--mu', '-m',
                        dest='mu',
                        type = float,
                        help='magnetic vector potential multiplier (default: 1.0)'
                        )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
