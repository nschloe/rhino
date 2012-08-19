# -*- coding: utf-8 -*-
# ==============================================================================
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
import time
import matplotlib.pyplot as pp
import numpy as np

import matplotlib2tikz
import voropy

#from lobpcg import lobpcg as my_lobpcg
import pynosh.modelevaluator_nls as nme
import pynosh.bordered_modelevaluator as bme
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    mesh, point_data, field_data = voropy.read(args.filename,
                                               timestep=args.timestep
                                               )

    num_nodes = len(mesh.node_coords)

    if not args.mu is None:
        mu = args.mu
        print 'Using mu=%g from command line.' % mu
    elif 'mu' in field_data:
        mu = field_data['mu']
    else:
        raise ValueError('Parameter ''mu'' not found in file. Please provide on command line.')

    if not args.g is None:
        g = args.g
        print 'Using g=%g from command line.' % g
    elif 'g' in field_data:
        g = field_data['g']
    else:
        raise ValueError('Parameter ''g'' not found in file. Please provide on command line.')

    # build the model evaluator
    nls_modeleval = nme.NlsModelEvaluator(mesh,
                                          g = g,
                                          V = point_data['V'],
                                          A = point_data['A'],
                                          mu = mu
                                          )

    psi0 = point_data['psi'][:,0] + 1j * point_data['psi'][:,1]

    if args.bordering:
        # Build bordered system.
        x0 = np.empty(num_nodes+1, dtype=complex)
        x0[0:num_nodes] = psi0
        x0[-1] = 0.0
        # Use psi0 as initial bordering.
        modeleval = bme.BorderedModelEvaluator(nls_modeleval)
    else:
        x0 = psi0
        modeleval = nls_modeleval

    if not args.series:
        # compute the eigenvalues once
        #p0 = 1j * psi0
        #p0 /= np.sqrt(modeleval.inner_product(p0, p0))
        #y0 = modeleval.get_jacobian(psi0) * p0
        #print '||(ipsi) J (ipsi)|| =', np.linalg.norm(y0)

        ## Check with the rotation vector.
        #grad_psi0 = mesh.compute_gradient(psi0)
        #x_tilde = np.array( [-mesh.node_coords[:,1], mesh.node_coords[:,0]] ).T
        #p1 = np.sum(x_tilde * grad_psi0, axis=1)
        #mesh.write('test.e', point_data={'x grad': p1})
        #nrm_p1 = np.sqrt(modeleval.inner_product(p1, p1))
        #p1 /= nrm_p1
        #y1 = modeleval.get_jacobian(psi0) * p1
        #print '||(grad) J (grad)|| =', np.linalg.norm(y1)

        # Check the equality
        #    grad(|psi|^2 psi) = 2 |psi|^2 grad(psi) + psi^2 grad(psi)*.
        #
        #p2 = mesh.compute_gradient(psi0 * abs(psi0)**2)
        #gradPsi0 = mesh.compute_gradient(psi0)
        #p2d = 2 * np.multiply(abs(psi0)**2, gradPsi0.T).T \
        #    + np.multiply(psi0**2, gradPsi0.conjugate().T).T
        #mesh.write('diff.vtu',
                   #point_data = {'psi': psi0, 'p2': p2, 'p2d': p2d, 'diff': diff}
                   #)

        # Check the equality
        #    grad(|psi|^2) = 2 Re(psi* grad(psi)).
        #
        #p2 = mesh.compute_gradient(abs(psi0)**2)
        #p2d = 2 * np.multiply(psi0.conjugate(), mesh.compute_gradient(psi0).T).T.real
        #diff = p2 - p2d
        #mesh.write('diff.vtu',
        #           point_data = {'psi': psi0, 'p2': p2, 'p2d': p2d, 'diff': diff}
        #           )

        #J = modeleval.get_jacobian(psi0)
        #K = modeleval._keo
        #x = np.random.rand(len(psi0))
        #print 'x', np.linalg.norm(J*x - K*x/ mesh.control_volumes.reshape(x.shape))

        eigenvals, X = _compute_eigenvalues(args.operator,
                                            args.eigenvalue_type,
                                            args.num_eigenvalues,
                                            None,
                                            x0[:,None],
                                            modeleval
                                            )

        print 'The following eigenvalues were computed:'
        print sorted(eigenvals)

        # Check residuals.
        print 'Residuals:'
        for k in xrange(len(eigenvals)):
            # Convert to complex representation.
            z = X[0::2,k] + 1j * X[1::2,k]
            z /= np.sqrt(modeleval.inner_product(z, z))
            y0 = modeleval.get_jacobian(x0) * z
            print np.linalg.norm(y0 - eigenvals[k] * z)

        # Normalize and store all eigenvectors & values.
        print 'Storing corresponding eigenstates...',
        k = 0
        for k in xrange(len(eigenvals)):
            filename = 'eigen%d.vtu' % k
            # Convert to complex representation.
            z = X[0::2,k] + 1j * X[1::2,k]
            z /= np.sqrt(modeleval.inner_product(z, z))
            mesh.write(filename,
                       point_data = {'psi': point_data['psi'], 'A': point_data['A'], 'V': point_data['V'], 'eigen': z},
                       field_data = {'g': g, 'mu': mu, 'eigenvalue': eigenvals[k] }
                       )
        print 'done.'
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
            modeleval.set_parameter(mu)
            eigenvals, X = _compute_eigenvalues(args.operator,
                                                args.eigenvalue_type,
                                                args.num_eigenvalues,
                                                X[:, 0],
                                                modeleval
                                                )
            #small_eigenval, X = my_lobpcg( modeleval._keo,
                                          #X,
                                          #tolerance = 1.0e-5,
                                          #maxiter = len(pynoshmesh.nodes),
                                          #verbosity = 1
                                        #)
            #print 'Calculated values: ', small_eigenval
            #alpha = modeleval.keo_smallest_eigenvalue_approximation()
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
                         modeleval
                         ):
    if operator_type == 'k':
        modeleval._assemble_keo()
        A = modeleval._keo
    elif operator_type == 'p':
        A = modeleval.get_preconditioner(psi)
    elif operator_type == 'j':
        jac = modeleval.get_jacobian(psi)
        # Consider bordering.
        #A = _complex_with_bordering2real( jac )
        A = _complex2real( jac )
    elif operator_type == 'pj':
        # build preconditioned operator
        prec_inv = modeleval.get_preconditioner_inverse(psi)
        jacobian = modeleval.get_jacobian(psi)
        def _apply_prec_jacobian(phi):
            return prec_inv * (jacobian * phi)
        num_unknowns = len(modeleval.mesh.node_coords)
        A = LinearOperator((num_unknowns, num_unknowns),
                            _apply_prec_jacobian,
                            dtype = complex
                            )
    else:
        raise ValueError('Unknown operator \'%s\'.' % operator_type)

    print 'Compute the %d %s eigenvalues of %s...' \
          % (num_eigenvalues, eigenvalue_type, operator_type)
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
def _complex2real( op ):
    '''For a given complex-valued operator C^n -> C^n, returns the
    corresponding real-valued operator R^{2n} -> R^{2n}.'''
    def _jacobian_wrap_apply( x ):
        # Build complex-valued representation.
        z = x[0::2] + 1j * x[1::2]
        z_out = op * z
        # Build real-valued representation.
        x_out = np.empty(x.shape)
        x_out[0::2] = z_out.real
        x_out[1::2] = z_out.imag
        return x_out

    return LinearOperator((2*op.shape[0], 2*op.shape[1]),
                          _jacobian_wrap_apply,
                          dtype = float
                          )
# ==============================================================================
def _complex_with_bordering2real( op ):
    def _jacobian_wrap_apply( x ):
        # Build complex-valued representation.
        z = np.empty(n+1, dtype=complex)
        a = x[0::2]
        b = x[1::2]
        z[0:n] = x[0:-1:2] + 1j * x[1:-1:2]
        z[n] = x[-1]
        z_out = op * z
        # Build real-valued representation.
        x_out = np.empty(x.shape)
        x_out[0:-1:2] = z_out[0:n].real
        x_out[1:-1:2] = z_out[0:n].imag
        assert abs(z_out[-1].imag) < 1.0e-15
        x_out[-1] = z_out[-1].real
        return x_out

    n = op.shape[0] - 1
    N = 2*n + 1
    return LinearOperator((N, N),
                          _jacobian_wrap_apply,
                          dtype = float
                          )
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
                        help='magnetic vector potential multiplier'
                        )

    parser.add_argument('--g', '-g',
                        dest='g',
                        type = float,
                        help='coupling parameter'
                        )

    parser.add_argument('--bordering', '-b',
                        default = False,
                        action = 'store_true',
                        help = 'use the bordered formulation to counter the nullspace (default: false)'
                        )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
