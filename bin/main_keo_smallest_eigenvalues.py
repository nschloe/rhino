# -*- coding: utf-8 -*-
# ==============================================================================
import sys
from scipy.sparse.linalg import eigen, eigen_symmetric
import time
import matplotlib.pyplot as pp

#from lobpcg import lobpcg as my_lobpcg
import vtkio
from model_evaluator import *
import matplotlib2tikz
# ==============================================================================
def _main():
    '''
    Main function.
    '''
    opts, args = _parse_input_arguments()

    # read the mesh
    print "Reading the state (and mesh)..."
    try:
        mesh, psi, field_data = vtkio.read_mesh( opts.filename )
    except AttributeError:
        print "Could not read from file ", opts.filename, "."
        sys.exit()
    print " done."

    # build the model evaluator
    mu = 1.0e-0
    ginla_modelval = ginla_model_evaluator( mu )
    ginla_modelval.set_mesh( mesh )

    # set the range of parameters
    steps = 11
    mus = np.linspace( 0.0, 0.5, steps )

    #small_eigenvals = np.zeros( len(mus) )
    #large_eigenvals = np.zeros( len(mus) )

    # initial guess for the eigenvectors
    small_eigenvals = []
    small_eigenvals_approx = []
    num_eigenvalues = 10
    X = np.ones( (len(mesh.nodes), 1) )
    #X[:,0] = 1.0
    eigenvals = np.empty( (len(mus), num_eigenvalues  ) )
    # --------------------------------------------------------------------------
    k = 0
    for mu in mus:
        ginla_modelval.set_parameter( mu )

        # get the KEO
        if ginla_modelval._keo is None:
            ginla_modelval._assemble_kinetic_energy_operator()

        print 'Compute smallest eigenvalues for mu =', mu, '..'
        # get smallesteigenvalues
        start_time = time.clock()
        small_eigenval, X = eigen( ginla_modelval._keo,
                                   k = num_eigenvalues,
                                   sigma = None,
                                   which = 'SM',
                                   v0 = X[:,0],
                                   return_eigenvectors = True
                                 )
        small_eigenval = small_eigenval.real
        #small_eigenval, X = my_lobpcg( ginla_modelval._keo,
                                       #X,
                                       #tolerance = 1.0e-5,
                                       #maxiter = len(mesh.nodes),
                                       #verbosity = 1
                                     #)
        end_time = time.clock()
        print "done. (", end_time - start_time, "s)."
        print "Calculated value: ", small_eigenval
        alpha = ginla_modelval.keo_smallest_eigenvalue_approximation()
        print "Linear approximation: ", alpha
        small_eigenvals_approx.append( alpha )
        print
        eigenvals[k, :] = small_eigenval
        k += 1
    # --------------------------------------------------------------------------

    # plot all the eigenvalues as balls
    _plot_eigenvalue_series( mus, eigenvals )

    pp.plot( mus,
             small_eigenvals_approx,
             '--'
           )
    #pp.legend()
    pp.title( 'Smallest magnitude eigenvalues of KEO' )

    pp.ylim( ymin = 0.0 )

    pp.xlabel( '$\mu$' )

    pp.show()

    matplotlib2tikz.save( "smallest-ev.tikz",
                          figurewidth = "\\figurewidth",
                          figureheight = "\\figureheight"
                        )
    return
# ==============================================================================
def _plot_eigenvalue_series( x, eigenvals ):
    '''
    Plotting series of eigenvalues can be hard to make visually appealing. The
    reason for this is that at each data point, the values are mostly ordered
    in some way, not respecting previous calculations. When two eigenvalues
    "cross" -- and this notion doesn't actually exist -- then the colors of
    the two crossing parts change.
    This function tries to take care of this by guessing which are the
    corresponding values by linear extrapolation.
    '''
    # --------------------------------------------------------------------------
    def _linear_extrapolation( x, Y, x2 ):
        '''Linear extrapolation.
        '''
        return ( (Y[1, :] - Y[0, :]) * x2 + x[1]*Y[0, :] - Y[1, :]*x[0] ) \
               / (x[1] - x[0])
    # --------------------------------------------------------------------------
    def _permutation_match( y, y2 ):
        '''Returns the permutation of y that best matches y2.
        '''
        n = len(y2)
        assert len(y) == n
        #y = np.array( y ) # to make Boolean indices possible
        y_new = np.empty( n )
        active_set = np.ones( n, dtype = bool )
        for k in xrange( n ):
            diff = abs(y - y2[k])
            # TODO what if the same index is found more than once?
            # This needs to be prevented.
            min_index = min(xrange(len(diff)), key=diff.__getitem__)
            y_new[ k ] = y[ min_index ]
        return y_new
    # --------------------------------------------------------------------------

    num_steps, num_eigenvalues = eigenvals.shape
    reordered_eigenvalues = np.empty( eigenvals.shape )

    # insert the first step as is
    reordered_eigenvalues = eigenvals.copy()

    for k in xrange( 1, num_steps ): # leave the first step as is
        if k == 1:
            # constant extrapolation
            eigenvals_extrapolation = reordered_eigenvalues[k-1,:]
        else:
            # linear extrapolation
            eigenvals_extrapolation = _linear_extrapolation( x[k-2:k],
                                                             reordered_eigenvalues[k-2:k,:],
                                                             x[k] )

        # match the the extrapolation
        reordered_eigenvalues[k,:] = _permutation_match( reordered_eigenvalues[k,:],
                                                         eigenvals_extrapolation
                                                       )

    # plot it
    for k in range(num_eigenvalues):
        pp.plot( x,
                 reordered_eigenvalues[:,k],
                '-x'
              )
        k += 1
# ==============================================================================
def _parse_input_arguments():
    '''
    Parse input arguments.
    '''
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option( "-f", "--file",
                       dest = "filename",
                       type = str,
                       help = "read mesh from VTKFILE",
                       metavar = "VTKFILE"
                     )
    #parser.add_option("-q", "--quiet",
                      #action="store_false", dest="verbose", default=True,
                      #help="don't print status messages to stdout")

    (opts, args) = parser.parse_args()
    return opts, args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
