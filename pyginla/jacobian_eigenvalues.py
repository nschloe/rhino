#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Compute the eigenvalues of the Jacobian operator for a number of states.
'''
# ==============================================================================
import vtkio
import numerical_methods as nm
from scipy.sparse.linalg import LinearOperator, eigen
import time
import glob

import matplotlib.pyplot as pp
from matplotlib import rc
rc( 'text', usetex = True )
rc( 'font', family = 'serif' )

from model_evaluator import *
from preconditioners import *
import matplotlib2tikz
# ==============================================================================
def _main():
    '''
    Main function.
    '''
    # parse input arguments
    opts, args = _parse_input_arguments()

    state_files = sorted( glob.glob( str(opts.foldername) + '/solution*.vtu' ) )

    tol = 1.0e-8
    maxiter = 5000

    # Create the model evaluator.
    # Get mu and mesh for the first grid. This way, the mesh doesn't need to
    # be reset in each step; this assumes of course that the mesh doesn't
    # change throughout the computation.
    print "Reading the state \"" + state_files[0] + "\"..."
    try:
        mesh, psi, field_data = vtkio.read_mesh( state_files[0] )
    except AttributeError:
        print "Could not read from file ", state_files[0], "."
        raise
    print " done."
    ginla_modelval = ginla_model_evaluator( field_data["mu"] )
    ginla_modelval.set_mesh( mesh )

    # initial value at the beginning of the iteration
    X = np.ones( (len(mesh.nodes), 1) )

    num_eigenvalues = 5
    eigenvals = np.empty( (len(state_files), num_eigenvalues  ) )

    # --------------------------------------------------------------------------
    # loop over the meshes and compute
    num_iterations = []
    k = 0
    for state_file in state_files:
        # ----------------------------------------------------------------------
        # read and set the mesh
        print
        print "Reading the state \"" + state_file + "\"..."
        try:
            mesh, psi, field_data = vtkio.read_mesh( state_file )
        except AttributeError:
            print "Could not read from file ", state_file, "."
            raise
        print " done."

        mu = field_data[ "mu" ]

        ginla_modelval.set_parameter( mu )
        ginla_modelval.set_current_psi( psi )

        # ----------------------------------------------------------------------
        # recreate all the objects necessary to perform the precondictioner run
        num_unknowns = len( mesh.nodes )

        # create the linear operator
        ginla_jacobian = LinearOperator( (num_unknowns, num_unknowns),
                                         matvec = ginla_modelval.compute_jacobian,
                                         dtype = complex
                                       )

        ## initial guess for all operations
        #psi0 = np.random.rand( num_unknowns ) \
             #+ 1j * np.random.rand( num_unknowns )

        print 'Compute smallest eigenvalues for state \"' + state_file+ '\"...'
        # get smallesteigenvalues
        start_time = time.clock()
        small_eigenval, X = eigen( ginla_jacobian,
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
        print
        eigenvals[k, :] = small_eigenval
        k += 1
        #pp.semilogy( relresvec )
        #pp.show()
    # --------------------------------------------------------------------------

    # plot all the eigenvalues as balls
    _plot_eigenvalue_series( mus, eigenvals )

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
def _parse_input_arguments():
    '''
    Parse input arguments.
    '''
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option( "-f", "--folder",
                       dest = "foldername",
                       type = str,
                       help = "read states \"solution*.vtu\" from FOLDER",
                       metavar = "FOLDER"
                     )
    #parser.add_option("-q", "--quiet",
                      #action="store_false", dest="verbose", default=True,
                      #help="don't print status messages to stdout")

    (opts, args) = parser.parse_args()
    return opts, args
# ==============================================================================
if __name__ == "__main__":
    _main()

    #import cProfile
    #cProfile.run( '_main()', 'pfvm_profile.dat' )
# ==============================================================================
