#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''void
'''
# ==============================================================================
import vtkio
import numerical_methods as nm
from scipy.sparse.linalg import LinearOperator
from lobpcg import lobpcg
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
    '''Main function.
    '''
    # parse input arguments
    opts, args = _parse_input_arguments()

    state_files = sorted( glob.glob( str(opts.foldername) + '/solution*.vtu' ) )

    print state_files[0]

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

    # create precondictioner object
    precs = preconditioners( ginla_modelval )
    precs.set_mesh( mesh )


    # create and initial guess for the eigenvectors
    num_unknowns = len( mesh.nodes )
    num_eigenvalues = 10
    blockvector_x = np.random.rand( num_unknowns, num_eigenvalues )

    # --------------------------------------------------------------------------
    # loop over the meshes and compute
    eigenvalue_series = []
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

        mu = field_data["mu"]

        ginla_modelval.set_parameter( mu )
        ginla_modelval.set_current_psi( psi )

        precs.set_parameter( mu )
        # ----------------------------------------------------------------------
        # recreate all the objects necessary to perform the precondictioner run

        ## create precondictioner
        #prec_keolu = LinearOperator( (num_unknowns, num_unknowns),
                                    #matvec = precs.keo_lu,
                                    #dtype = complex
                                  #)

        # create the linear operator
        ginla_jacobian = LinearOperator( (num_unknowns, num_unknowns),
                                         matvec = ginla_modelval.compute_jacobian,
                                         dtype = complex
                                       )

        # ----------------------------------------------------------------------
        # build the kinetic energy operator
        print "Building the KEO..."
        start_time = time.clock()
        ginla_modelval._assemble_kinetic_energy_operator()
        end_time = time.clock()
        print "done. (", end_time - start_time, "s)."
        # ----------------------------------------------------------------------
        # Run the preconditioners and gather the relative residuals.
        print "Compute the eigenvalues with LOBPCG.."
        start_time = time.clock()
        eigenvalues, blockvector_x = lobpcg( ginla_jacobian,
                                             blockvector_x )
        end_time = time.clock()
        print "done (", end_time - start_time, "s,", len(relresvec)-1 ," iters)."
        #pp.semilogy( relresvec )
        #pp.show()
        # ----------------------------------------------------------------------
        # append the number of iterations to the data
        eigenvalue_series.append( eigenvalues )
        # ----------------------------------------------------------------------
    print ( eigenvalue_series )

    ## plot the number of iterations
    #pp.plot( num_iterations, 'o' )

    ## add title and so forth
    #pp.title( 'CG convergence for $J$' )
    #pp.xlabel( 'Continuation step $k$' )
    #pp.ylabel( "Number of CG iterations till $<10^{-10}$" )

    #matplotlib2tikz.save( "pcg-iterations.tikz",
                          #figurewidth = "\\figurewidth",
                          #figureheight = "\\figureheight"
                        #)
    #pp.show()
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
