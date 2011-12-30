#! /usr/bin/env python
# -*- coding: utf-8 -*-

import mesh_io
import ginla_modelevaluator
#import numerical_methods as nm
import numpy as np
import scipy.sparse.linalg
#from scipy.sparse import identity
#from scipy.linalg import norm
import time
from math import sqrt


#import matplotlib.pyplot as pp
#from matplotlib import rc
#rc( 'text', usetex = True )
#rc( 'font', family = 'serif' )

#from lobpcg import lobpcg
#import matplotlib2tikz
# ==============================================================================
def _main():
    '''Main function.
    '''
    # --------------------------------------------------------------------------
    args = _parse_input_arguments()
    # --------------------------------------------------------------------------
    # read the mesh
    print "Reading the state (and mesh)..."
    mesh, psi, A, field_data = mesh_io.read_mesh( args.filename,
                                                  timestep=args.timestep
                                                )
    print " done."
    # --------------------------------------------------------------------------
    # build the model evaluator
    mu = 0.0
    ginla_modeleval = ginla_modelevaluator.GinlaModelEvaluator( mesh, A, mu )
    # --------------------------------------------------------------------------
    # build the kinetic energy operator
    print "Building the KEO..."
    start_time = time.clock()
    ginla_modeleval._assemble_keo()
    end_time = time.clock()
    print "done (", end_time - start_time, 's).'
    # --------------------------------------------------------------------------
    # find the lowest energy eigenstate
    # try eigen_symmetric
    # prepare initial guess
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #(m,n) = ginla_modeleval._keo.shape
    #X = np.zeros( (n,1) )
    #X[0] = 1.0
    #eigenvalues, eigenvectors = lobpcg( ginla_modeleval._keo,
                                        #X,
                                        #B = None,
                                        #M = None,
                                        #Y = None,
                                        #tol = None,
                                        #maxiter = 20,
                                        #largest = False, # get smallest
                                        #verbosityLevel = 0,
                                        #retLambdaHistory = False,
                                        #retResidualNormsHistory = False
                                      #)
    #print 'eigenvalues', eigenvalues
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print "Find lowest energy eigenstate/-value..."
    start_time = time.clock()
    lambd, v = scipy.sparse.linalg.eigs( ginla_modeleval._keo,
                      k = 1,
                      #M = None,
                      sigma = None,
                      which = 'SM', # 'LM'
                      #v0 = None,
                      #ncv = None,
                      maxiter = 1e10,
                      #tol = 1.0e-10,
                      return_eigenvectors = True
                    )
    end_time = time.clock()
    print "done (", end_time - start_time, 's).'
    v = v[:,0]
    print lambd
    # normalize the state in the native norm
    v /= sqrt(ginla_modeleval.inner_product(v, v))
    # --------------------------------------------------------------------------
    # write the state out to a file
    mesh_io.write( 'lowest-energy-state-mu0-0.e', mesh,
                   extra_arrays = { 'e0': v }
                 )
    # --------------------------------------------------------------------------
    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Compute KEO/Jacobian eigenvalues from a given state file.'
                                    )

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'ExodusII file containing the geometry and state'
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

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================