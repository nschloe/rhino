#! /usr/bin/env python
# -*- coding: utf-8 -*-

import vtkio
#import numerical_methods as nm
import sys
from scipy.sparse.linalg import eigen, eigen_symmetric
from scipy.sparse import identity
from scipy.linalg import norm
import time

from lobpcg import lobpcg

#import matplotlib.pyplot as pp
#from matplotlib import rc
#rc( 'text', usetex = True )
#rc( 'font', family = 'serif' )

from model_evaluator import *
#from lobpcg import lobpcg
#from preconditioners import *
#import matplotlib2tikz
# ==============================================================================
def _main():
    '''
    Main function.
    '''
    # --------------------------------------------------------------------------
    opts, args = _parse_input_arguments()
    # --------------------------------------------------------------------------
    # read the mesh
    print "Reading the state (and mesh)..."
    try:
        mesh = vtkio.read_mesh( opts.filename )
    except AttributeError:
        print "Could not read from file ", opts.filename, "."
        sys.exit()
    print " done."
    # --------------------------------------------------------------------------
    # build the model evaluator
    mu = 0.0
    ginla_modelval = ginla_model_evaluator( mu )
    ginla_modelval.set_mesh( mesh )
    # --------------------------------------------------------------------------
    # build the kinetic energy operator
    print "Building the KEO..."
    start_time = time.clock()
    ginla_modelval._assemble_kinetic_energy_operator()
    end_time = time.clock()
    print "done (", end_time - start_time, 's).'
    # --------------------------------------------------------------------------
    # find the lowest energy eigenstate
    # try eigen_symmetric
    # prepare initial guess
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    (m,n) = ginla_modelval._keo.shape
    X = np.zeros( (n,1) )
    X[0] = 1.0
    eigenvalues, eigenvectors = lobpcg( ginla_modelval._keo,
                                        X,
                                        B = None,
                                        M = None,
                                        Y = None,
                                        tol = None,
                                        maxiter = 20,
                                        largest = False, # get smallest
                                        verbosityLevel = 0,
                                        retLambdaHistory = False,
                                        retResidualNormsHistory = False
                                      )
    print 'eigenvalues', eigenvalues
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print "Find lowest energy eigenstate/-value..."
    start_time = time.clock()
    lambd, v = eigen( ginla_modelval._keo,
                      k = 1,
                      #M = None,
                      sigma = None,
                      which = 'SM',
                      #v0 = None,
                      #ncv = None,
                      maxiter = 1e10,
                      #tol = 1.0e-10,
                      return_eigenvectors = True
                    )
    end_time = time.clock()
    print "done (", end_time - start_time, 's).'
    v = v[:,0]
    print lambd[0]
    print norm( v )
    # normalize the state in the discretized l2-norm
    v /= ginla_modelval.norm( v )
    print abs( v )
    print ginla_modelval.norm( v )
    # --------------------------------------------------------------------------
    # write the state out to a file
    vtkio.write_mesh( 'lowest-energy-state-mu0-0.vtu', mesh, X = v )
    # --------------------------------------------------------------------------
    return
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