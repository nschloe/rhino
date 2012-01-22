#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve the Ginzburg--Landau equation.
'''
# ==============================================================================
import mesh_io
import numerical_methods as nm
import sys
import numpy as np
import matplotlib.pyplot as pp

import ginla_modelevaluator
# ==============================================================================
def _main():
    '''Main function.
    '''
    filename = _parse_input_arguments()

    # read the mesh
    print "Reading the mesh...",
    mesh, psi, A, field_data = mesh_io.read_mesh( filename )
    print "done."

    # build the model evaluator
    mu = 8.0e-2
    ginla_modelval = ginla_modelevaluator.GinlaModelEvaluator( mesh, A, mu )

    # initial guess
    num_nodes = len( mesh.nodes )
    psi0 = np.ones( num_nodes,
                    dtype = complex
                  )

    print "Performing Newton iteration...",
    # perform newton iteration
    newton_out = nm.newton( psi0,
                            ginla_modelval,
                            linear_solver = nm.minres,
                            nonlinear_tol = 1.0e-10,
                            forcing_term = 'constant', #'type 2'
                            eta0 = 1.0e-11,
                            use_preconditioner = False,
                            deflate_ix = True
                          )
    print " done."
    print newton_out[2]
    assert( newton_out[1] == 0 )

    multiplot_data_series( newton_out[3] )
    pp.show()

    # energy of the state
    print "Energy of the solution state: %g." % ginla_modelval.energy( newton_out[0] )

    #print "Performing Poor man's continuation..."
    #nm.poor_mans_continuation( psi0,
                               #ginla_modelval,
                               #mu,
                               #initial_step_size = 1.0e-2,
                               #nonlinear_tol = 1.0e-8
                             #)

    ## write the solution to a file
    #sol_filename = "solution.vtu"
    #vtkio.write_mesh( sol_filename, mesh, psi_sol )

    return
# ==============================================================================
def multiplot_data_series( list_of_data_vectors ):
    '''Plot a list of data vectors with increasing black value.'''
    num_plots = len( list_of_data_vectors )
    for k, relresvec in enumerate(list_of_data_vectors):
        pp.semilogy(relresvec, color=str(1.0 - float(k+1)/num_plots))
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Find solutions to the Ginzburg--Landau equation.' )

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'ExodusII file containing the geometry and initial state'
                       )

    args = parser.parse_args()
    
    return args.filename
# ==============================================================================
if __name__ == "__main__":
    _main()

    #import cProfile
    #cProfile.run( '_main()', 'pfvm_profile.dat' )
# ==============================================================================
