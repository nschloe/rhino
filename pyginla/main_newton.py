#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve the Ginzburg--Landau equation.
'''
# ==============================================================================
import mesh_io
import numerical_methods as nm
import sys
import numpy as np

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
    mu = 1.0e-1
    ginla_modelval = ginla_modelevaluator.GinlaModelEvaluator( mesh, A, mu )

    # initial guess
    num_nodes = len( mesh.nodes )
    psi0 = np.ones( num_nodes,
                    dtype = complex
                  )

    print "Performing Newton iteration...",
    # perform newton iteration
    psi_sol, info, iters = nm.newton( psi0,
                                      ginla_modelval,
                                      nonlinear_tol = 1.0e-10,
                                      forcing_term = 'constant'
                                    )
    print " done."
    assert( info == 0 )

    # energy of the state
    print "Energy of the solution state: %g." % ginla_modelval.energy( psi_sol )

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
