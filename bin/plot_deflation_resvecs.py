#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve the Ginzburg--Landau equation.
'''
# ==============================================================================
import pyginla.numerical_methods as nm
import pyginla.ginla_modelevaluator as gm
import mesh.mesh_io
import numpy as np
import matplotlib.pyplot as pp
#import matplotlib2tikz
# ==============================================================================
def _main():
    '''Main function.
    '''
    filename = _parse_input_arguments()

    # read the mesh
    print "Reading the mesh...",
    pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )
    print "done."

    # build the model evaluator
    mu = 8.0e-2
    ginla_modelval = gm.GinlaModelEvaluator( pyginlamesh, A, mu )

    # initial guess
    num_nodes = len( pyginlamesh.nodes )
    psi0 = np.ones( (num_nodes,1),
                    dtype = complex
                  )

    nums_deflation_vectors = range(10)
    last_resvecs = []
    for num_deflation_vectors in nums_deflation_vectors:
        print 'Performing Newton iteration with %d deflation vectors...' % num_deflation_vectors
        # perform newton iteration
        newton_out = nm.newton( psi0,
                                ginla_modelval,
                                linear_solver = nm.minres,
                                nonlinear_tol = 1.0e-10,
                                forcing_term = 'constant', #'type 2'
                                eta0 = 1.0e-13,
                                use_preconditioner = True,
                                deflate_ix = True,
                                num_deflation_vectors = num_deflation_vectors
                              )
        print " done."
        assert newton_out[1] == 0, 'Newton did not converge.'

        last_resvecs.append(newton_out[3][-1])

    multiplot_data_series(last_resvecs)
    pp.title('Residual curves for the last Newton step. Darker=More deflation vectors.')
    pp.show()
    #matplotlib2tikz.save('w-defl.tex')

    return
# ==============================================================================
def multiplot_data_series( list_of_data_vectors ):
    '''Plot a list of data vectors with increasing black value.'''
    num_plots = len( list_of_data_vectors )
    for k, relresvec in enumerate(list_of_data_vectors):
        pp.semilogy(relresvec, color=str(1.0 - float(k+1)/num_plots))
    pp.xlabel('MINRES step')
    pp.ylabel('||r||/||b||')
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
