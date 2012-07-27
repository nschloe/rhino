#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Solve the linearized Ginzburg--Landau problem.
'''
# ==============================================================================
from solver_diagnostics import solver_diagnostics # pyamg
from scipy.sparse import spdiags
import numpy as np

import voropy
import pynosh.nls_modelevaluator
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    #print "Reading the mesh...",
    mesh, point_data, field_data = voropy.read(args.filename, timestep=args.timestep)
    psi0 = point_data['psi'][:,0] + 1j * point_data['psi'][:,1]

    # build the model evaluator
    mu = 1.0e-1
    g = 1.0
    num_nodes = len(mesh.node_coords)
    V = -np.ones(num_nodes)
    modeleval = pynosh.nls_modelevaluator.NlsModelEvaluator(mesh, g=g, V=V, A=point_data['A'], mu=mu)

    # build preconditioner
    if modeleval._keo is None:
        modeleval._assemble_keo()
    if mesh.control_volumes is None:
        mesh.compute_control_volumes()

    if modeleval._g > 0:
        alpha = modeleval._g * 2.0 * (psi0.real**2 + psi0.imag**2)
        a = alpha.T * mesh.control_volumes.T
        D = spdiags(alpha.T * mesh.control_volumes.T, [0], num_nodes, num_nodes)
        prec = modeleval._keo + D
    else:
        prec = modeleval._keo

    # https://code.google.com/p/pyamg/source/browse/trunk/Examples/SolverDiagnostics/solver_diagnostics.py
    solver_diagnostics(prec,
                       fname='solver_diagnostic',
                       definiteness='positive',
                       symmetry='hermitian'
                       )
          
    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Does a parameter sweep for PyAMG on K.'
                                    )

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file containing the geometry and initial state'
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
