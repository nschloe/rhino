#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Solve the linearized Ginzburg--Landau problem.
'''
# ==============================================================================
from solver_diagnostics import solver_diagnostics # pyamg
from scipy.sparse import spdiags

import mesh.mesh_io
import pyginla.ginla_modelevaluator
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    #print "Reading the mesh...",
    pyginlamesh, psi0, A, field_data = \
        mesh.mesh_io.read_mesh(args.filename, timestep=args.timestep)


    # build the model evaluator
    mu = 1.0e-1
    ginla_modelval = pyginla.ginla_modelevaluator.GinlaModelEvaluator(pyginlamesh, A, mu)

    # build preconditioner
    if ginla_modelval._keo is None:
        ginla_modelval._assemble_keo()
    if ginla_modelval.control_volumes is None:
        ginla_modelval._compute_control_volumes()
    num_nodes = len(ginla_modelval.control_volumes)
    absPsi0Squared = psi0.real**2 + psi0.imag**2
    D = spdiags(2 * absPsi0Squared.T * ginla_modelval.control_volumes.T, [0],
                num_nodes, num_nodes)
    prec = ginla_modelval._keo + D

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
                         help    = 'ExodusII file containing the geometry and initial state'
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
