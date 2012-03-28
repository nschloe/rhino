#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Compute the eigenvalues of the Jacobian operator for a number of states.
'''
# ==============================================================================
import numpy as np

import voropy
from pyginla import ginla_modelevaluator as gm
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
# ==============================================================================
def _main():
    '''Main function.
    '''
    # parse input arguments
    args = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    mesh, point_data, field_data = voropy.read( args.filename )
    print 'done.'

    # build the model evaluator
    if 'mu' in field_data:
        mu = field_data['mu']
        print 'Using mu=%g as found in file.' % mu
    else:
        mu = 0.2
        print 'Using mu=%g.' % mu
    ginla_modeleval = gm.GinlaModelEvaluator(mesh, point_data['A'], mu)

    # state psi0
    num_nodes = len(mesh.node_coords)
    if 'psi' in point_data:
        point_data['psi'] = point_data['psi'][:,0] \
                          + 1j * point_data['psi'][:,1]
        psi0 = np.reshape(point_data['psi'], (num_nodes,1))
    else:
        psi0 = 1.0 * np.ones((num_nodes,1), dtype=complex)

    # get jacobian
    jacobian = ginla_modeleval.get_jacobian(psi0)

    # Define "real-valued" operator.
    def jac_real(real_phi):
        # convert to complex
        phi = real_phi[::2] + 1j * real_phi[1::2]
        out = jacobian * phi
        # convert to real and return
        out_real = np.empty( len(real_phi) )
        out_real[::2] = out.real
        out_real[1::2] = out.imag
        return out_real
    num_unknowns = len(psi0)
    jac_real_op = LinearOperator((2*num_unknowns, 2*num_unknowns),
                                 jac_real,
                                 dtype = float
                                 )

    # get smallesteigenvalues
    print 'Compute smallest-magnitude eigenvalues...'
    lambd = eigs(jacobian,
                  k = 2,
                  #sigma = None,
                  #which = 'SM',
                  #v0 = np.ones((2*num_unknowns,1)),
                  #return_eigenvectors = False
                  )
    print 'done.'
    print lambd

    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Compute eigenvalues of the Jacobian for a given state.' )

    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'ExodusII file containing the geometry and state'
                       )

    return parser.parse_args()
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
