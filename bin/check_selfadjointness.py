#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
import numpy as np

import mesh.mesh_io
import pyginla.ginla_modelevaluator
#import pyginla.numerical_methods as nm
# ==============================================================================
def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh(args.filename,
                                                             timestep=args.timestep
                                                             )
    # build the model evaluator
    mu = 1.0e-1
    ginla_modeleval = \
        pyginla.ginla_modelevaluator.GinlaModelEvaluator(pyginlamesh, A, mu)

    N = len( pyginlamesh.nodes )
    current_psi = np.random.rand(N,1) + 1j * np.random.rand(N,1)

    print 'machine eps = %g' % np.finfo(np.complex).eps
    
    # check the jacobian operator
    J = ginla_modeleval.get_jacobian(current_psi)
    print 'max(|<v,Ju> - <Jv,u>|) = %g' % _check_selfadjointness(J, ginla_modeleval.inner_product)

    # check the preconditioner
    P = ginla_modeleval.get_preconditioner(current_psi)
    print np.linalg.norm( (P - P.T.conj()).todense() )
    print 'max(|<v,Pu> - <Pv,u>|) = %g' % _check_selfadjointness(P)

    ## check the preconditioner
    #Pinv = ginla_modeleval._get_preconditioner_inverse_amg(current_psi)
    #print 'max(|<v,P^{-1}u> - <P^{-1}v,u>|) = %g' % _check_selfadjointness(Pinv)

    return
# ==============================================================================
def _check_selfadjointness( operator, inner_product ):
    N = operator.shape[0]
    num_samples = 1000
    max_discrepancy = 0.0
    for k in xrange(num_samples):
        u = np.random.rand(N,1) + 1j * np.random.rand(N,1)
        v = np.random.rand(N,1) + 1j * np.random.rand(N,1)
        alpha = inner_product(v, operator*u)[0,0]
        beta = inner_product(operator*v, u)[0,0]
        max_discrepancy = max(max_discrepancy, abs(alpha-beta))

    return max_discrepancy
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Solve the linearized Ginzburg--Landau problem.'
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
