#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Timings.
'''
# ==============================================================================
import timeit
# ==============================================================================
def _main():
    args = _parse_input_arguments()

    create_modeleval = '''
import numpy as np
import pyginla.gp_modelevaluator as gpm
import voropy
mesh, point_data, field_data = voropy.read( '%s' )
modeleval = gpm.GrossPitaevskiiModelEvaluator(mesh,
                                              g = field_data['g'],
                                              V = point_data['V'],
                                              A = point_data['A'],
                                              mu = field_data['mu'])
# initial guess
num_nodes = len(mesh.node_coords)
psi0Name = 'psi0'
psi0 = np.reshape(point_data[psi0Name][:,0] + 1j * point_data[psi0Name][:,1],
                      (num_nodes,1))
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
''' % args.filename

    repeats = 1

    my_setup = '''
J = modeleval.get_jacobian(psi0)
'''
    stmt = '''
_ = J * phi0
'''
    # make sure to execute the operation once such that all initializations are performed
    timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
    print stmt
    print timings
    print min(timings)


    my_setup = '''
M = modeleval._get_preconditioner_inverse_amg(psi0, amg_cycles=1)
'''
    stmt = '''
_ = M * phi0
'''
    # make sure to execute the operation once such that all initializations are performed
    timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
    print stmt
    print timings
    print min(timings)


    my_setup = '''
M = modeleval._get_preconditioner_inverse_amg(psi0, amg_cycles=np.inf)
'''
    stmt = '''
_ = M * phi0
'''
    # make sure to execute the operation once such that all initializations are performed
    timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
    print stmt
    print timings
    print min(timings)


    my_setup = '''
import pyginla.numerical_methods as nm
k = 1
J = modeleval.get_jacobian(psi0)
W = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
JW = J * W
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
P = nm.get_projection(W, JW, b, phi0, inner_product = modeleval.inner_product)
'''
    stmt = '''
print P
_ = P * phi0
'''
    # make sure to execute the operation once such that all initializations are performed
    timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
    print stmt
    print timings
    print min(timings)

    retiurn
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Unit timer.' )

    parser.add_argument('filename',
                        metavar = 'FILE',
                        type    = str,
                        help    = 'Mesh file containing the geometry and initial state'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
