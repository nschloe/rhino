#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Timings.
'''
# ==============================================================================
import timeit
import numpy as np
# ==============================================================================
def _main():
    args = _parse_input_arguments()

    for filename in args.filenames:
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
''' % filename

        repeats = 5

        my_setup = '''
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
J = modeleval.get_jacobian(psi0)
'''
        stmt = '''
_ = J * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
modeleval._preconditioner_type = 'cycles'
modeleval._num_amg_cycles = 1
M = modeleval.get_preconditioner_inverse(psi0)
'''
        stmt = '''
_ = M * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
modeleval._preconditioner_type = 'exact'
M = modeleval.get_preconditioner_inverse(psi0)
'''
        stmt = '''
_ = M * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
import pyginla.numerical_methods as nm
k = 1
J = modeleval.get_jacobian(psi0)
W = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
JW = J * W
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
P, x0new = nm.get_projection(W, JW, b, phi0, inner_product = modeleval.inner_product)
'''
        stmt = '''
_ = P * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt, 1
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
import pyginla.numerical_methods as nm
J = modeleval.get_jacobian(psi0)
k = 10
W = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
JW = J * W
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
P, x0new = nm.get_projection(W, JW, b, phi0, inner_product = modeleval.inner_product)
'''
        stmt = '''
_ = P * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt, 10
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
import pyginla.numerical_methods as nm
k = 50
J = modeleval.get_jacobian(psi0)
W = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
JW = J * W
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
P, x0new = nm.get_projection(W, JW, b, phi0, inner_product = modeleval.inner_product)
'''
        stmt = '''
_ = P * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt, 50
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
import pyginla.numerical_methods as nm
k = 100
J = modeleval.get_jacobian(psi0)
W = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
JW = J * W
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
P, x0new = nm.get_projection(W, JW, b, phi0, inner_product = modeleval.inner_product)
'''
        stmt = '''
_ = P * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt, 100
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi1 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
'''
        stmt = '''
_ = modeleval.inner_product(phi0, phi1)
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi1 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
'''
        stmt = '''
_ = phi0 + phi1
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)


        my_setup = '''
J = modeleval.get_jacobian(psi0)
modeleval._preconditioner_type = 'cycles'
modeleval._num_amg_cycles = 1
M = modeleval.get_preconditioner_inverse(psi0)
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
import pyginla.numerical_methods as nm
'''
        stmt = '''
nm.minres(J, b, phi0, M=M, maxiter=1, inner_product=modeleval.inner_product)
'''
        # make sure to execute the operation once such that all initializations are performed
        print stmt
        timings = timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=repeats, number=1)
        print min(timings), np.mean(timings), max(timings)

    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Unit timer.' )

    parser.add_argument('filenames',
                        metavar = 'FILE',
                        type = str,
                        nargs = '+',
                        help = 'Mesh files containing the geometries'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
