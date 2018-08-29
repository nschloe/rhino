# -*- coding: utf-8 -*-
#
'''Timings.
'''
import timeit
import numpy as np
import pynosh.yaml
import datetime
import os

def _main():
    args = _parse_input_arguments()

    ye = pynosh.yaml.YamlEmitter()

    filename = args.filename

    ye.begin_doc()
    ye.add_comment('Timing results with unit_timer.py (%r, %s).' % (os.uname()[1], datetime.datetime.now()))
    ye.begin_map()
    ye.add_key_value('filename', filename)
    ye.add_key_value('number', args.number)
    ye.add_key('tests')

    ye.begin_seq()

    create_modeleval = '''
import numpy as np
import pynosh.modelevaluator_nls as gpm
import voropy
mesh, point_data, field_data = voropy.read( '%s' )
modeleval = gpm.NlsModelEvaluator(mesh,
                                              V = point_data['V'],
                                              A = point_data['A'],
                                              )
# initial guess
num_nodes = len(mesh.node_coords)
psi0Name = 'psi0'
psi0 = np.reshape(point_data[psi0Name][:,0] + 1j * point_data[psi0Name][:,1],
                      (num_nodes,1))
''' % filename

    targetfunctions = {}

    def unit_jacobian():
        my_setup = '''
import pynosh.numerical_methods as nm
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
J = modeleval.get_jacobian(psi0, field_data['mu'], field_data['g'])
'''
        stmt = '''
_ = J * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['jacobian'] = unit_jacobian


    def unit_cycles():
        my_setup = '''
import pynosh.numerical_methods as nm
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
modeleval._preconditioner_type = 'cycles'
modeleval._num_amg_cycles = 1
Minv = modeleval.get_preconditioner_inverse(psi0, field_data['mu'], field_data['g'])
'''
        stmt = '''
_ = Minv * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['cycles'] = unit_cycles


    def unit_exact():
        my_setup = '''
import pynosh.numerical_methods as nm
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
modeleval._preconditioner_type = 'exact'
Minv = modeleval.get_preconditioner_inverse(psi0, field_data['mu'], field_data['g'])
'''
        stmt = '''
_ = Minv * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['exact'] = unit_exact

    def unit_prec():
        my_setup = '''
import pynosh.numerical_methods as nm
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
modeleval._preconditioner_type = 'exact'
M = modeleval.get_preconditioner(psi0, field_data['mu'], field_data['g'])
'''
        stmt = '''
_ = M * phi0
'''
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['prec'] = unit_prec

    def unit_projection():
        runs = []
        for num_defl_vecs in args.num_defl_vecs:
            my_setup = '''
import pynosh.numerical_methods as nm
k = %d
J = modeleval.get_jacobian(psi0)
W = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
JW = J * W
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
P, x0new = nm.get_projection(W, JW, b, phi0, inner_product = modeleval.inner_product)
''' % num_defl_vecs
            stmt = '''
_ = P * phi0
'''
            # make sure to execute the operation once such that all initializations are performed
            runs.append ( {'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number),
                    'k': num_defl_vecs} )
        return runs
    targetfunctions['projection'] = unit_projection

    def unit_inner():
        my_setup = '''
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi1 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
'''
        stmt = '''
_ = modeleval.inner_product(phi0, phi1)
'''
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['inner'] = unit_inner

    def unit_daxpy():
        my_setup = '''
phi0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
phi1 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
'''
        stmt = '''
_ = phi0 + phi1
'''
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['daxpy'] = unit_daxpy

    minres_setup = '''
# MINRES code copied & pasted
from pynosh.numerical_methods import _apply, _norm
N = len(b)
Mlb = _apply(Ml, b)
MMlb = _apply(M, Mlb)
norm_MMlb = _norm(Mlb, MMlb, inner_product = inner_product)
r0 = b - _apply(A, x0)
Mlr0 = _apply(Ml, r0)
MMlr0 = _apply(M, Mlr0)
norm_MMlr0 = _norm(Mlr0, MMlr0, inner_product = inner_product)
relresvec = [norm_MMlr0 / norm_MMlb]

V = np.c_[np.zeros(N), MMlr0 / norm_MMlr0]
P = np.c_[np.zeros(N), Mlr0 / norm_MMlr0]
W = np.c_[np.zeros(N), np.zeros(N)]
ts = 0.0           # (non-existing) first off-diagonal entry (corresponds to pi1)
y  = [norm_MMlr0, 0] # first entry is (updated) residual
G2 = np.eye(2)     # old givens rotation
G1 = np.eye(2)     # even older givens rotation ;)
yk = np.zeros((N,1), dtype=complex)
xk = x0.copy()
'''
    minres_step = '''
# out = nm.minres(J, b, phi0, M=M, maxiter=1, inner_product=modeleval.inner_product, timer=True)
# MINRES code copied & pasted
k = 0
while relresvec[-1] > tol and k < maxiter:
    # ---------------------------------------------------------------------
    # Lanczos
    tsold = ts
    z  = _apply(Mr, V[:,[1]])
    z  = _apply(A, z)
    z  = _apply(Ml, z)

    z  = z - tsold * P[:,[0]]
    td = inner_product(V[:,[1]], z)[0,0]
#    if abs(td.imag) > 1.0e-12:
#        print 'Warning (iter %d): abs(td.imag) = %g > 1e-12' % (k+1, abs(td.imag))
    td = td.real
    z  = z - td * P[:,[1]]

    # needed for QR-update:
    R = _apply(G1, [0, tsold])
    R = np.append(R, [0.0, 0.0])

    # Apply the preconditioner.
    v  = _apply(M, z)
    alpha = inner_product(z, v)[0,0]
#    if abs(alpha.imag)>1e-12:
#        print 'Warning (iter %d): abs(alpha.imag) = %g > 1e-12' % (k+1, abs(alpha.imag))
    alpha = alpha.real
    if alpha<0.0:
#        print 'Warning (iter %d): alpha = %g < 0' % (k+1, alpha)
        alpha = 0.0
    ts = np.sqrt( alpha )

    if ts > 0.0:
        P  = np.c_[P[:,[1]], z / ts]
        V  = np.c_[V[:,[1]], v / ts]
    else:
        P  = np.c_[P[:,[1]], np.zeros(N)]
        V  = np.c_[V[:,[1]], np.zeros(N)]

    # store new vectors in full basis

    # ----------------------------------------------------------------------
    # (implicit) update of QR-factorization of Lanczos matrix
    R[2:4] = [td, ts]
    R[1:3] = _apply(G2, R[1:3])
    G1 = G2.copy()
    # compute new givens rotation.
    gg = np.linalg.norm( R[2:4] )
    gc = R[2] / gg
    gs = R[3] / gg
    G2 = np.array([ [gc,  gs],
                    [-gs, gc] ])
    R[2] = gg
    R[3] = 0.0
    y = _apply(G2, y)

    # ----------------------------------------------------------------------
    # update solution
    z  = (V[:,0:1] - R[0]*W[:,0:1] - R[1]*W[:,1:2]) / R[2]
    W  = np.c_[W[:,1:2], z]
    yk = yk + y[0] * z
    y  = [y[1], 0]

    k += 1
# end MINRES iteration
# --------------------------------------------------------------------------
'''

    def unit_minres():
        my_setup = '''
A = modeleval.get_jacobian(psi0)
#modeleval._preconditioner_type = 'exact'
modeleval._preconditioner_type = 'cycles'
modeleval._num_amg_cycles = 1
M = modeleval.get_preconditioner_inverse(psi0, field_data['mu'], field_data['g'])
x0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)

maxiter = 1
inner_product = modeleval.inner_product
Ml = None
Mr = None
tol = 1.0e-15
''' + minres_setup
        stmt = minres_step
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['minres'] = unit_minres

    def unit_minres_full():
        my_setup = '''
A = modeleval.get_jacobian(psi0)
#modeleval._preconditioner_type = 'exact'
modeleval._preconditioner_type = 'cycles'
modeleval._num_amg_cycles = 1
Minv = modeleval.get_preconditioner_inverse(psi0, field_data['mu'], field_data['g'])
x0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
import pynosh.numerical_methods as nm
'''
        stmt = '''
out = nm.minres(A, b, x0, M=Minv, maxiter=1, inner_product=modeleval.inner_product, timer=True)
#print out['times']
'''
        # make sure to execute the operation once such that all initializations are performed
        return [{'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)}]
    targetfunctions['minres-full'] = unit_minres_full



    def unit_setup():
        runs = []
        for num_defl_vecs in args.num_defl_vecs:
            my_setup = '''
A = modeleval.get_jacobian(psi0)
#modeleval._preconditioner_type = 'exact'
modeleval._preconditioner_type = 'cycles'
modeleval._num_amg_cycles = 1
M = modeleval.get_preconditioner(psi0, field_data['mu'], field_data['g'])
Minv = modeleval.get_preconditioner_inverse(psi0, field_data['mu'], field_data['g'])
x0 = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
b = np.random.rand(num_nodes,1) + 1j * np.random.rand(num_nodes,1)
import pynosh.numerical_methods as nm
k = %d
W = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
AW = nm._apply(A, W)
Vfull = np.random.rand(num_nodes,k) + 1j * np.random.rand(num_nodes,k)
Hfull = np.random.rand(k,k-1) + 1j * np.random.rand(k,k-1)
''' % num_defl_vecs
            stmt = '''
Q, R = nm.qr(W, inner_product=modeleval.inner_product)
P0, x0new = nm.get_projection(W, AW, b, x0,
                              inner_product = modeleval.inner_product
                              )
ritz_vals, ritz_vecs, norm_ritz_res = nm.get_ritz(W, AW, Vfull, Hfull,
                                                  M = Minv, Minv=M,
                                                  inner_product = modeleval.inner_product)
'''
            # make sure to execute the operation once such that all initializations are performed
            runs.append({'timings': timeit.repeat(stmt = stmt, setup=create_modeleval+my_setup+stmt, repeat=args.repeats, number=args.number)})
        return runs
    targetfunctions['setup'] = unit_setup



    for target in args.target:

        runs = targetfunctions[target]()

        #ye.add_comment(  [min(timings), np.mean(timings), max(timings)] )
        for run in runs:
            ye.begin_map()
            ye.add_key_value('target', target)
            for k,v in run.items():
                ye.add_key_value(k, v)
            ye.end_map()

    ye.end_seq()
    ye.end_map()
    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Unit timer.' )

    parser.add_argument('filename',
                        metavar = 'FILE',
                        type = str,
                        help = 'Mesh files containing the geometries'
                        )

    parser.add_argument('--target', '-t',
                        nargs = '+',
                        choices = ['jacobian', 'cycles', 'exact', 'prec', 'projection', 'inner', 'daxpy', 'minres-step', 'minres-full', 'setup'],
                        help = 'target for timing benchmark'
                        )

    parser.add_argument('--num-defl-vecs', '-d',
                        type = int,
                        nargs = '+',
                        default = [1],
                        help = 'number of deflation vectors (default: 1)'
                        )

    parser.add_argument('--repeats', '-r',
                        type = int,
                        default = 5,
                        help = 'How often to run the timings including the setup (default: 5)'
                        )

    parser.add_argument('--number', '-n',
                        type = int,
                        default = 1,
                        help = 'How often to run the statement (default: 1)'
                        )

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
