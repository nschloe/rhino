# -*- coding: utf-8 -*-
import numpy as np
import numerical_methods as nm
from scipy.sparse.linalg import LinearOperator
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class BorderedModelEvaluator:
    '''Wraps a given model evaluator in a bordering strategy.
    Does not work with preconditioners.
    '''
    # ==========================================================================
    def __init__(self, modeleval):
        '''Initialization.
        '''
        self.inner_modeleval = modeleval
        self.dtype = self.inner_modeleval.dtype
        return
    # ==========================================================================
    def compute_f(self, x):
        '''Compute bordered F.
        '''
        n = len(x) - 1
        # Set the new psi0. This is essential for the bordering
        # in the Jacobian since -i*psi0 must not be in the
        # range of the inner Jacobian which is the case
        # for i*x0.
        psi0 = x[0:n]
        res = np.empty((n+1,1), dtype=self.dtype)
        # Right border: -i * psi0 * eta.
        res[0:n] = self.inner_modeleval.compute_f(x[0:n]) \
                 - 1j * psi0 * x[n]
        # Lower border: <-i psi0, psi>.
        # With the above setting psi0=x, this is always 0.
        res[n] = self.inner_modeleval.inner_product(-1j*psi0, x[0:n])

        return res
    # ==========================================================================
    def get_jacobian(self, x0):
        '''Jacobian of the bordered system.
        '''
        # ----------------------------------------------------------------------
        def _apply_jacobian( x ):
            y = np.empty(x.shape, dtype=self.dtype)
            y[0:n] = inner_jacobian * x[0:n] + b.reshape(x[0:n].shape) * x[n]
            assert abs(x[n].imag) < 1.0e-15, 'Not real-valued: %r.'  % x[n]
            y[n] = self.inner_modeleval.inner_product(c, x[0:n]) + d * x[n].real
            return y
        # ----------------------------------------------------------------------
        assert x0 is not None

        n = len(x0) - 1
        psi0 = x0[0:n]
        b = - 1j * psi0
        c = - 1j * psi0
        d = 0.0
        inner_jacobian = self.inner_modeleval.get_jacobian(psi0)

        return LinearOperator( (n+1, n+1),
                               _apply_jacobian,
                               dtype = self.dtype
                             )
    # ==========================================================================
    def get_jacobian_inverse(self, x0):
        '''Preconditioner based on Schur complement.'''
        # ----------------------------------------------------------------------
        def _apply_jacobian_inverse(x):
            '''Schur thing.'''
            # Test: Use with Jacobian solved exactly.
            x0new = np.zeros((n, 1), dtype=complex)
            phi0 = x[0:n]
            phi0 = phi0.reshape((n,1))
            out0 = nm.minres(jacobian, phi0,
                             x0new,
                             tol = 1.0e-11,
                             M = prec,
                             maxiter = 500,
                             inner_product = self.inner_modeleval.inner_product,
                             #explicit_residual = True,
                             #timer=timer
                             )
            assert out0['info'] == 0
            z0 = out0['xk'].reshape(n)

            x0new = np.zeros((n, 1), dtype=complex)
            out1 = nm.minres(jacobian, b,
                             x0new,
                             tol = 1.0e-11,
                             M = prec,
                             maxiter = 500,
                             inner_product = self.inner_modeleval.inner_product,
                             #explicit_residual = True,
                             #timer=timer
                             )
            assert out1['info'] == 0
            z1 = out1['xk'].reshape(n)

            # Schur complement.
            s = d - self.inner_modeleval.inner_product(c, z1)
            assert abs(s) > 1.0e-15
            tmp = self.inner_modeleval.inner_product(c, z0)

            assert abs(s.imag) < 1.0e-15
            s = s.real
            assert abs(tmp.imag) < 1.0e-15
            tmp = tmp.real
            assert abs(x[n].imag) < 1.0e-15
            xn = x[n].real

            y = np.empty(x.shape, dtype=self.dtype)
            y[0:n] = z0 + z1 * 1.0/s * tmp - z1 / s * xn
            y[n] = - 1.0/s * tmp + 1.0/s * xn

            return y
        # ----------------------------------------------------------------------
        n = len(x0) - 1
        psi0 = x0[0:n]

        b = - 1j * psi0
        c = - 1j * psi0
        d = 0.0

        jacobian = self.inner_modeleval.get_jacobian(psi0)
        prec = self.inner_modeleval.get_preconditioner_inverse(psi0)

        return LinearOperator((n+1, n+1),
                              _apply_jacobian_inverse,
                              dtype = self.dtype
                              )
    # ==========================================================================
    def get_preconditioner(self, psi0):
        # Preconditioner not supported.
        return None
    # ==========================================================================
    def get_preconditioner_inverse(self, x0):
        '''Preconditioner based on Schur complement.'''
        # ----------------------------------------------------------------------
        def _apply_precon_inverse(x):
            '''Schur thing.'''
            # Approximate solutions of
            #    J^{-1} phi
            # and
            #    J^{-1} b.
            z0 = prec * x[0:n]
            z1 = prec * b.reshape(z0.shape)

            # Schur complement.
            s = d - self.inner_modeleval.inner_product(c, z1)
            assert abs(s) > 1.0e-15
            tmp = self.inner_modeleval.inner_product(c, z0)

            assert abs(s.imag) < 1.0e-15
            s = s.real
            assert abs(tmp.imag) < 1.0e-15
            tmp = tmp.real
            assert abs(x[n].imag) < 1.0e-15
            xn = x[n].real

            y = np.empty(x.shape, dtype=self.dtype)
            y[0:n] = z0 + z1 * 1.0/s * tmp - z1 / s * xn
            y[n] = - 1.0/s * tmp + 1.0/s * xn

            return y
        # ----------------------------------------------------------------------
        n = len(x0) - 1
        psi0 = x0[0:n]

        b = - 1j * psi0
        c = - 1j * psi0
        d = 0.0

        prec = self.inner_modeleval.get_preconditioner_inverse(psi0)
        if prec:
            return LinearOperator((n+1, n+1),
                                  _apply_precon_inverse,
                                  dtype = self.dtype
                                  )
        else:
            return None
    # ==========================================================================
    def inner_product(self, x0, x1):
        '''The inner product of the bordered problem.
        '''
        n = len(x0) - 1
        return self.inner_modeleval.inner_product(x0[0:n], x1[0:n]) \
             + x0[n] * x1[n]
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
