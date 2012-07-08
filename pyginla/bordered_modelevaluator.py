# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse.linalg import LinearOperator
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class BorderedModelEvaluator:
    '''Wraps a given model evaluator in a bordering strategy.
    Does not work with preconditioners.
    '''
    # ==========================================================================
    def __init__(self, modeleval, psi0):
        '''Initialization.
        '''
        self.psi0 = psi0.copy()
        self.inner_modeleval = modeleval
        self.dtype = self.inner_modeleval.dtype
        return
    # ==========================================================================
    def compute_f(self, x):
        '''Compute bordered F.
        '''
        n = len(x)-1
        # Set the new psi0. This is essential for the bordering
        # in the Jacobian since -i*psi0 must not be in the
        # range of the inner Jacobian which is the case
        # for i*x0.
        self.psi0 = x[0:n]
        res = np.empty((n+1,1), dtype=self.dtype)
        res[0:n] = self.inner_modeleval.compute_f(x[0:n])
        # Right border: -i * psi0 * eta.
        res[0:n] += -1j * self.psi0 * x[n]
        # Lower border: <-i psi0, psi>.
        # With the above setting psi0=x, this is always 0.
        res[n] = self.inner_modeleval.inner_product(-1j*self.psi0, x[0:n])

        return res
    # ==========================================================================
    def get_jacobian(self, x0):
        '''Jacobian of the bordered system.
        '''
        # ----------------------------------------------------------------------
        def _apply_jacobian( x ):
            y = np.empty((n+1,1), dtype=self.dtype)
            y[0:n] = inner_jacobian * x[0:n] \
                   - 1j * self.psi0 * x[n]
            y[n] = self.inner_modeleval.inner_product(-1j*self.psi0, x[0:n])
            return y
        # ----------------------------------------------------------------------
        assert x0 is not None

        n = len(x0) - 1
        inner_jacobian = self.inner_modeleval.get_jacobian(x0[0:n])

        return LinearOperator( (n+1, n+1),
                               _apply_jacobian,
                               dtype = self.dtype
                             )
    # ==========================================================================
    def get_preconditioner(self, psi0):
        # Preconditioner not supported.
        return None
    # ==========================================================================
    def get_preconditioner_inverse(self, psi0):
        # Preconditioner not supported.
        return None
    # ==========================================================================
    def inner_product(self, x0, x1):
        '''The inner product of the bordered problem.
        '''
        n = len(x0) - 1
        return self.inner_modeleval.inner_product(x0[0:n],x1[0:n]) \
             + x0[n] * x1[n]
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
