import numerical_methods
import numpy as np
import scipy
import unittest
# ==============================================================================
class TestLinearSolvers(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _create_sym_matrix( self, num_unknowns ):
        A = np.random.rand(num_unknowns, num_unknowns)
        A = 0.5 * ( A + A.transpose() )
        return A
    # --------------------------------------------------------------------------
    def _create_spd_matrix( self, num_unknowns ):
        A = self._create_sym_matrix( num_unknowns )
        # Make sure that the lowest eigenvalue is 1.
        D,V = np.linalg.eig( A )
        A = A + (1.0-min(D)) * np.eye( num_unknowns )
        return A
    # --------------------------------------------------------------------------
    def _create_sym_indef_matrix( self, num_unknowns ):
        A = self._create_sym_matrix( num_unknowns )
        # Make sure that the lowest eigenvalue is -1 and the largest 1.
        D,V = np.linalg.eig( A )
        I = np.eye( num_unknowns ) 
        A = 2.0 / (max(D)-min(D)) * (A - min(D) * I) - 1.0 * I
        return A
    # --------------------------------------------------------------------------
    def test_cg_dense(self):
        # Create dense problem.
        num_unknowns = 5
        A = self._create_spd_matrix( num_unknowns )
        rhs = np.random.rand(num_unknowns)
        x0 = np.zeros( num_unknowns )
        # Solve using CG.
        tol = 1.0e-13
        x, info = numerical_methods.cg( A, rhs, x0, tol=tol )
        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - np.dot(A, x)
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_cg_sparse(self):
        # Create sparse problem.
        num_unknowns = 100
        # A = C^H C  with C=diag(0,a,b)
        a = 10
        b = 5.14 + 5j
        diag = a*np.ones(num_unknowns)
        supdiag = b*np.ones(num_unknowns)
        data = np.array([ diag, supdiag ])
        C = scipy.sparse.spdiags(data, [0, 1], num_unknowns, num_unknowns) 
        A = C.transpose().conj() * C
        rhs = np.random.rand(num_unknowns)
        x0 = np.zeros( num_unknowns )
        # Solve using CG.
        tol = 1.0e-11
        x, info, relresvec, _ = numerical_methods.cg_wrap( A, rhs, x0, tol=tol)
        # Make sure the method converged.
        #self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A * x
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_dense(self):
        # Create regular dense problem.
        num_unknowns = 5
        A = self._create_sym_indef_matrix( num_unknowns )
        rhs = np.random.rand( num_unknowns )
        x0 = np.zeros( num_unknowns )
        # Solve using MINRES.
        tol = 1.0e-13
        x, info = numerical_methods.minres( A, rhs, x0, tol=tol )
        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - np.dot(A, x)
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_sparse_spd(self):
        # Create sparse HPD problem.
        num_unknowns = 100
        # A = C^H C  with C=diag(0,a,b)
        a = 10
        b = 5.14 + 5j
        diag = a*np.ones(num_unknowns)
        supdiag = b*np.ones(num_unknowns)
        data = np.array([ diag, supdiag ])
        C = scipy.sparse.spdiags(data, [0, 1], num_unknowns, num_unknowns) 
        A = C.transpose().conj() * C
        rhs = np.random.rand( num_unknowns )
        x0 = np.zeros( num_unknowns )
        # Solve using MINRES.
        tol = 1.0e-13
        x, info = numerical_methods.minres( A, rhs, x0, tol=tol )
        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A*x
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_dense_complex(self):
        # Create regular dense problem.
        num_unknowns = 3
        A = np.array( [[1,-2+1j,0], [-2-1j,2,-1], [0,-1,3]] )
        rhs = np.ones( num_unknowns )
        x0 = np.zeros( num_unknowns )
        # Solve using MINRES.
        tol = 1.0e-15
        x, info = numerical_methods.minres( A, rhs, x0, tol=tol )
        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - np.dot(A, x)
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_sparse_indef(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        # A = C^H C  with C=diag(0,a,b)
        a = 10
        b = 5.14 + 5j
        diag = range(1,num_unknowns+1);
        diag.extend(range(-1,-num_unknowns-1,-1))
        data = np.array(diag)
        A = scipy.sparse.spdiags(data, [0], 2*num_unknowns, 2*num_unknowns) 
        rhs = np.random.rand(2*num_unknowns)
        x0 = np.zeros(2*num_unknowns )

        # Solve using MINRES.
        tol = 1.0e-11
        x, info = numerical_methods.minres( A, rhs, x0, tol=tol, maxiter=4*num_unknowns )
        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A * x
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
#    def test_lobpcg(self):
#        num_unknowns = 5
#        data = np.array([ -1.0 * np.ones(num_unknowns),
#                           2.0 * np.ones(num_unknowns),
#                          -1.0 * np.ones(num_unknowns)
#                       ])
#        diags = np.array([-1,0,1])
#        A = scipy.sparse.spdiags(data, diags, num_unknowns, num_unknowns)
#        rhs = np.random.rand(num_unknowns)
#        x0 = np.zeros( (num_unknowns,1) )
#        x0[:,0] = np.random.rand( num_unknowns )
#        w, v = scipy.sparse.linalg.lobpcg(A, x0, maxiter=10000, largest=True, verbosityLevel=0 )
#        tol = 1.0e-11
#        res = A*v - w*v
#        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
#    # --------------------------------------------------------------------------
    def test_gmres_sparse(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        h = 1.0/(num_unknowns+1)
        b = 5*num_unknowns*(1 + 2j)
        c = 2
        data = np.array([ (-1.0/h**2 - b/(2*h)) * np.ones(num_unknowns),
                           (2.0/h**2 + c ) * np.ones(num_unknowns),
                          (-1.0/h**2 + b/(2*h)) * np.ones(num_unknowns)
                       ])
        diags = np.array([-1,0,1])
        A = scipy.sparse.spdiags(data, diags, num_unknowns, num_unknowns)
        rhs = np.random.rand(num_unknowns)
        x0 = np.zeros( num_unknowns )
        # Solve using CG.
        tol = 1.0e-11
        x, info = numerical_methods.gmres( A, rhs, x0, tol=tol )
        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A * x
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
