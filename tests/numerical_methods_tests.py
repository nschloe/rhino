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
    def _create_sparse_hpd_matrix ( self, num_unknowns ):
        o = np.ones(num_unknowns)
        L = scipy.sparse.spdiags( [-o, 2.*o, -o], [-1,0,1], num_unknowns, num_unknowns)
        V = scipy.sparse.lil_matrix( (num_unknowns, num_unknowns) , dtype=complex)
        for i in range(0,num_unknowns):
            if i % 2:
                v = 1.
            else:
                v = 1.0j
            V[i,i] = v
        A = V.T.conj() * L * V
        # A has the same eigenvalues as L and A is Hermitian but non-symmetric
        return A.tocsr()
    # --------------------------------------------------------------------------
    def _create_sparse_herm_indef_matrix ( self, num_unknowns ):
        # Check if number of unknowns is multiple of 2
        self.assertEqual(num_unknowns % 2, 0)

        # Create block diagonal matrix [A,0; 0,-A] with a HPD block A
        A = self._create_sparse_hpd_matrix( num_unknowns/2 )
        return scipy.sparse.bmat([[A,None],[None,-A]]).tocsr()
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
        A = self._create_sparse_hpd_matrix( num_unknowns )
        rhs = np.ones(num_unknowns)
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
        rhs = np.ones( num_unknowns )
        x0 = np.zeros( num_unknowns )

        # Solve using MINRES.
        tol = 1.0e-13
        x, info, relresvec = numerical_methods.minres( A, rhs, x0, tol=tol )

        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - np.dot(A, x)
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_sparse_hpd(self):
        # Create sparse HPD problem.
        num_unknowns = 100
        A = self._create_sparse_hpd_matrix(num_unknowns)
        rhs = np.ones( num_unknowns )
        x0 = np.zeros( num_unknowns )

        # Solve using MINRES.
        tol = 1.0e-10
        x, info, relresvec = numerical_methods.minres( A, rhs, x0, tol=tol)

        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A*x
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
        # Is last residual in relresvec equal to explicitly computed residual?
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), relresvec[-1], delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_dense_complex(self):
        # Create regular dense problem.
        num_unknowns = 3
        A = np.array( [[1,-2+1j,0], [-2-1j,2,-1], [0,-1,3]] )
        rhs = np.ones( num_unknowns )
        x0 = np.zeros( num_unknowns )

        # Solve using MINRES.
        tol = 1.0e-14
        x, info, relresvec = numerical_methods.minres( A, rhs, x0, tol=tol )

        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - np.dot(A, x)
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_sparse_indef(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        A = self._create_sparse_herm_indef_matrix(num_unknowns)
        rhs = np.ones(num_unknowns)
        x0 = np.zeros(num_unknowns )

        # Solve using spsolve.
        xexact = scipy.sparse.linalg.spsolve(A, rhs)
        # Solve using MINRES.
        tol = 1.0e-10
        x, info, relresvec, errvec = numerical_methods.minres( A, rhs, x0, tol=tol, maxiter=4*num_unknowns, explicit_residual=True, exact_solution=xexact)

        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A * x
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
        # Check error.
        self.assertAlmostEqual( np.linalg.norm(xexact - x) - errvec[-1], 0.0, delta=1e-10 )
    # --------------------------------------------------------------------------
    def test_minres_lanczos(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        A = self._create_sparse_herm_indef_matrix(num_unknowns)
        rhs = np.ones(num_unknowns)
        x0 = np.zeros(num_unknowns )
        self._create_sparse_herm_indef_matrix(4)

        # Solve using MINRES.
        tol = 1.0e-10
        x, info, relresvec, Vfull, Pfull, Tfull = numerical_methods.minres( A, rhs, x0, tol=tol, maxiter=num_unknowns, explicit_residual=False, return_lanczos=True, full_reortho=True )

        # Make sure the method converged.
        self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A * x
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
        # Check if Lanczos relation holds
        res = A*Vfull[:,0:-1] - Vfull*Tfull
        self.assertAlmostEqual( np.linalg.norm(res), 0.0, delta=1e-9 )
        # Check if Lanczos basis is orthonormal w.r.t. inner product
        res = np.eye(Vfull.shape[1]) - np.dot( Pfull.T.conj(), Vfull )
        self.assertAlmostEqual( np.linalg.norm(res), 0.0, delta=1e-9 )
    # --------------------------------------------------------------------------
    def test_minres_deflation(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        A = self._create_sparse_herm_indef_matrix(num_unknowns)
        rhs = np.ones(num_unknowns)
        x0 = np.zeros(num_unknowns )
        self._create_sparse_herm_indef_matrix(4)

        # get projection
        from scipy.sparse.linalg import eigs
        D, W = eigs(A)
        W = W + 1.e-4*np.random.rand(W.shape[0], W.shape[1])
        from scipy.linalg import qr
        W, R = qr(W, mode='economic')
        P, x0new = numerical_methods.get_projection( A, rhs, x0, W )

        # Solve using MINRES.
        tol = 1.0e-10
        x, info, relresvec, Vfull, Pfull, Tfull = numerical_methods.minres( A, rhs, x0new, Mr=P, tol=tol, maxiter=num_unknowns, full_reortho=True, return_lanczos=True )

        numerical_methods.get_ritz( A, W, Vfull, Tfull )

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
        self.assertAlmostEqual( np.linalg.norm(res) / np.linalg.norm(rhs),
                                0.0,
                                delta=tol )
    # --------------------------------------------------------------------------
    def test_newton(self):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        class QuadraticModelEvaluator:
            '''Simple model evalator for f(x) = x^2 - 2.'''
            def __init__(self):
                self._x0 = None
                self.dtype = float
            def compute_f(self, x):
                return x**2 - 2.0
            def set_current_x(self, x):
                self._x0 = x
            def apply_jacobian(self, x):
                return 2.0 * self._x0 * x
            def inner_product(self, x, y):
                return np.vdot(x, y)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        qmodeleval = QuadraticModelEvaluator()
        x0 = np.array( [1.0] )
        tol = 1.0e-10
        x, error_code, resvec = numerical_methods.newton( x0, qmodeleval,
                                                          nonlinear_tol=tol )
        self.assertEqual(error_code, 0)
        self.assertAlmostEqual(x[0], np.sqrt(2.0), delta=tol)
    # --------------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
