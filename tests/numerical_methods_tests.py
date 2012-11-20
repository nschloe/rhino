import pynosh.numerical_methods as nm
import numpy as np
import scipy
import unittest
# ==============================================================================
class TestLinearSolvers(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _create_herm_matrix( self, num_unknowns ):
        # create hermitian toeplitz matrix
        c = np.r_[num_unknowns, np.array(range(num_unknowns-1,0,-1)) * (1+1j)/ (np.sqrt(2)*num_unknowns) ]
        import scipy.linalg
        A = scipy.linalg.toeplitz(c)
        return A
    # --------------------------------------------------------------------------
    def _create_hpd_matrix( self, num_unknowns ):
        A = self._create_herm_matrix( num_unknowns )
        # Make sure that the lowest eigenvalue is 1.
        D = np.linalg.eigvalsh( A )
        A = A + (1.0-min(D)) * np.eye( num_unknowns )
        return A
    # --------------------------------------------------------------------------
    def _create_sym_indef_matrix( self, num_unknowns ):
        A = self._create_herm_matrix( num_unknowns )
        # Make sure that the lowest eigenvalue is -1 and the largest 1.
        D,V = np.linalg.eig( A )
        I = np.eye( num_unknowns )
        A = 2.0 / (max(D)-min(D)) * (A - min(D) * I) - 1.0 * I
        return A
    # --------------------------------------------------------------------------
    def _create_sparse_hpd_matrix(self, num_unknowns):
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
    def _create_sparse_herm_indef_matrix(self, num_unknowns):
        # Check if number of unknowns is multiple of 2
        self.assertEqual(num_unknowns % 2, 0)

        # Create block diagonal matrix [A,0; 0,-A] with a HPD block A
        A = self._create_sparse_hpd_matrix(int(num_unknowns/2))
        return scipy.sparse.bmat([[A,None],[None,-A]]).tocsr()
    def _create_sparse_nonherm_matrix(self, num_unknowns, b, c):
        h = 1.0/(num_unknowns+1)
        data = np.array([ (-1.0/h**2 - b/(2*h)) * np.ones(num_unknowns),
                           (2.0/h**2 + c ) * np.ones(num_unknowns),
                          (-1.0/h**2 + b/(2*h)) * np.ones(num_unknowns)
                       ])
        diags = np.array([-1,0,1])
        return scipy.sparse.spdiags(data, diags, num_unknowns, num_unknowns)
    # --------------------------------------------------------------------------
    def test_cg_dense(self):
        # Create dense problem.
        num_unknowns = 5
        A = self._create_hpd_matrix( num_unknowns )
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using CG.
        tol = 1.0e-13
        out = nm.cg( A, rhs, x0, tol=tol )

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - np.dot(A, out['xk'])
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_cg_sparse(self):
        # Create sparse problem.
        num_unknowns = 100
        A = self._create_sparse_hpd_matrix( num_unknowns )
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using CG.
        tol = 1.0e-11
        out = nm.cg(A, rhs, x0, tol=tol)

        # Make sure the method converged.
        #self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A * out['xk']
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_cg_sparse_prec(self):
        # Create sparse problem.
        num_unknowns = 100
        A = self._create_sparse_hpd_matrix( num_unknowns )
        M = self._create_hpd_matrix( num_unknowns )
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using CG.
        tol = 1.0e-11
        out = nm.cg(A, rhs, x0, tol=tol, maxiter=2*num_unknowns, M=M)

        # Make sure the method converged.
        #self.assertEqual(info, 0)
        # Check the residual.
        res = rhs - A * out['xk']
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_dense(self):
        # Create regular dense problem.
        num_unknowns = 5
        A = self._create_sym_indef_matrix( num_unknowns )
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using MINRES.
        tol = 1.0e-13
        out = nm.minres( A, rhs, x0, tol=tol )

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - np.dot(A, out['xk'])
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_sparse_hpd(self):
        # Create sparse HPD problem.
        num_unknowns = 100
        A = self._create_sparse_hpd_matrix(num_unknowns)
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using MINRES.
        tol = 1.0e-10
        out = nm.minres( A, rhs, x0, tol=tol)

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - A*out['xk']
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
        # Is last residual in relresvec equal to explicitly computed residual?
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), out['relresvec'][-1], delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_dense_complex(self):
        # Create regular dense problem.
        num_unknowns = 3
        A = np.array( [[1,-2+1j,0], [-2-1j,2,-1], [0,-1,3]] )
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using MINRES.
        tol = 1.0e-14
        out = nm.minres( A, rhs, x0, tol=tol )

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - np.dot(A, out['xk'])
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_minres_sparse_indef(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        A = self._create_sparse_herm_indef_matrix(num_unknowns)
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using spsolve.
        xexact = scipy.sparse.linalg.spsolve(A, rhs)
        xexact = np.reshape(xexact, (len(xexact),1))
        # Solve using MINRES.
        tol = 1.0e-10
        out = nm.minres( A, rhs, x0, tol=tol, maxiter=4*num_unknowns, explicit_residual=True, exact_solution=xexact)

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - A * out['xk']
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
        # Check error.
        self.assertAlmostEqual( np.linalg.norm(xexact - out['xk']) - out['errvec'][-1], 0.0, delta=1e-10 )
    # --------------------------------------------------------------------------
    def test_minres_sparse_indef_precon(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        A = self._create_sparse_herm_indef_matrix(num_unknowns)
        M = self._create_hpd_matrix( num_unknowns )
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # Solve using spsolve.
        xexact = scipy.sparse.linalg.spsolve(A, rhs)
        xexact = np.reshape(xexact, (len(xexact),1))
        # Solve using MINRES.
        tol = 1.0e-10
        out = nm.minres( A, rhs, x0, tol=tol, maxiter=100*num_unknowns, explicit_residual=True, exact_solution=xexact, M=M)

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)

        # compute M-norm of residual
        res = rhs - A * out['xk']
        Mres = np.dot(M, res)
        norm_res = np.sqrt(np.vdot(res, Mres))

        # compute M-norm of rhs
        Mrhs = np.dot(M, rhs)
        norm_rhs = np.sqrt(np.vdot(rhs, Mrhs))

        # Check the residual.
        self.assertAlmostEqual( norm_res/norm_rhs, 0.0, delta=tol )
        # Check error.
        self.assertAlmostEqual( np.linalg.norm(xexact - out['xk']) - out['errvec'][-1], 0.0, delta=1e-10 )
    # --------------------------------------------------------------------------
    def test_minres_lanczos(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        A = self._create_sparse_herm_indef_matrix(num_unknowns)
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )
        self._create_sparse_herm_indef_matrix(4)

        # Solve using MINRES.
        tol = 1.0e-9
        out = nm.minres( A, rhs, x0, tol=tol, maxiter=num_unknowns,
                         explicit_residual=False, return_basis=True,
                         full_reortho=True
                         )

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - A * out['xk']
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0,
                                delta=tol )
        # Check if Lanczos relation holds
        res = A*out['Vfull'][:,0:-1] - np.dot(out['Vfull'], out['Hfull'])
        self.assertAlmostEqual( np.linalg.norm(res), 0.0, delta=1e-8 )
        # Check if Lanczos basis is orthonormal w.r.t. inner product
        max_independent = min([num_unknowns,out['Vfull'].shape[1]])
        res = np.eye(max_independent) - \
                np.dot( out['Pfull'][:,0:max_independent].T.conj(),
                        out['Vfull'][:,0:max_independent] )
        self.assertAlmostEqual( np.linalg.norm(res), 0.0, delta=1e-8 )
    # --------------------------------------------------------------------------
    def test_minres_deflation(self):
        # Create sparse symmetric problem.
        num_unknowns = 100
        A = self._create_sparse_herm_indef_matrix(num_unknowns)
        rhs = np.ones( (num_unknowns,1) )
        x0 = np.zeros( (num_unknowns,1) )

        # get projection
        from scipy.sparse.linalg import eigs
        num_vecs = 6
        D, W = eigs(A, k=num_vecs, v0=np.ones((num_unknowns,1)))

        # Uncomment next lines to perturb W
        #W = W + 1.e-10*np.random.rand(W.shape[0], W.shape[1])
        #from scipy.linalg import qr
        #W, R = qr(W, mode='economic')

        AW = nm._apply(A, W)
        P, x0new = nm.get_projection( W, AW, rhs, x0 )

        # Solve using MINRES.
        tol = 1.0e-9
        out = nm.minres( A, rhs, x0new, Mr=P, tol=tol, maxiter=num_unknowns-num_vecs, full_reortho=True, return_basis=True )

        # TODO: move to new unit test
        o = nm.get_p_ritz( W, AW, out['Vfull'], out['Hfull'] )
        ritz_vals, ritz_vecs = nm.get_p_ritz( W, AW, out['Vfull'], out['Hfull'] )

        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - A * out['xk']
        self.assertAlmostEqual( np.linalg.norm(res)/np.linalg.norm(rhs), 0.0, delta=tol )
    # --------------------------------------------------------------------------
    def test_get_projection(self):
        N = 100
        A = scipy.sparse.spdiags( range(1,N+1), [0], N, N)
        b = np.ones( (N,1) )
        x0 = np.ones( (N,1) )

        # 'last' 2 eigenvectors
        W = np.zeros( (N,2) )
        W[-2,0]=1.
        W[-1,1]=1
        AW = A*W

        P, x0new = nm.get_projection(W, AW, b, x0)

        # Check A*(P*I) against exact A*P
        AP = A*(P*np.eye(N))
        AP_exact = scipy.sparse.spdiags(list(range(1,N-1))+[0,0], [0], N, N)
        self.assertAlmostEqual( np.linalg.norm(AP-AP_exact), 0.0, delta=1e-14 )

        # Check x0new
        x0new_exact = np.ones( (N,1) )
        x0new_exact[N-2:N,0] = [1./(N-1),1./N]
        self.assertAlmostEqual( np.linalg.norm(x0new-x0new_exact), 0.0, delta=1e-14 )

    # --------------------------------------------------------------------------
    def test_get_ritz(self):
        N = 10
        A = scipy.sparse.spdiags( range(1,N+1), [0], N, N)
        b = np.ones( (N,1) )
        x0 = np.ones( (N,1) )

        # 'last' 2 eigenvectors
        W = np.zeros( (N,2) )
        W[-2,0]=1.
        W[-1,1]=1
        AW = A*W

        # Get the projection
        P, x0new = nm.get_projection(W, AW, b, x0)

        # Run MINRES (we are only interested in the Lanczos basis and tridiag matrix)
        out = nm.minres(A, b, x0new, Mr=P, tol=1e-14, maxiter=11,
                        full_reortho=True,
                        return_basis=True
                        )

        # Get Ritz pairs
        ritz_vals, ritz_vecs = nm.get_p_ritz(W, AW,
                                             out['Vfull'],
                                             out['Hfull']
                                             )

        # Check Ritz pair residuals
        #ritz_res_exact = A*ritz_vecs - np.dot(ritz_vecs,np.diag(ritz_vals))
        #for i in range(0,len(ritz_vals)):
            #norm_ritz_res_exact = np.linalg.norm(ritz_res_exact[:,i])
            #self.assertAlmostEqual( abs(norm_ritz_res[i] - norm_ritz_res_exact), 0.0, delta=1e-13 )

        # Check if Ritz values / vectors corresponding to W are still there ;)
        order = np.argsort(ritz_vals)
        # 'last' eigenvalue
        self.assertAlmostEqual( abs(ritz_vals[order[-1]] - N), 0.0, delta=1e-13 )
        self.assertAlmostEqual( abs(ritz_vals[order[-2]] - (N-1)), 0.0, delta=1e-13 )
        # now the eigenvectors
        self.assertAlmostEqual( abs(nm._ipstd(ritz_vecs[:,order[-1]],W[:,1])), 1.0, delta=1e-13 )
        self.assertAlmostEqual( abs(nm._ipstd(ritz_vecs[:,order[-2]],W[:,0])), 1.0, delta=1e-13 )

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
        # Create sparse non-Hermitian problem.
        N = 100
        A = self._create_sparse_nonherm_matrix(N, 5*N*(1+2j), 2)
        rhs = np.ones( (N,1) )
        x0 = np.zeros( (N,1) )
        # Solve using GMRES.
        tol = 1.0e-11
        out = nm.gmres( A, rhs, x0, tol=tol )
        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - A * out['xk']
        self.assertAlmostEqual( np.linalg.norm(res) / np.linalg.norm(rhs),
                                0.0,
                                delta=tol )
    # --------------------------------------------------------------------------
    def test_gmres_sparse_Minner(self):
        # Create sparse non-Hermitian problem.
        N = 100
        A = self._create_sparse_nonherm_matrix(N, 5*N*(1+2j), 2)
        rhs = np.ones( (N,1) )
        x0 = np.zeros( (N,1) )
        M = scipy.sparse.spdiags( range(1,N+1), [0], N, N)
        # Solve using GMRES.
        tol = 1.0e-11
        out = nm.gmres( A, rhs, x0, tol=tol, M=M )
        # Make sure the method converged.
        self.assertEqual(out['info'], 0)
        # Check the residual.
        res = rhs - A * out['xk']
        Mres = M*res
        norm_res = np.sqrt(np.dot(res.T.conj(), Mres))
        self.assertAlmostEqual( norm_res / np.linalg.norm(rhs),
                                0.0,
                                delta=tol )
    # --------------------------------------------------------------------------
    def test_newton(self):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        class QuadraticModelEvaluator:
            '''Simple model evalator for f(x) = x^2 - 2.'''
            def __init__(self):
                self.dtype = float
            def compute_f(self, x):
                return x**2 - 2.0
            def get_jacobian(self, x0):
                return 2.0 * x0
            def get_preconditioner(self, x0):
                return None
            def get_preconditioner_inverse(self, x0):
                return None
            def inner_product(self, x, y):
                return np.dot(x.T.conj(), y)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        qmodeleval = QuadraticModelEvaluator()
        x0 = np.array( [[1.0]] )
        tol = 1.0e-10
        out = nm.newton(x0, qmodeleval, nonlinear_tol=tol)
        self.assertEqual(out['info'], 0)
        self.assertAlmostEqual(out['x'][0,0], np.sqrt(2.0), delta=tol)
    # --------------------------------------------------------------------------
    #def test_jacobi_davidson(self):
        #num_unknowns = 10
        ##A = self._create_sym_matrix(num_unknowns)
        #A = self._create_sparse_herm_indef_matrix(num_unknowns)
        #v0 = np.ones((num_unknowns, 1))
        #tol = 1.0e-8
        #out = nm.jacobi_davidson(A, v0, tol = tol )
        #self.assertEqual(out[2], 0) # info == 0
        ## check eigenresidual
        #res = A*out[1] - out[0]*out[1]
        #assert np.norm(res, res) < tol
    # --------------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
