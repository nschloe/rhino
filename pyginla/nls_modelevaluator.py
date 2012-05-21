#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Provide tools for solving the nonlinear Schrödinger equation.
'''
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import spdiags
import cmath
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class NlsModelEvaluator:
    '''Nonlinear Schr"odinger model evaluator class.
    '''
    # ==========================================================================
    def __init__(self, mesh, kappa):
        '''Initialization. Set mesh.
        '''
        self.dtype = complex
        self.mesh = mesh
        self.kappa = kappa
        self._laplacian = None
        self._D = None
        self._edgecoeff_cache = None
        self.num_cycles = []
        return
    # ==========================================================================
    def compute_f( self, psi ):
        '''Computes the nonlinear Schr"odinger residual for a state psi.
        '''
        if self._laplacian is None:
            self._assemble_laplacian()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        res = (- 0.5 * self._laplacian * psi) / self.mesh.control_volumes[:,None]  \
              + self.kappa * abs(psi)**2 * psi

        return res
    # ==========================================================================
    def get_jacobian(self, psi0):
        '''Returns a LinearOperator object that defines the matrix-vector
        multiplication scheme for the Jacobian operator as in

            A phi + B phi*

        with

            A = K + 2*|psi|^2,
            B = diag( psi^2 ).
        '''
        # ----------------------------------------------------------------------
        def _apply_jacobian( phi ):
            x = (- 0.5 * self._laplacian * phi) / self.mesh.control_volumes.reshape(phi.shape) \
                + kAlpha.reshape(phi.shape) * phi \
                + kPsi0Squared.reshape(phi.shape) * phi.conj()
            return x
        # ----------------------------------------------------------------------
        assert psi0 is not None

        if self._laplacian is None:
            self._assemble_laplacian()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        kAlpha = self.kappa * 2.0*(psi0.real**2 + psi0.imag**2)
        kPsi0Squared = self.kappa * psi0**2

        num_unknowns = len(self.mesh.node_coords)
        return LinearOperator( (num_unknowns, num_unknowns),
                               _apply_jacobian,
                               dtype = self.dtype
                             )
    # ==========================================================================
    def get_preconditioner(self, psi0):
        '''Return the preconditioner.
        '''
        # ----------------------------------------------------------------------
        def _apply_precon(phi):
            return (- 0.5 * self._laplacian * phi) / self.mesh.control_volumes.reshape(phi.shape) \
                   + 2.0 * absPsi0Squared.reshape(phi.shape) * phi
        # ----------------------------------------------------------------------
        assert( psi0 is not None )
        num_unknowns = len(self.mesh.node_coords)

        if self._laplacian is None:
            self._assemble_laplacian()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        absPsi0Squared = psi0.real**2 + psi0.imag**2

        return LinearOperator((num_unknowns, num_unknowns),
                              _apply_precon,
                              dtype = self.dtype
                              )
    # ==========================================================================
    def get_preconditioner_inverse(self, psi0):
        return self._get_preconditioner_inverse_amg(psi0)
        #return self._get_preconditioner_inverse_directsolve(psi0)
    # ==========================================================================
    def _get_preconditioner_inverse_amg(self, psi0):
        '''Use AMG to invert M approximately.
        '''
        import pyamg
        # ----------------------------------------------------------------------
        def _apply_inverse_prec_customcg(phi):
            rhs = self.mesh.control_volumes[:, None] * phi
            x0 = np.zeros((num_unknowns, 1), dtype=complex)
            out = nm.cg(prec, rhs, x0,
                        tol = 1.0e-13,
                        M = amg_prec,
                        #explicit_residual = False
                        )
            if out['info'] != 0:
                print 'Preconditioner did not converge; last residual: %g' \
                      % out['relresvec'][-1]
            # Forget about the cycle used to gauge the residual norm.
            self.num_cycles += [len(out['relresvec']) - 1]
            return out['xk']
        # ----------------------------------------------------------------------
        def _apply_inverse_prec_pyamgsolve(phi):
            rhs = self.mesh.control_volumes[:, None] * phi
            x0 = np.zeros((num_unknowns, 1), dtype=complex)
            x = np.empty((num_nodes,1), dtype=complex)
            num_cycles = 1
            residuals = []
            x[:,0] = prec_amg_solver.solve(rhs,
                                            x0 = x0,
                                            maxiter = num_cycles,
                                            tol = 0.0,
                                            accel = None,
                                            residuals=residuals
                                            )
            # Alternative for one cycle:
            # amg_prec = prec_amg_solver.aspreconditioner( cycle='V' )
            # x = amg_prec * rhs
            self.num_cycles += [num_cycles]
            return x
        # ----------------------------------------------------------------------
        if self._laplacian is None:
            self._assemble_laplacian()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        num_nodes = len(self.mesh.node_coords)
        absPsi0Squared = psi0.real**2 + psi0.imag**2
        D = spdiags(2.0 * absPsi0Squared.T * self.mesh.control_volumes.T, [0],
                    num_nodes, num_nodes)

        prec = - 0.5 * self._laplacian + D

        # The preconditioner assumes the eigenvalue 0 iff mu=0 and psi=0.
        # This may lead to problems if mu=0 and the Newton iteration
        # converges to psi=0 for psi0 != 0.
        #import scipy.sparse.linalg
        #lambd, v = scipy.sparse.linalg.eigs(prec, which='SM')
        #assert all(abs(lambd.imag) < 1.0e-15)
        #print '||psi||^2 = %g' % np.linalg.norm(absPsi0Squared)
        #print 'lambda =', lambd.real

        prec_amg_solver = \
            pyamg.smoothed_aggregation_solver(prec,
            strength=('evolution', {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}),
            smooth=('energy', {'weighting': 'local', 'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
            Bimprove=None,
            aggregate='standard',
            presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
            postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
            max_levels=25,
            coarse_solver='pinv'
            )

        num_unknowns = len(psi0)

        precon_type = 'pyamg solve'
        if precon_type == 'custom cg':
            import numerical_methods as nm
            amg_prec = prec_amg_solver.aspreconditioner( cycle='V' )
            return LinearOperator((num_unknowns, num_unknowns),
                                  _apply_inverse_prec_customcg,
                                  dtype = self.dtype
                                  )
        elif precon_type == 'pyamg solve':
            return LinearOperator((num_unknowns, num_unknowns),
                                  _apply_inverse_prec_pyamgsolve,
                                  dtype = self.dtype
                                  )
        else:
            raise ValueError('Unknown preconditioner type \'%s\'.' % precon_type)
    # ==========================================================================
    def _get_preconditioner_inverse_directsolve(self, psi0):
        '''Use a direct solver for M^{-1}.
        '''
        from scipy.sparse.linalg import spsolve
        # ----------------------------------------------------------------------
        def _apply_inverse_prec(phi):
            return spsolve(prec, phi)
        # ----------------------------------------------------------------------
        prec = self.get_preconditioner(psi0)
        num_unknowns = len(psi0)
        return LinearOperator((num_unknowns, num_unknowns),
                              _apply_inverse_prec,
                              dtype = self.dtype
                              )
    # ==========================================================================
    def inner_product( self, phi0, phi1 ):
        '''The natural inner product of the problem.
        '''
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        if len(phi0.shape)==1:
            scaledPhi0 = self.mesh.control_volumes * phi0
        elif len(phi0.shape)==2:
            scaledPhi0 = self.mesh.control_volumes[:,None] * phi0

        # np.vdot only works for vectors, so use np.dot(....T.conj()) here.
        return np.dot(scaledPhi0.T.conj(), phi1).real
    # ==========================================================================
    def energy(self, psi):
        '''Compute the Gibbs free energy.
        Not really a norm, but a good measure for our purposes here.
        '''
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        alpha = -self.inner_product(psi**2, psi**2)

        return alpha.real / self.mesh.control_volumes.sum()
    # ==========================================================================
    def set_parameter(self, kappa):
        '''Update the parameter.
        '''
        self.kappa = kappa
        return
    # ==========================================================================
    def _assemble_laplacian( self ):
        '''Assemble the kinetic energy operator.'''

        # Create the matrix structure.
        num_nodes = len(self.mesh.node_coords)
        self._laplacian = sparse.lil_matrix((num_nodes, num_nodes),
                                            dtype = complex
                                           )
        # Build caches.
        if self._edgecoeff_cache is None:
            self._build_edgecoeff_cache()
        if self.mesh.edges is None:
            self.mesh.create_adjacent_entities()

        # loop over all edges
        for k, node_indices in enumerate(self.mesh.edges['nodes']):
            # Fetch the cached values.
            alpha = self._edgecoeff_cache[k]
            # Sum them into the matrix.
            self._laplacian[node_indices[0], node_indices[0]] -= alpha
            self._laplacian[node_indices[0], node_indices[1]] += alpha
            self._laplacian[node_indices[1], node_indices[0]] += alpha
            self._laplacian[node_indices[1], node_indices[1]] -= alpha

        # transform the matrix into the more efficient CSR format
        self._laplacian = self._laplacian.tocsr()

        return
    # ==========================================================================
    def _build_edgecoeff_cache( self ):
        '''Build cache for the edge coefficients.
        (in 2D: coedge-edge ratios).'''

        # make sure the mesh has edges
        if self.mesh.edges is None:
            self.mesh.create_adjacent_entities()

        num_edges = len(self.mesh.edges)
        self._edgecoeff_cache = np.zeros(num_edges, dtype=float)

        if self.mesh.cells_volume is None:
            self.mesh.create_cells_volume()
        vols = self.mesh.cells_volume

        # Precompute edges.
        edges = self.mesh.node_coords[self.mesh.edges['nodes'][:,1]] \
              - self.mesh.node_coords[self.mesh.edges['nodes'][:,0]]

        # Calculate the edge contributions cell by cell.
        for vol, cell in zip(vols, self.mesh.cells):
            cell_edge_gids = cell['edges']
            # Build the equation system:
            # The equation
            #
            # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>
            #
            # has to hold for all vectors u in the plane spanned by the edges,
            # particularly by the edges themselves.
            A = np.dot(edges[cell_edge_gids], edges[cell_edge_gids].T)
            # Careful here! As of NumPy 1.7, np.diag() returns a view.
            rhs = vol * np.diag(A).copy()
            A = A**2

            # Append the the resulting coefficients to the coefficient cache.
            # The system is posdef iff the simplex isn't degenerate.
            try:
                self._edgecoeff_cache[cell_edge_gids] += \
                    linalg.solve(A, rhs, sym_pos=True)
            except np.linalg.linalg.LinAlgError:
                # The matrix A appears to be singular,
                # and the only circumstance that makes this
                # happening is the cell being degenerate.
                # Hence, it has volume 0, and so all the edge
                # coefficients are 0, too.
                # Hence, do nothing.
                pass

        return
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
