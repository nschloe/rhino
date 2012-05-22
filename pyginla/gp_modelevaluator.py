#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Provide information around the Gross--Pitaevskii equations.
'''
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import spdiags
import cmath
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class GrossPitaevskiiModelEvaluator:
    '''Gross-Pitaevskii model evaluator class.
    Incorporates
       * Ginzburg--Landau: g=1.0, V==-1.0, and some magnetic potential A.
       * Nonlinear Schrödinger: g=1.0, V==0.0, A==0.0.
    '''
    # ==========================================================================
    def __init__(self, mesh, g=0.0, V=None, A=None, mu=0.0 ):
        '''Initialization. Set mesh.
        '''
        self.dtype = complex
        self.mesh = mesh
        n = len(mesh.node_coords)
        self._g = g
        if V is None:
            self._V = np.zeros(n)
        else:
            self._V = V
        if A is None:
            self._raw_magnetic_vector_potential = np.zeros((n,3))
        else:
            self._raw_magnetic_vector_potential = A
        self.mu = mu
        self._keo = None
        self._prec = None
        self._D = None
        self._edgecoeff_cache = None
        self._mvp_edge_cache = None
        self.num_cycles = []
        return
    # ==========================================================================
    def compute_f(self, psi):
        '''Computes the Gross--Pitaevskii residual

            GP(psi) = K*psi + (V + g*|psi|^2) * psi
        '''
        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        res = (self._keo * psi) / self.mesh.control_volumes[:,None]  \
            + (self._V.reshape(psi.shape) + self._g * abs(psi)**2) * psi

        return res
    # ==========================================================================
    def get_jacobian(self, psi0):
        '''Returns a LinearOperator object that defines the matrix-vector
        multiplication scheme for the Jacobian operator as in

            A phi + B phi*

        with

            A = K + I * (V + g * 2*|psi|^2),
            B = g * diag( psi^2 ).
        '''
        # ----------------------------------------------------------------------
        def _apply_jacobian( phi ):
            x = (self._keo * phi) / self.mesh.control_volumes.reshape(phi.shape) \
                + alpha.reshape(phi.shape) * phi \
                + gPsi0Squared.reshape(phi.shape) * phi.conj()
            return x
        # ----------------------------------------------------------------------
        assert psi0 is not None

        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()
        alpha = self._V.reshape(psi0.shape) + self._g * 2.0*(psi0.real**2 + psi0.imag**2)
        gPsi0Squared = self._g * psi0**2

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
            return (self._keo * phi) / self.mesh.control_volumes.reshape(phi.shape) \
                  + alpha.reshape(phi.shape) * phi
        # ----------------------------------------------------------------------
        assert( psi0 is not None )
        num_unknowns = len(self.mesh.node_coords)

        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        if self._g > 0.0:
            alpha = self._g * 2.0 * (psi0.real**2 + psi0.imag**2)
        else:
            alpha = np.zeros((len(psi0),1))

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
        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        num_nodes = len(self.mesh.node_coords)
        if self._g > 0.0:
            alpha = self._g * 2.0 * (psi0.real**2 + psi0.imag**2)
            D = spdiags(alpha.T * self.mesh.control_volumes.T, [0],
                        num_nodes, num_nodes)
            prec = self._keo + D
        else:
            prec = self._keo

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
    def set_parameter(self, mu):
        '''Update the parameter.
        '''
        self.mu = mu
        self._keo = None
        self._mvp_edge_cache = None
        return
    # ==========================================================================
    def _assemble_keo( self ):
        '''Assemble the kinetic energy operator.'''

        # Create the matrix structure.
        num_nodes = len(self.mesh.node_coords)
        self._keo = sparse.lil_matrix((num_nodes, num_nodes),
                                      dtype = complex
                                      )
        # Build caches.
        if self._edgecoeff_cache is None:
            self._build_edgecoeff_cache()
        if self._mvp_edge_cache is None:
            self._build_mvp_edge_cache()
        if self.mesh.edges is None:
            self.mesh.create_adjacent_entities()

        # loop over all edges
        for k, node_indices in enumerate(self.mesh.edges['nodes']):
            # Fetch the cached values.
            alpha = self._edgecoeff_cache[k]
            alphaExp0 = alpha * cmath.exp(1j * self._mvp_edge_cache[k])
            # Sum them into the matrix.
            self._keo[node_indices[0], node_indices[0]] += alpha
            self._keo[node_indices[0], node_indices[1]] -= alphaExp0.conj()
            self._keo[node_indices[1], node_indices[0]] -= alphaExp0
            self._keo[node_indices[1], node_indices[1]] += alpha

        # transform the matrix into the more efficient CSR format
        self._keo = self._keo.tocsr()

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
    def _build_mvp_edge_cache( self ):
        '''Builds the cache for the magnetic vector potential.'''

        # make sure the mesh has edges
        if self.mesh.edges is None:
            self.mesh.create_adjacent_entities()

        # Approximate the integral
        #
        #    I = \int_{x0}^{xj} (xj-x0)/|xj-x0| . A(x) dx
        #
        # numerically by the midpoint rule, i.e.,
        #
        #    I ~ (xj-x0) . A( 0.5*(xj+x0) )
        #      ~ (xj-x0) . 0.5*( A(xj) + A(x0) )
        #
        # The following computes the dot-products of all those
        # edges[i], mvp[i], and put the result in the cache.
        edges = self.mesh.node_coords[self.mesh.edges['nodes'][:,1]] \
              - self.mesh.node_coords[self.mesh.edges['nodes'][:,0]]
        mvp = 0.5 * (self._get_mvp(self.mesh.edges['nodes'][:,1]) \
                    +self._get_mvp(self.mesh.edges['nodes'][:,0]))
        self._mvp_edge_cache = np.sum(edges * mvp, 1)

        return
    # ==========================================================================
    def _get_mvp(self, index):
        return self.mu * self._raw_magnetic_vector_potential[index]
    # ==========================================================================
    #def keo_smallest_eigenvalue_approximation( self ):
        #'''Returns
           #<v,Av> / <v,v>
        #with v = ones and A = KEO - Laplace.
        #This is linear approximation for the smallest magnitude eigenvalue
        #of KEO.
        #'''
        #num_nodes = len( self.mesh.nodes )

        ## compute the FVM entities for the mesh
        #if self._edge_lengths is None or self._coedge_edge_ratios is None:
            #self._create_fvm_entities()

        #k = 0
        #sum = 0.0
        #for element in self.mesh.cells:
            ## loop over the edges
            #l = 0
            #for edge in element.edges:
                ## --------------------------------------------------------------
                ## Compute the integral
                ##
                ##    I = \int_{x0}^{xj} (xj-x0).A(x) dx
                ##
                ## numerically by the midpoint rule, i.e.,
                ##
                ##    I ~ |xj-x0| * (xj-x0) . A( 0.5*(xj+x0) ).
                ##
                #node0 = self.mesh.nodes[ edge[0] ]
                #node1 = self.mesh.nodes[ edge[1] ]
                #midpoint = 0.5 * ( node0 + node1 )

                ## Instead of projecting onto the normalized edge and then
                ## multiplying with the edge length for the approximation of the
                ## integral, just project on the not normalized edge.
                #a_integral = np.dot( node1 - node0,
                                     #self._magnetic_vector_potential( midpoint )
                                   #)

                ## sum it in
                #sum += 2.0 * self._coedge_edge_ratios[k][l] * \
                       #( 1.0 - math.cos( a_integral ) )
                #l += 1
            #k += 1

        #return sum / len( self.mesh.nodes )
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
