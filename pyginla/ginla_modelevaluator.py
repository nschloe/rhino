#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Provide tools for solving the Ginzburg--Landau equations.
'''
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import spdiags
import cmath
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class GinlaModelEvaluator:
    '''Ginzburg--Landau model evaluator class.
    '''
    # ==========================================================================
    def __init__(self, mesh, A, mu):
        '''Initialization. Set mesh.
        '''
        self.dtype = complex
        self.mesh = mesh
        self._raw_magnetic_vector_potential = A
        self.mu = mu
        self._T = 0.0
        self._keo = None
        self._D = None
        self._edgecoeff_cache = None
        self._mvp_edge_cache = None
        self._prec_type = 'amg' #'direct'
        return
    # ==========================================================================
    def compute_f( self, psi ):
        '''Computes the Ginzburg--Landau residual for a state psi.
        '''
        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        res = (- self._keo * psi) / self.mesh.control_volumes  \
              + psi * ( 1.0-self._T - abs( psi )**2 )

        return res
    # ==========================================================================
    def get_jacobian(self, psi0):
        '''Returns a LinearOperator object that defines the matrix-vector
        multiplication scheme for the Jacobian operator as in

            A phi + B phi*

        with

            A = - K + I * ( 1-T - 2*|psi|^2 ),
            B = - diag( psi^2 ).
        '''
        # ----------------------------------------------------------------------
        def _apply_jacobian( phi ):
            return (- self._keo * phi) / self.mesh.control_volumes \
                + alpha * phi \
                - psi0Squared * phi.conj()
        # ----------------------------------------------------------------------
        assert( psi0 is not None )
        num_unknowns = len(self.mesh.nodes)

        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        alpha = ( 1.0-self._T - 2.0*(psi0.real**2 + psi0.imag**2) )
        psi0Squared = psi0**2

        return LinearOperator( (num_unknowns, num_unknowns),
                               _apply_jacobian,
                               dtype = self.dtype
                             )
    # ==========================================================================
    def get_jacobian_blocks(self, psi0):
        '''Returns A and B of the Jacobian operator
            A = - K + I * ( 1-T - 2*|psi|^2 ),
            B = - diag( psi^2 ).
        '''
        raise RuntimeError('Not implemented.')
        #assert( psi0 is not None )
        #num_unknowns = len(psi0)
        #if self._keo is None:
            #self._assemble_keo()
        #absPsi0Squared = psi0.real**2 + psi0.imag**2
        #A = -self._keo + spdiags(1.0 - self._T - 2.0*absPsi0Squared.T,
                                 #[0], num_unknowns, num_unknowns
                                 #)
        #B = spdiags(-psi0**2, [0], num_unknowns, num_unknowns)
        return A, B
    # ==========================================================================
    def get_preconditioner(self, psi0):
        '''Return the preconditioner.
        '''
        # ----------------------------------------------------------------------
        def _apply_precon(x):
            return (self._keo * x) / self.mesh.control_volumes \
                + 2.0 * absPsi0Squared * x
        # ----------------------------------------------------------------------
        assert( psi0 is not None )
        num_unknowns = len(self.mesh.nodes)

        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        absPsi0Squared = psi0.real**2 + psi0.imag**2
        return LinearOperator((num_unknowns, num_unknowns),
                              _apply_precon,
                              dtype = self.dtype
                              )
    # ==========================================================================
    def get_preconditioner_inverse(self, psi0):
        if self._prec_type == 'amg':
            return self._get_preconditioner_inverse_amg(psi0)
        elif self._prec_type == 'direct':
            return self._get_preconditioner_inverse_directsolve(psi0)
        else:
            raise ValueError('Unknown preconditioner type \'%s\'.' %
                             self._prec_type )
    # ==========================================================================
    def _get_preconditioner_inverse_amg(self, psi0):
        '''Use AMG to invert M approximately.
        '''
        import pyamg
        import numerical_methods as nm
        # ----------------------------------------------------------------------
        def _apply_inverse_prec(phi):
            rhs = self.mesh.control_volumes * phi

            precon_type = 'one cycle'
            if precon_type == 'custom cg':
                x0 = np.zeros((num_unknowns, 1), dtype=complex)
                out = nm.cg(prec, rhs, x0,
                            tol = 1.0e-13,
                            M = amg_prec,
                            #explicit_residual = False
                            )
                if out['info'] != 0:
                    print 'Preconditioner did not converge; last residual: %g' \
                          % out['relresvec'][-1]
                return out['xk']
            elif precon_type == 'pyamg solve':
                x0 = np.zeros((num_unknowns, 1), dtype=complex)
                x = np.empty((num_nodes,1), dtype=complex)
                x[:,0] = prec_amg_solver.solve(rhs,
                                               x0 = x0,
                                               maxiter = 1,
                                               tol = 0.0,
                                               accel = None
                                               )
            elif precon_type == 'one cycle':
                x = amg_prec * rhs
                #x = prec_amg_solver.solve(rhs, maxiter=1, cycle='V', tol=1e-15)
            else:
                raise ValueError('Unknown preconditioner type \'%s\'.' % precon_type)

            return x
        # ----------------------------------------------------------------------
        if self._keo is None:
            self._assemble_keo()
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        num_nodes = len(self.mesh.nodes)
        absPsi0Squared = psi0.real**2 + psi0.imag**2
        D = spdiags(2 * absPsi0Squared.T * self.mesh.control_volumes.T, [0],
                    num_nodes, num_nodes)

        prec = self._keo + D

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

        amg_prec = prec_amg_solver.aspreconditioner( cycle='V' )

        num_unknowns = len(psi0)
        return LinearOperator((num_unknowns, num_unknowns),
                              _apply_inverse_prec,
                              dtype = self.dtype
                              )
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
            scaledPhi0 = self.mesh.control_volumes[:,0] * phi0
        elif len(phi0.shape)==2:
            scaledPhi0 = self.mesh.control_volumes * phi0

        # np.vdot only works for vectors, so use np.dot(....T.conj()) here.
        return np.dot(scaledPhi0.T.conj(), phi1).real
    # ==========================================================================
    def energy( self, psi ):
        '''Compute the Gibbs free energy.
        Not really a norm, but a good measure for our purposes here.
        '''
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes()

        alpha = - np.vdot( psi**2, self.mesh.control_volumes * psi**2 )
        assert( abs( alpha.imag ) < 1.0e-10 )

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
        num_nodes = len( self.mesh.nodes )
        self._keo = sparse.lil_matrix( ( num_nodes, num_nodes ),
                                       dtype = complex
                                     )
        # Build caches.
        if self._edgecoeff_cache is None:
            self._build_edgecoeff_cache()
        if self._mvp_edge_cache is None:
            self._build_mvp_edge_cache()
        if self.mesh.edgesNodes is None:
            self.mesh.create_adjacent_entities()

        # loop over all edges
        for k, node_indices in enumerate(self.mesh.edgesNodes):
            # Fetch the cached values.
            alpha = self._edgecoeff_cache[k]
            alphaExp0 = alpha * cmath.exp(1j * self._mvp_edge_cache[k])
            # Sum them into the matrix.
            self._keo[node_indices[0], node_indices[0]] += alpha
            self._keo[node_indices[0], node_indices[1]] -= alphaExp0.conjugate()
            self._keo[node_indices[1], node_indices[0]] -= alphaExp0
            self._keo[node_indices[1], node_indices[1]] += alpha

        # Row-scale.
        # This is *much* faster than individually accessing
        # self.control_volumes in each step of the iteration above.
        # (Bad random access performance of np.array?)
        #if self.control_volumes is None:
            #self._compute_control_volumes()
        #D = spdiags(1.0/self.control_volumes.T, [0], num_nodes, num_nodes)
        #self._keo = D * self._keo

        # transform the matrix into the more efficient CSR format
        self._keo = self._keo.tocsr()

        return
    # ==========================================================================
    def _build_edgecoeff_cache( self ):
        '''Build cache for the edge coefficients.
        (in 2D: coedge-edge ratios).'''

        # make sure the mesh has edges
        if self.mesh.cellsEdges is None:
            self.mesh.create_adjacent_entities()

        num_edges = len(self.mesh.edgesNodes)
        self._edgecoeff_cache = np.zeros(num_edges, dtype=float)

        if self.mesh.cellsVolume is None:
            self.mesh.create_cells_volume()
        vols = self.mesh.cellsVolume

        # Precompute edges.
        edges = np.empty(num_edges, dtype=np.dtype((float,3)))
        for edge_id, edge_nodes in enumerate(self.mesh.edgesNodes):
            edges[edge_id] = self.mesh.nodes[edge_nodes[1]] \
                           - self.mesh.nodes[edge_nodes[0]]

        # Calculate the edge contributions cell by cell.
        num_local_edges = len(self.mesh.cellsEdges[0])
        A   = np.empty( (num_local_edges, num_local_edges), dtype = float )
        rhs = np.empty( num_local_edges, dtype = float )
        for cell_id, cellEdges in enumerate(self.mesh.cellsEdges):
            # Build local edge coordinates.
            local_edges = np.empty(num_local_edges, dtype=np.dtype((float,3)))
            for k, edgeNodes in enumerate(self.mesh.edgesNodes[cellEdges]):
                local_edges[k] = self.mesh.nodes[edgeNodes[1]] \
                               - self.mesh.nodes[edgeNodes[0]]

            # Build the equation system:
            # The equation
            #
            # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>
            #
            # has to hold for all vectors u in the plane spanned by the edges,
            # particularly by the edges themselves.
            for i in xrange( num_local_edges ):
                rhs[i] = vols[cell_id] * np.dot(edges[cellEdges[i]], edges[cellEdges[i]])
                # Fill the upper triangle of the symmetric matrix A.
                for j in xrange(i, num_local_edges):
                    A[i, j] = np.dot(edges[cellEdges[i]], edges[cellEdges[j]])**2

            # Append the the resulting coefficients to the coefficient cache.
            # The system is posdef iff the simplex isn't degenerate.
            coeffs = linalg.solve(A, rhs, sym_pos=True)
            self._edgecoeff_cache[cellEdges] += coeffs

        return
    # ==========================================================================
    def _build_mvp_edge_cache( self ):
        '''Builds the cache for the magnetic vector potential.'''

        # make sure the mesh has edges
        if self.mesh.edgesNodes is None:
            self.mesh.create_adjacent_entities()

        num_edges = len(self.mesh.edgesNodes)
        self._mvp_edge_cache = np.empty(num_edges, dtype=float)

        # Loop over the all local edges of all cells.
        for edge_id, node_indices in enumerate(self.mesh.edgesNodes):
            # ----------------------------------------------------------
            # Approximate the integral
            #
            #    I = \int_{x0}^{xj} (xj-x0)/|xj-x0| . A(x) dx
            #
            # numerically by the midpoint rule, i.e.,
            #
            #    I ~ (xj-x0) . A( 0.5*(xj+x0) )
            #      ~ (xj-x0) . 0.5*( A(xj) + A(x0) )
            #
            edge = self.mesh.nodes[node_indices[1]] \
                 - self.mesh.nodes[node_indices[0]]
            mvp = 0.5 * (self._get_mvp(node_indices[1]) \
                       + self._get_mvp(node_indices[0]))
            self._mvp_edge_cache[edge_id] = np.dot(edge, mvp)

        return
    # ==========================================================================
    def _get_mvp( self, index ):
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
    def _show_covolume(self, edge_id):
        '''For debugging.'''
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import vtk

        # get cell circumcenters
        if self.mesh.cell_circumcenters is None:
            self.mesh.create_cell_circumcenters()
        cell_ccs = self.mesh.cell_circumcenters

        # get face circumcenters
        if self.mesh.face_circumcenters is None:
            self.mesh.create_face_circumcenters()
        face_ccs = self.mesh.face_circumcenters

        edge_nodes = self.mesh.nodes[self.mesh.edgesNodes[edge_id]]
        edge_midpoint = 0.5 * ( edge_nodes[0] + edge_nodes[1] )

        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # plot all adjacent cells
        col = 'k'
        for cell_id in self.mesh.edgesCells[edge_id]:
            for edge in self.mesh.cellsEdges[cell_id]:
                x = self.mesh.nodes[self.mesh.edgesNodes[edge]]
                ax.plot(x[:,0], x[:,1], x[:,2], col)

        # make clear which is the edge
        ax.plot(edge_nodes[:,0], edge_nodes[:,1], edge_nodes[:,2], color=col, linewidth=3.0 )

        # plot covolume        
        for face_id in self.mesh.edgesFaces[edge_id]:
            ccs = cell_ccs[ self.mesh.facesCells[face_id] ]
            v = np.empty(3, dtype=np.dtype((float,2)))
            #col = '0.5'
            col = 'g'
            if len(ccs) == 2:
                ax.plot(ccs[:,0], ccs[:,1], ccs[:,2], color=col)
            elif len(ccs) == 1:
                face_cc = face_ccs[face_id]
                ax.plot([ccs[0][0],face_cc[0]], [ccs[0][1],face_cc[1]], [ccs[0][2],face_cc[2]], color=col)
            else:
                raise RuntimeError('???')
        #ax.plot([edge_midpoint[0]], [edge_midpoint[1]], [edge_midpoint[2]], 'ro')

        # highlight cells
        print self.mesh.edgesCells[edge_id]
        highlight_cells = [3]
        col = 'r'
        for k in highlight_cells:
            cell_id = self.mesh.edgesCells[edge_id][k]
            ax.plot([cell_ccs[cell_id,0]], [cell_ccs[cell_id,1]], [cell_ccs[cell_id,2]],
                    color = col, marker='o')
            for edge in self.mesh.cellsEdges[cell_id]:
                x = self.mesh.nodes[self.mesh.edgesNodes[edge]]
                ax.plot(x[:,0], x[:,1], x[:,2], col, linestyle='dashed')

        plt.show()
        return
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
