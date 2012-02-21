#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Provide tools for solving the Ginzburg--Landau equations.
'''
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import spdiags
import math, cmath
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# ==============================================================================
def get_triangle_area(node0, node1, node2):
    '''Returns the area of triangle spanned by the two given edges.'''
    #edge0 = node0 - node1
    #edge1 = node1 - node2
    #return 0.5 * np.linalg.norm( np.cross( edge0, edge1 ) )
    import vtk
    return abs(vtk.vtkTriangle.TriangleArea(node0, node1, node2))
# ==============================================================================
def get_tetrahedron_volume(node0, node1, node2, node3):
    '''Returns the volume of a tetrahedron given by the nodes.
    '''
    #edge0 = node0 - node1
    #edge1 = node1 - node2
    #edge2 = node2 - node3
    #edge3 = node3 - node0

    #alpha = np.vdot( edge0, np.cross(edge1, edge2) )
    #norm_prod = np.linalg.norm(edge0) \
              #* np.linalg.norm(edge1) \
              #* np.linalg.norm(edge2)
    #if abs(alpha) / norm_prod < 1.0e-5:
        ## Edges probably conplanar. Take a different set.
        #alpha = np.vdot( edge0, np.cross(edge1, edge3) )
        #norm_prod = np.linalg.norm(edge0) \
                  #* np.linalg.norm(edge1) \
                  #* np.linalg.norm(edge3)

    #return abs( alpha ) / 6.0
    import vtk
    return abs(vtk.vtkTetra.ComputeVolume(node0, node1, node2, node3))
# ==============================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class GinlaModelEvaluator:
    '''Ginzburg--Landau model evaluator class.
    '''
    # ==========================================================================
    def __init__( self, mesh, A, mu ):
        '''Initialization. Set mesh.
        '''
        self.dtype = complex
        self.mesh = mesh
        self._raw_magnetic_vector_potential = A
        self.mu = mu
        self._T = 0.0
        self._keo = None
        self._D = None
        self.control_volumes = None
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
        if self.control_volumes is None:
            self._compute_control_volumes()

        res = (- self._keo * psi) / self.control_volumes  \
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
            return (- self._keo * phi) / self.control_volumes \
                + alpha * phi \
                - psi0Squared * phi.conj()
        # ----------------------------------------------------------------------
        assert( psi0 is not None )
        num_unknowns = len(self.mesh.nodes)

        if self._keo is None:
            self._assemble_keo()
        if self.control_volumes is None:
            self._compute_control_volumes()

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
            return (self._keo * x) / self.control_volumes \
                + 2.0 * absPsi0Squared * x
        # ----------------------------------------------------------------------
        if self._keo is None:
            self._assemble_keo()

        absPsi0Squared = psi0.real**2 + psi0.imag**2
        num_unknowns = len(psi0)
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
            rhs = self.control_volumes * phi

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
            else:
                raise ValueError('Unknown preconditioner type \'%s\'.' % precon_type)

            return x
        # ----------------------------------------------------------------------
        if self._keo is None:
            self._assemble_keo()
        if self.control_volumes is None:
            self._compute_control_volumes()

        num_nodes = len(self.control_volumes)
        absPsi0Squared = psi0.real**2 + psi0.imag**2
        D = spdiags(2 * absPsi0Squared.T * self.control_volumes.T, [0],
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
        if self.control_volumes is None:
            self._compute_control_volumes()

        if len(phi0.shape)==1:
            scaledPhi0 = self.control_volumes[:,0] * phi0
        elif len(phi0.shape)==2:
            scaledPhi0 = self.control_volumes * phi0

        # np.vdot only works for vectors, so use np.dot(....T.conj()) here.
        return np.dot(scaledPhi0.T.conj(), phi1).real
    # ==========================================================================
    def energy( self, psi ):
        '''Compute the Gibbs free energy.
        Not really a norm, but a good measure for our purposes here.
        '''
        if self.control_volumes is None:
            self._compute_control_volumes()

        alpha = - np.vdot( psi**2, self.control_volumes * psi**2 )
        assert( abs( alpha.imag ) < 1.0e-10 )

        return alpha.real / self.control_volumes.sum()
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

        # loop over all edges
        self.mesh.create_adjacent_entities()
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
        self.mesh.create_adjacent_entities()

        num_edges = len(self.mesh.edgesNodes)
        self._edgecoeff_cache = np.zeros(num_edges, dtype=float)

        # Calculate the edge contributions cell by cell.
        for cellNodes, cellEdges in zip(self.mesh.cellsNodes,self.mesh.cellsEdges):
            # We only deal with simplices.
            num_local_edges = len(cellEdges)
            # Build local edge coordinates.
            local_edges = np.empty(num_local_edges, dtype=np.dtype((float,3)))
            for k, edgeNodes in enumerate(self.mesh.edgesNodes[cellEdges]):
                local_edges[k] = self.mesh.nodes[edgeNodes[1]] \
                               - self.mesh.nodes[edgeNodes[0]]

            # Compute the volume of the simplex.
            if num_local_edges == 3:
                vol = get_triangle_area(self.mesh.nodes[cellNodes[0]],
                                        self.mesh.nodes[cellNodes[1]],
                                        self.mesh.nodes[cellNodes[2]])
            elif num_local_edges == 6:
                vol = get_tetrahedron_volume(self.mesh.nodes[cellNodes[0]],
                                             self.mesh.nodes[cellNodes[1]],
                                             self.mesh.nodes[cellNodes[2]],
                                             self.mesh.nodes[cellNodes[3]])
            else:
                raise RuntimeError( 'Unknown geometry with %d edges.'
                                    % num_local_edges )

            # Build the equation system:
            # The equation
            #
            # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>
            #
            # has to hold for all vectors u in the plane spanned by the edges,
            # particularly by the edges themselves.
            A   = np.zeros( (num_local_edges, num_local_edges), dtype = float )
            rhs = np.empty( num_local_edges, dtype = float )
            for i in xrange( num_local_edges ):
                rhs[i] = vol * np.dot(local_edges[i], local_edges[i])
                # Fill the upper triangle of the symmetric matrix A.
                for j in xrange(i, num_local_edges):
                    alpha = np.dot(local_edges[i], local_edges[j])
                    A[i, j] = alpha**2

            # Append the the resulting coefficients to the coefficient cache.
            # The system is posdef iff the simplex isn't degenerate.
            coeffs = linalg.solve( A, rhs, sym_pos=True )
            for k, coeff in enumerate(coeffs):
                self._edgecoeff_cache[cellEdges[k]] += coeff

        return
    # ==========================================================================
    def _build_mvp_edge_cache( self ):
        '''Builds the cache for the magnetic vector potential.'''

        # make sure the mesh has edges
        self.mesh.create_adjacent_entities()

        num_edges = len(self.mesh.edgesNodes)
        self._mvp_edge_cache = np.zeros(num_edges, dtype=float)

        # Loop over the all local edges of all cells.
        for k, node_indices in enumerate(self.mesh.edgesNodes):
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
            self._mvp_edge_cache[k] = np.dot(edge, mvp)

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
    def _compute_control_volumes( self ):
        '''Computes the area of the control volumes.
        '''
        num_local_nodes = len(self.mesh.cellsNodes[0])
        if num_local_nodes == 3:
            self._compute_control_volumes_2d()
        elif num_local_nodes == 4:
            #self._compute_control_volumes_3d()
            #cv = self.control_volumes.copy()
            #ec = self.edge_contribs
            #print
            self._compute_control_volumes_3d_old()
            #print ec - self.edge_contribs
            #self._show_covolume(54)
            #print cv[2], self.control_volumes[2]
            #print cv[2] - self.control_volumes[2]
            #for edge_id, node_ids in enumerate(self.mesh.edgesNodes):
                #if 2 in node_ids:
                    #print edge_id
        else:
            raise ValueError('Control volumes can only be constructed ' +
                             'for triangles and tetrahedra.')

        return
    # ==========================================================================
    def _compute_control_volumes_2d( self ):
        num_nodes = len(self.mesh.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float)

        # compute cell circumcenters
        num_cells = len(self.mesh.cellsNodes)
        circumcenters = np.empty(num_cells, dtype=np.dtype((float,3)))
        for k, cellNodes in enumerate(self.mesh.cellsNodes):
            circumcenters[k] = self._triangle_circumcenter(self.mesh.nodes[cellNodes])

        if self.mesh.edgesNodes is None:
            self.mesh.create_adjacent_entities()

        # Precompute edge lengths.
        num_edges = len(self.mesh.edgesNodes)
        edge_lengths = np.empty(num_edges, dtype=float)
        for k in xrange(num_edges):
            nodes = self.mesh.nodes[self.mesh.edgesNodes[k]]
            edge_lengths[k] = np.linalg.norm(nodes[1] - nodes[0])

        # Precompute edge ("face") normals. Do that in such a way that the
        # face normals points in the direction of the cell with the higher
        # cell ID.
        normals = np.zeros(num_edges, dtype=np.dtype((float,3)))
        for cell_id, cellEdges in enumerate(self.mesh.cellsEdges):
            # Loop over the local faces.
            for k in xrange(3):
                edge_id = cellEdges[k]
                # Compute the normal in the direction of the higher cell ID,
                # or if this is a boundary face, to the outside of the domain.
                neighbor_cell_ids = self.mesh.edgesCells[edge_id]
                if cell_id == neighbor_cell_ids[0]:
                    edge_nodes = self.mesh.nodes[self.mesh.edgesNodes[edge_id]]
                    # The current cell is the one with the lower ID.
                    # Get "other" node (aka the one which is not in the current
                    # "face").
                    other_node_id = self.mesh.cellsNodes[cell_id][k]
                    # Get any direction other_node -> face.
                    # As reference, any point in face can be taken, e.g.,
                    # the first face corner point
                    # self.mesh.edgesNodes[edge_id][0].
                    normals[edge_id] = edge_nodes[0] \
                                     - self.mesh.nodes[other_node_id]
                    # Make it orthogonal to the face.
                    edge_dir = (edge_nodes[1] - edge_nodes[0]) / edge_lengths[edge_id]
                    normals[edge_id] -= np.dot(normals[edge_id], edge_dir) * edge_dir
                    # Normalization.
                    normals[edge_id] /= np.linalg.norm(normals[edge_id])

        # Compute covolumes and control volumes.
        for k in xrange(num_edges):
            # Get the circumcenters of the adjacent cells.
            cc = circumcenters[self.mesh.edgesCells[k]]
            node_ids = self.mesh.edgesNodes[k]
            if len(cc) == 2: # interior cell
                # TODO check out if this holds true for bent surfaces too
                coedge = cc[1] - cc[0]
            elif len(cc) == 1: # boundary cell
                node_coords = self.mesh.nodes[node_ids]
                edge_midpoint = 0.5 * (node_coords[0] + node_coords[1])
                coedge = edge_midpoint - cc[0]
            else:
                raise RuntimeError('A face should have either 1 or two adjacent cells.')

            # Project the coedge onto the outer normal. The two vectors should
            # be parallel, it's just the sign of the coedge length that is to
            # be determined here.
            covolume = np.dot(coedge, normals[k])
            pyramid_volume = 0.5 * edge_lengths[k] * covolume / 2
            self.control_volumes[node_ids] += pyramid_volume

        return
    # ==========================================================================
    def _triangle_circumcenter(self, x):
        '''Compute the circumcenter of a triangle.
        '''
        import vtk
        # Project triangle to 2D.
        v = np.empty(3, dtype=np.dtype((float,2)))
        vtk.vtkTriangle.ProjectTo2D(x[0], x[1], x[2], v[0], v[1], v[2])
        # Get the circumcenter in 2D.
        cc_2d = np.empty(2,dtype=float)
        vtk.vtkTriangle.Circumcircle(v[0], v[1], v[2], cc_2d)
        # Project back to 3D by using barycentric coordinates.
        bcoords = np.empty(3,dtype=float)
        vtk.vtkTriangle.BarycentricCoords(cc_2d, v[0], v[1], v[2], bcoords)
        m = bcoords[0] * x[0] + bcoords[1] * x[1] + bcoords[2] * x[2]

        #a = x[0] - x[1]
        #b = x[1] - x[2]
        #c = x[2] - x[0]
        #w = np.cross(a, b)
        #omega = 2.0 * np.dot(w, w)
        #if abs(omega) < 1.0e-10:
            #raise ZeroDivisionError( 'The nodes don''t seem to form '
                                    #+ 'a proper triangle.' )
        #alpha = -np.dot(b, b) * np.dot(a, c) / omega
        #beta  = -np.dot(c, c) * np.dot(b, a) / omega
        #gamma = -np.dot(a, a) * np.dot(c, b) / omega
        #m = alpha * x[0] + beta * x[1] + gamma * x[2]

        ## Alternative implementation from
        ## https://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        #a = x[1] - x[0]
        #b = x[2] - x[0]
        #alpha = np.dot(a, a)
        #beta = np.dot(b, b)
        #w = np.cross(a, b)
        #omega = 2.0 * np.dot(w, w)
        #m = np.empty(3)
        #m[0] = x[0][0] + ((alpha * b[1] - beta * a[1]) * w[2]
                          #-(alpha * b[2] - beta * a[2]) * w[1]) / omega
        #m[1] = x[0][1] + ((alpha * b[2] - beta * a[2]) * w[0]
                          #-(alpha * b[0] - beta * a[0]) * w[2]) / omega
        #m[2] = x[0][2] + ((alpha * b[0] - beta * a[0]) * w[1]
                          #-(alpha * b[1] - beta * a[1]) * w[0]) / omega

        return m
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
    # ==========================================================================
    def _compute_control_volumes_3d(self):

        if self.mesh.edgesNodes is None:
            self.mesh.create_adjacent_entities()
          
        # get cell circumcenters
        if self.mesh.cell_circumcenters is None:
            self.mesh.create_cell_circumcenters()
        cell_ccs = self.mesh.cell_circumcenters

        # get face circumcenters
        if self.mesh.face_circumcenters is None:
            self.mesh.create_face_circumcenters()
        face_ccs = self.mesh.face_circumcenters

        # Precompute edge lengths.
        num_edges = len(self.mesh.edgesNodes)
        edge_lengths = np.empty(num_edges, dtype=float)
        for edge_id in xrange(num_edges):
            nodes = self.mesh.nodes[self.mesh.edgesNodes[edge_id]]
            edge_lengths[edge_id] = np.linalg.norm(nodes[1] - nodes[0])

        # Precompute face normals. Do that in such a way that the
        # face normals points in the direction of the cell with the higher
        # cell ID.
        num_faces = len(self.mesh.facesNodes)
        normals = np.zeros(num_faces, dtype=np.dtype((float,3)))
        for cell_id, cellFaces in enumerate(self.mesh.cellsFaces):
            # Loop over the local faces.
            for k in xrange(4):
                face_id = cellFaces[k]
                # Compute the normal in the direction of the higher cell ID,
                # or if this is a boundary face, to the outside of the domain.
                neighbor_cell_ids = self.mesh.facesCells[face_id]
                if cell_id == neighbor_cell_ids[0]:
                    # The current cell is the one with the lower ID.
                    face_nodes = self.mesh.nodes[self.mesh.facesNodes[face_id]]
                    # Get "other" node (aka the one which is not in the current
                    # face).
                    other_node_id = self.mesh.cellsNodes[cell_id][k]
                    # Get any direction other_node -> face.
                    # As reference, any point in face can be taken, e.g.,
                    # the face circumcenter.
                    normals[face_id] = face_ccs[face_id] \
                                     - self.mesh.nodes[other_node_id]
                    if face_id == 2:
                        tmp = normals[face_id]
                    # Make it orthogonal to the face by doing Gram-Schmidt
                    # with the two edges of the face.
                    edge_id = self.mesh.facesEdges[face_id][0]
                    nodes = self.mesh.nodes[self.mesh.edgesNodes[edge_id]]
                    # No need to compute the norm of the first edge -- it's
                    # already here!
                    v0 = (nodes[1] - nodes[0]) / edge_lengths[edge_id]
                    edge_id = self.mesh.facesEdges[face_id][1]
                    nodes = self.mesh.nodes[self.mesh.edgesNodes[edge_id]]
                    v1 = nodes[1] - nodes[0]
                    v1 -= np.dot(v1, v0) * v0
                    v1 /= np.linalg.norm(v1)
                    normals[face_id] -= np.dot(normals[face_id], v0) * v0
                    normals[face_id] -= np.dot(normals[face_id], v1) * v1
                    # Normalization.
                    normals[face_id] /= np.linalg.norm(normals[face_id])

        # Compute covolumes and control volumes.
        num_nodes = len(self.mesh.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float)

        self.edge_contribs = np.zeros((num_edges,1), dtype = float)
        for edge_id in xrange(num_edges):
            covolume = 0.0
            edge_node_ids = self.mesh.edgesNodes[edge_id]
            edge_midpoint = 0.5 * ( self.mesh.nodes[edge_node_ids[0]]
                                  + self.mesh.nodes[edge_node_ids[1]])
            for face_id in self.mesh.edgesFaces[edge_id]:
                face_cc = face_ccs[face_id]
                # Get the circumcenters of the adjacent cells.
                cc = cell_ccs[self.mesh.facesCells[face_id]]
                if len(cc) == 2: # interior face
                    coedge = cc[1] - cc[0]
                elif len(cc) == 1: # boundary face
                    coedge = face_cc - cc[0]
                else:
                    raise RuntimeError('A face should have either 1 or 2 adjacent cells.')
                # Project the coedge onto the outer normal. The two vectors
                # should be parallel, it's just the sign of the coedge length
                # that is to be determined here.
                h = np.dot(coedge, normals[face_id])
                alpha = np.linalg.norm(face_cc - edge_midpoint)                    
                covolume += 0.5 * alpha * h

            pyramid_volume = 0.5 * edge_lengths[edge_id] * covolume / 3

            if edge_id == 54:
                self.control_volumes[edge_node_ids] += pyramid_volume

        return
    # ==========================================================================
    def _compute_control_volumes_3d_old(self):
        # ----------------------------------------------------------------------
        def _tetrahedron_circumcenter( x ):
            '''Computes the center of the circumsphere of a tetrahedron.
            '''
            ## http://www.cgafaq.info/wiki/Tetrahedron_Circumsphere
            #b = x[1] - x[0]
            #c = x[2] - x[0]
            #d = x[3] - x[0]

            #omega = (2.0 * np.dot( b, np.cross(c, d)))

            #if abs(omega) < 1.0e-10:
                #raise ZeroDivisionError( 'Tetrahedron is degenerate.' )
            #m = x[0] + (   np.dot(b, b) * np.cross(c, d)
                            #+ np.dot(c, c) * np.cross(d, b)
                            #+ np.dot(d, d) * np.cross(b, c)
                          #) / omega

            import vtk
            m = np.empty(3,float)
            vtk.vtkTetra.Circumsphere(x[0], x[1], x[2], x[3], m)

            return m
        # ----------------------------------------------------------------------
        def _compute_covolume(edge_node_ids, cc, other_node_ids, verbose=False):
            covolume = 0.0
            edge_nodes = self.mesh.nodes[edge_node_ids]
            edge_midpoint = 0.5 * (edge_nodes[0] + edge_nodes[1])

            other_nodes = self.mesh.nodes[other_node_ids]

            # Use the triangle (MP, other_nodes[0], other_nodes[1]] )
            # (in this order) to gauge the orientation of the two triangles that
            # compose the quadrilateral.
            gauge = np.cross(other_nodes[0] - edge_midpoint,
                             other_nodes[1] - edge_midpoint)

            # Compute the area of the quadrilateral.
            # There are some really tricky degenerate cases here, i.e.,
            # combinations of when ccFace{0,1}, cc, sit outside of the
            # tetrahedron.

            # Compute the circumcenters of the adjacent faces.
            ccFace0 = self._triangle_circumcenter([edge_nodes[0], edge_nodes[1], other_nodes[0]])

            # Add the area of the first triangle (MP,ccFace0,cc).
            # This makes use of the right angles.
            triangleArea0 = 0.5 \
                          * np.linalg.norm(edge_midpoint - ccFace0) \
                          * np.linalg.norm(ccFace0 - cc)

            # Check if the orientation of the triangle (MP,ccFace0,cc)
            # coincides with the orientation of the gauge triangle. If yes, add
            # the area, subtract otherwise.
            triangleNormal0 = np.cross(ccFace0 - edge_midpoint,
                                       cc - edge_midpoint)
            # copysign takes the absolute value of the first argument and the
            # sign of the second.
            covolume += math.copysign(triangleArea0,
                                      np.dot(triangleNormal0, gauge))

            ccFace1 = self._triangle_circumcenter([edge_nodes[0], edge_nodes[1], other_nodes[1]])

            # Add the area of the second triangle (MP,cc,ccFace1).
            # This makes use of the right angles.
            triangleArea1 = 0.5 \
                          * np.linalg.norm(edge_midpoint - ccFace1) \
                          * np.linalg.norm(ccFace1 - cc)

            # Check if the orientation of the triangle (MP,cc,ccFace1)
            # coincides with the orientation of the gauge triangle. If yes, add
            # the area, subtract otherwise.
            triangleNormal1 = np.cross(cc - edge_midpoint,
                                       ccFace1 - edge_midpoint)
            # copysign takes the absolute value of the first argument and the
            # sign of the second.
            covolume += math.copysign(triangleArea1,
                                      np.dot(triangleNormal1, gauge))
            return covolume
        # ----------------------------------------------------------------------
        def _without(myset, e):
            other_indices = []
            for k in myset:
                if k not in e:
                    other_indices.append( k )
            return other_indices
        # ----------------------------------------------------------------------
        # Precompute edge lengths.
        if self.mesh.edgesNodes is None:
            self.mesh.create_adjacent_entities()

        num_edges = len(self.mesh.edgesNodes)
        edge_lengths = np.empty(num_edges, dtype=float)
        for edge_id in xrange(num_edges):
            nodes = self.mesh.nodes[self.mesh.edgesNodes[edge_id]]
            edge_lengths[edge_id] = np.linalg.norm(nodes[1] - nodes[0])

        num_nodes = len(self.mesh.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float )

        self.edge_contribs = np.zeros((num_edges,1), dtype = float )
        for cell_id, cellNodes in enumerate(self.mesh.cellsNodes):
            # Compute the circumcenter of the cell.
            cc = _tetrahedron_circumcenter( self.mesh.nodes[cellNodes] )

            # Iterate over pairs of nodes aka local edges.
            for edge_id in self.mesh.cellsEdges[cell_id]:
                indices = self.mesh.edgesNodes[edge_id]

                other_indices = _without(cellNodes, indices)
                covolume = _compute_covolume(indices,
                                             cc,
                                             other_indices,
                                             verbose = 0 in indices and edge_id == 0 and cell_id == 0
                                             )

                pyramid_volume = 0.5 * edge_lengths[edge_id] * covolume / 3
                # control volume contributions
                self.control_volumes[indices] += pyramid_volume

        return
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
