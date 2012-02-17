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
import itertools
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

        # Returning just one cycle is not enough here. Possibly related:
        #   Valeria Simoncini and Daniel B. Szyld,
        #   Flexible Inner-Outer Krylov Subspace Methods,
        #   SIAM Journal on Numerical Analysis, vol. 40 (2003), pp. 2219-2239.
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
        self.mesh.create_edges()
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
        self.mesh.create_edges()

        num_edges = len(self.mesh.edgesNodes)
        self._edgecoeff_cache = np.zeros(num_edges, dtype=float)

        # Calculate the edge contributions cell by cell.
        for cellNodes, cellEdges in zip(self.mesh.cellsNodes,self.mesh.cellsEdges):
            # Get the edge coordinates.
            num_local_nodes = len( cellNodes )
            # We only deal with simplices.
            num_local_edges = num_local_nodes*(num_local_nodes-1) / 2
            local_edge = []
            # Loop over all pairs of (local) nodes.
            for index0, index1 in itertools.combinations(cellNodes, 2):
                node0 = self.mesh.nodes[index0]
                node1 = self.mesh.nodes[index1]
                local_edge.append( node1 - node0 )

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
                rhs[i] = vol * np.dot(local_edge[i], local_edge[i])
                # Fill the upper triangle of the symmetric matrix A.
                for j in xrange(i, num_local_edges):
                    alpha = np.dot(local_edge[i], local_edge[j])
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
        self.mesh.create_edges()

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
            self._compute_control_volumes_3d()
        else:
            raise ValueError('Control volumes can only be constructed ' +
                             'for triangles and tetrahedra.')

        return
    # ==========================================================================
    def _compute_control_volumes_2d( self ):
        # ----------------------------------------------------------------------
        def _compute_covolume( x0, x1, circumcenter, other0 ):
            edge_midpoint = 0.5 * ( x0 + x1 )
            coedge_length = np.linalg.norm( edge_midpoint - circumcenter )

            # The only difficulty here is to determine whether the length of
            # coedge is to be taken positive or negative.
            # To this end, make sure that the order (x0, cc, edge_midpoint)
            # is of the same orientation as (x0, other0, edge_midpoint).
            a = other0 - x0
            c = edge_midpoint - x0
            b = cc - x0
            #cell_normal = np.cross(a, c)
            #cc_normal = np.cross(b, c)
            #alpha0 = np.dot( cc_normal, cell_normal )
            alpha = np.dot(b, a) * np.dot(c, c) - np.dot(b, c) * np.dot(a, c)

            # math.copysign() takes the absolute value of the first argument
            # and the sign of the second.
            return math.copysign( coedge_length, alpha )
        # ----------------------------------------------------------------------
        num_nodes = len(self.mesh.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float )
        for cellNodes in self.mesh.cellsNodes:

            num_local_nodes = len(cellNodes)
            local_nodes = np.empty(num_local_nodes, dtype=((float,3)))
            for k, node_index in enumerate(cellNodes):
                local_nodes[k] = self.mesh.nodes[node_index]

            # Compute the circumcenter of the cell.
            cc = self._triangle_circumcenter(local_nodes)
            ## Compute the cell normal.
            #cell_normal = np.cross(local_nodes[1]-local_nodes[0],
                                   #local_nodes[2]-local_nodes[0])
            #cell_normal /= np.linalg.norm(cell_normal)

            # Iterate over pairs of nodes aka local edges.
            for e0 in xrange( num_local_nodes ):
                index0 = cellNodes[e0]
                for e1 in xrange( e0+1, num_local_nodes ):
                    index1 = cellNodes[e1]
                    edge_length = np.linalg.norm( local_nodes[e0]
                                                - local_nodes[e1] )

                    other_indices = self._get_other_indices( e0, e1 )
                    covolume = _compute_covolume(local_nodes[e0],
                                                 local_nodes[e1],
                                                 cc,
                                                 local_nodes[other_indices[0]]
                                                 )

                    pyramid_volume = 0.5 * edge_length * covolume / 2

                    # control volume contributions
                    self.control_volumes[ index0 ] += pyramid_volume
                    self.control_volumes[ index1 ] += pyramid_volume
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
    def _get_other_indices(self, e0, e1):
        '''Given to indices between 0 and 3, return the other two out of
        [0, 1, 2, 3].'''
        other_indices = []
        for k in xrange(4):
            if k != e0 and k != e1:
                other_indices.append( k )

        return other_indices
    # ==========================================================================
    def _compute_control_volumes_3d(self):
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
        def _compute_covolume( x0, x1, cc, other0, other1 ):
            covolume = 0.0
            edge_midpoint = 0.5 * ( x0 + x1 )

            # Compute the circumcenters of the adjacent faces.
            ccFace0 = self._triangle_circumcenter( [x0, x1, other0] )
            ccFace1 = self._triangle_circumcenter( [x0, x1, other1] )

            # Compute the area of the quadrilateral.
            # There are some really tricky degenerate cases here, i.e.,
            # combinations of when ccFace{0,1}, cc, sit outside of the
            # tetrahedron.

            # Use the triangle (MP, localNodes[other[0]], localNodes[other[1]] )
            # (in this order) to gauge the orientation of the two triangles that
            # compose the quadrilateral.
            gauge = np.cross(other0 - edge_midpoint, other1 - edge_midpoint)

            # Add the area of the first triangle (MP,ccFace0,cc).
            # This makes use of the right angles.
            triangleHeight0 = np.linalg.norm(edge_midpoint - ccFace0)
            triangleArea0 = 0.5 \
                          * triangleHeight0 \
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

            # Add the area of the second triangle (MP,cc,ccFace1).
            # This makes use of the right angles.
            triangleHeight1 = np.linalg.norm(edge_midpoint - ccFace1)
            triangleArea1 = 0.5 \
                          * triangleHeight1 \
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
        num_nodes = len(self.mesh.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float )
        for cellNodes in self.mesh.cellsNodes:

            num_local_nodes = len(cellNodes)
            local_nodes = np.empty(num_local_nodes, dtype=((float,3)))
            for k, node_index in enumerate(cellNodes):
                local_nodes[k] = self.mesh.nodes[node_index]

            # Compute the circumcenter of the cell.
            cc = _tetrahedron_circumcenter( local_nodes )

            # Iterate over pairs of nodes aka local edges.
            for e0 in xrange( num_local_nodes ):
                index0 = cellNodes[e0]
                for e1 in xrange( e0+1, num_local_nodes ):
                    index1 = cellNodes[e1]
                    edge_length = np.linalg.norm( local_nodes[e0]
                                                - local_nodes[e1] )

                    other_indices = self._get_other_indices( e0, e1 )
                    covolume = _compute_covolume(local_nodes[e0],
                                                 local_nodes[e1],
                                                 cc,
                                                 local_nodes[other_indices[0]],
                                                 local_nodes[other_indices[1]]
                                                 )

                    pyramid_volume = 0.5*edge_length * covolume / 3

                    # control volume contributions
                    self.control_volumes[ index0 ] += pyramid_volume
                    self.control_volumes[ index1 ] += pyramid_volume
        return
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
