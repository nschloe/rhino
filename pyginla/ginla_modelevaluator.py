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
def get_triangle_area( edge0, edge1 ):
    '''Returns the area of triangle spanned by the two given edges.'''
    return 0.5 * np.linalg.norm( np.cross( edge0, edge1 ) )
# ==============================================================================
def get_tetrahedron_volume( edge0, edge1, edge2 ):
    '''Returns the volume of a tetrahedron spanned by the given edges.
    The edges must not be conplanar.'''
    alpha = np.vdot( edge0, np.cross(edge1, edge2) )
    norm_prod = np.linalg.norm(edge0) \
              * np.linalg.norm(edge1) \
              * np.linalg.norm(edge2)
    if abs(alpha) / norm_prod < 1.0e-5:
        raise ValueError( 'The edges seem to be conplanar.' )

    return abs( alpha ) / 6.0
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

        res = - self._keo * psi \
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
            absPsi0Squared = psi0.real**2 + psi0.imag**2
            return - self._keo * phi \
                + ( 1.0-self._T - 2.0*absPsi0Squared ) * phi \
                - psi0**2 * phi.conj()
        # ----------------------------------------------------------------------
        assert( psi0 is not None )
        num_unknowns = len(self.mesh.nodes)
        if self._keo is None:
            self._assemble_keo()
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
        assert( psi0 is not None )
        num_unknowns = len(psi0)
        if self._keo is None:
            self._assemble_keo()
        absPsi0Squared = psi0.real**2 + psi0.imag**2
        A = -self._keo + spdiags(1.0 - self._T - 2.0*absPsi0Squared.T,
                                 [0], num_unknowns, num_unknowns
                                 )
        B = spdiags(-psi0**2, [0], num_unknowns, num_unknowns)
        return A, B
    # ==========================================================================
    def get_preconditioner(self, psi0):
        '''Return the preconditioner.
        '''
        if self._keo is None:
            self._assemble_keo()
        absPsi0Squared = psi0.real**2 + psi0.imag**2
        return self._keo + spdiags(2.0*absPsi0Squared.T,
                                   [0], len(psi0), len(psi0)
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
            x0 = np.zeros((num_unknowns, 1), dtype=complex)
            out = nm.cg(prec, phi, x0,
                        tol = 1.0e-16,
                        #maxiter = 10,
                        M = amg_prec,
                        #explicit_residual = False,
                        inner_product = self.inner_product
                        )
            if out['info'] != 0:
                print 'Preconditioner did not converge; last residual: %g' \
                      % out['relresvec'][-1]
            return out['x']
        # ----------------------------------------------------------------------
        prec = self.get_preconditioner(psi0)
        prec_amg_solver = \
            pyamg.smoothed_aggregation_solver( prec,
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
        #return amg_prec

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
        return np.dot( scaledPhi0.T.conj(), phi1).real
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
        if self.control_volumes is None:
            self._compute_control_volumes()
        D = spdiags(1.0/self.control_volumes.T, [0], num_nodes, num_nodes)
        self._keo = D * self._keo

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
                vol = get_triangle_area( local_edge[0],
                                         local_edge[1] )
            elif num_local_edges == 6:
                try:
                    vol = get_tetrahedron_volume( local_edge[0],
                                                  local_edge[1],
                                                  local_edge[2] )
                except ValueError:
                    # If computing the volume throws an exception, then the
                    # edges chosen happened to be conplanar. Changing one of
                    # those fixes this.
                    vol = get_tetrahedron_volume( local_edge[0],
                                                  local_edge[1],
                                                  local_edge[3] )
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
            A   = np.empty( (num_local_edges, num_local_edges), dtype = float )
            rhs = np.empty( num_local_edges, dtype = float )
            for i in xrange( num_local_edges ):
                rhs[i] = vol * np.vdot( local_edge[i], \
                                        local_edge[i] )
                for j in xrange( num_local_edges ):
                    A[i, j] = np.vdot( local_edge[i],  \
                                       local_edge[j] ) \
                            * np.vdot( local_edge[j],  \
                                       local_edge[i] )

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
        # ----------------------------------------------------------------------
        def _triangle_circumcenter( x ):
            '''Compute the circumcenter of a triangle.
            '''
            a = x[0] - x[1]
            b = x[1] - x[2]
            c = x[2] - x[0]
            w = np.cross(a, b)
            omega = 2.0 * np.dot(w, w)

            if abs(omega) < 1.0e-10:
                raise ZeroDivisionError( 'The nodes don''t seem to form '
                                       + 'a proper triangle.' )

            alpha = -np.dot(b, b) * np.dot(a, c) / omega
            beta  = -np.dot(c, c) * np.dot(b, a) / omega
            gamma = -np.dot(a, a) * np.dot(c, b) / omega

            m = alpha * x[0] + beta * x[1] + gamma * x[2]

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
        # ----------------------------------------------------------------------
        def _tetrahedron_circumcenter( x ):
            '''Computes the center of the circumsphere of a tetrahedron.
            '''
            # http://www.cgafaq.info/wiki/Tetrahedron_Circumsphere
            b = x[1] - x[0]
            c = x[2] - x[0]
            d = x[3] - x[0]

            omega = (2.0 * np.dot( b, np.cross(c, d)))

            if abs(omega) < 1.0e-10:
                raise ZeroDivisionError( 'Tetrahedron is degenerate.' )
            return x[0] + (   np.dot(b, b) * np.cross(c, d)
                            + np.dot(c, c) * np.cross(d, b)
                            + np.dot(d, d) * np.cross(b, c)
                          ) / omega
        # ----------------------------------------------------------------------
        def _compute_covolume_2d( x0, x1, circumcenter, other0 ):
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
        def _compute_covolume_3d( x0, x1, cc, other0, other1 ):
            covolume = 0.0
            edge_midpoint = 0.5 * ( x0 + x1 )

            # Compute the circumcenters of the adjacent faces.
            ccFace0 = _triangle_circumcenter( [x0, x1, other0] )
            ccFace1 = _triangle_circumcenter( [x0, x1, other1] )

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
        def _get_other_indices( e0, e1 ):
            '''Given to indices between 0 and 3, return the other two out of
            [0, 1, 2, 3].'''
            other_indices = []
            for k in xrange(4):
                if k != e0 and k != e1:
                    other_indices.append( k )

            return other_indices
        # ----------------------------------------------------------------------

        num_nodes = len( self.mesh.nodes )
        self.control_volumes = np.zeros( num_nodes, dtype = float )
        for cellNodes in self.mesh.cellsNodes:

            local_node = []
            for node_index in cellNodes:
                local_node.append( self.mesh.nodes[node_index] )

            # Compute the circumcenter of the cell.
            num_local_nodes = len( cellNodes )
            if num_local_nodes == 3:
                cell_dim = 2
                cc = _triangle_circumcenter( local_node )
            elif num_local_nodes == 4:
                cell_dim = 3
                cc = _tetrahedron_circumcenter( local_node )
            else:
                raise ValueError( 'Control volumes can only be constructed ' \
                                  'for triangles and tetrahedra.' )

            # Iterate over pairs of nodes aka local edges.
            for e0 in xrange( num_local_nodes ):
                index0 = cellNodes[e0]
                for e1 in xrange( e0+1, num_local_nodes ):
                    index1 = cellNodes[e1]
                    edge_length = np.linalg.norm( local_node[e0]
                                                - local_node[e1] )

                    other_indices = _get_other_indices( e0, e1 )
                    if cell_dim == 2:
                        covolume = _compute_covolume_2d( local_node[e0],
                                                         local_node[e1],
                                                         cc,
                                                         local_node[other_indices[0]]
                                                       )
                    elif cell_dim == 3:
                        covolume = _compute_covolume_3d( local_node[e0],
                                                         local_node[e1],
                                                         cc,
                                                         local_node[other_indices[0]],
                                                         local_node[other_indices[1]]
                                                       )
                    else:
                        raise ValueError( 'Control volumes can only be constructed ' \
                                          'for triangles and tetrahedra.' )

                    pyramid_volume = 0.5*edge_length * covolume / cell_dim

                    # control volume contributions
                    self.control_volumes[ index0 ] += pyramid_volume
                    self.control_volumes[ index1 ] += pyramid_volume

        self.control_volumes = np.reshape( self.control_volumes,
                                           (len(self.control_volumes),1))
        return
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
