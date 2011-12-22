#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Provide tools for solving the Ginzburg--Landau equations.
'''
import mesh_io
import numpy as np
import scipy
from scipy import sparse, linalg
import math, cmath
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
        raise 'The following edges seem to be conplanar:'

    return abs( alpha ) / 6.0
# ==============================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class GinlaModelEvaluator:
    '''
    Ginzburg--Landau model evaluator class.
    '''
    # ==========================================================================
    def __init__( self, mesh, A, mu ):
        '''Initialization. Set mesh.
        '''
        self.mesh = mesh
        self._raw_magnetic_vector_potential = A
        self.mu = mu
        self._temperature = 0.0
        self._keo = None
        self.control_volumes = None
        self._edge_midpoints = None
        self._edgecoeff_cache = None
        self._mvp_edge_cache = None
        self._psi = None
        return
    # ==========================================================================
    def compute_f( self, psi ):
        '''Computes the Ginzburg--Landau residual for a state psi.
        '''
        if self._keo is None:
            self._assemble_keo()

        if self.control_volumes is None:
            self._compute_control_volumes()

        res = - self._keo * psi \
              + self.control_volumes * psi * \
                ( 1.0-self._temperature - abs( psi )**2 )

        return res
    # ==========================================================================
    def set_current_psi( self, current_psi ):
        '''Sets the current psi.
        '''
        self._psi = current_psi
        return
    # ==========================================================================
    def compute_jacobian( self, phi ):
        '''Defines the matrix vector multiplication scheme for the
        Jacobian operator as in

            A phi + B phi*

        with

            A = K - I * ( 1 - 2*|psi|^2 ),
            B = diag( psi^2 ).
        '''
        if self._keo is None:
            self._assemble_keo()

        if self.control_volumes is None:
            self._compute_control_volumes()

        assert( self._psi is not None )

        return - self._keo * phi \
            + self.control_volumes * \
                  ( 1.0-self._temperature - 2.0*abs(self._psi)**2 ) * phi \
            - self.control_volumes * self._psi**2 * phi.conjugate()
    # ==========================================================================
    def norm( self, psi ):
        '''Compute the discretized L2-norm.
        '''
        alpha = np.vdot( self.control_volumes * psi, psi )

        assert( abs( alpha.imag ) <= 1.0e-10 * abs( alpha ) )

        return math.sqrt( alpha.real )
    # ==========================================================================
    def energy( self, psi ):
        '''Compute the Gibbs free energy.
        Not really a norm, but a good measure for our purposes here.
        '''
        alpha = - np.vdot( self.control_volumes * psi**2, psi**2 )
        assert( abs( alpha.imag ) < 1.0e-10 )

        return alpha.real / self.control_volumes.sum()
    # ==========================================================================
    def set_parameter( self, mu ):
        '''Update the parameter.
        '''
        self.mu = mu
        self._keo = None
        return
    # ==========================================================================
    #def set_mesh( self, mesh ):
        #'''Update the mesh.
        #'''
        #self.mesh = mesh

        #self._keo = None
        #self.control_volumes = None
        #self._edge_lengths = None
        #self._coedge_edge_ratios = None

        #return
    # ==========================================================================
    def write( self, psi, filename ):
        '''Writes the mesh to a file.'''
        mesh_io.write_mesh( filename, self.mesh, psi )
        return
    # ==========================================================================
    def _assemble_keo( self ):
        '''Take a pick for the KEO assembler.
        '''
        return self._assemble_keo1()
    # ==========================================================================
    def _assemble_keo1( self ):
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

        # Loop over the all local edges of all cells.
        for ( cell_index, cell ) in enumerate( self.mesh.cells ):
            local_edge_index = 0
            num_local_nodes = len( cell.node_indices )
            for e0 in xrange( num_local_nodes ):
                index0 = cell.node_indices[e0]
                for e1 in xrange( e0+1, num_local_nodes ):
                    index1 = cell.node_indices[e1]

                    # Fetch the cached values.
                    a_integral = \
                        self._mvp_edge_cache[cell_index][local_edge_index]
                    alpha = self._edgecoeff_cache[cell_index][local_edge_index]

                    # Sum them into the matrix.
                    self._keo[ index0, index0 ] += alpha
                    self._keo[ index0, index1 ] -= alpha \
                                                 * cmath.exp( -1j * a_integral )
                    self._keo[ index1, index0 ] -= alpha \
                                                 * cmath.exp(  1j * a_integral )
                    self._keo[ index1, index1 ] += alpha

                    local_edge_index += 1

        # transform the matrix into the more efficient CSR format
        self._keo = self._keo.tocsr()

        return
    # ==========================================================================
    #def _assemble_keo2( self ):
        #'''
        #Create FVM equation system for Poisson's problem with Dirichlet boundary
        #conditions.
        #'''
        ## count the number of edges
        #num_edges = 0
        #for element in self.mesh.elements:
            #for edge in element.edges:
                #num_edges += 1

        #row  = np.empty( 4*num_edges, dtype = int )
        #col  = np.empty( 4*num_edges, dtype = int )
        #data = np.empty( 4*num_edges, dtype = complex )

        ## compute the FVM entities for the mesh
        #if self._edge_lengths is None or self._coedge_edge_ratios is None:
            #self._create_fvm_entities()

        #k = 0
        #ii = 0
        #for element in self.mesh.elements:
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
                #node0 = self.mesh.nodes[ edge[0] ].coords
                #node1 = self.mesh.nodes[ edge[1] ].coords
                #midpoint = 0.5 * ( node0 + node1 )

                ## Instead of projecting onto the normalized edge and then
                ## multiplying with the edge length for the approximation of the
                ## integral, just project on the not normalized edge.
                #a_integral = np.dot( node1 - node0,
                                     #self._magnetic_vector_potential( midpoint )
                                   #)

                ## sum it into the matrix
                #row[ ii ]  = edge[0]
                #col[ ii ]  = edge[0]
                #data[ ii ] = self._coedge_edge_ratios[k][l]
                #ii += 1

                #row[ ii ]  = edge[0]
                #col[ ii ]  = edge[1]
                #data[ ii ] = - self._coedge_edge_ratios[k][l] \
                            #* cmath.exp( -1j * a_integral )
                #ii += 1

                #row[ ii ]  = edge[1]
                #col[ ii ]  = edge[0]
                #data[ ii ] = - self._coedge_edge_ratios[k][l] \
                            #* cmath.exp(  1j * a_integral )
                #ii += 1

                #row[ ii ]  = edge[1]
                #col[ ii ]  = edge[1]
                #data[ ii ] = self._coedge_edge_ratios[k][l]
                #ii += 1

                #l += 1
            #k += 1

        #num_nodes = len( self.mesh.nodes )
        ## transform the matrix into the more efficient CSR format
        #self._keo = sparse.coo_matrix( (data,(row,col)),
                                       #shape=(num_nodes,num_nodes)
                                     #).tocsr()

        #return
    ## ==========================================================================
    #def _assemble_keo3( self ):
        #'''
        #Create FVM equation system for Poisson's problem with Dirichlet boundary
        #conditions.
        #'''
        #num_nodes = len( self.mesh.nodes )

        #if self._keo is None:
            #self._keo = sparse.lil_matrix( ( num_nodes, num_nodes ),
                                           #dtype = complex
                                         #)
        #else:
            #self._keo.data[:] = 0.0

        ## compute the FVM entities for the mesh
        #if self._edge_lengths is None or self._coedge_edge_ratios is None:
            #self._create_fvm_entities()

        #k = 0
        #for element in self.mesh.elements:
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
                #node0 = self.mesh.nodes[ edge[0] ].coords
                #node1 = self.mesh.nodes[ edge[1] ].coords
                #midpoint = 0.5 * ( node0 + node1 )

                ## Instead of projecting onto the normalized edge and then
                ## multiplying with the edge length for the approximation of the
                ## integral, just project on the not normalized edge.
                #a_integral = np.dot( node1 - node0,
                                     #self._magnetic_vector_potential( midpoint )
                                   #)

                ## sum it into the matrix
                #if self._coedge_edge_ratios[k][l] != 0.0:
                    #self._keo[ edge[0], edge[0] ] += self._coedge_edge_ratios[k][l]
                    #self._keo[ edge[0], edge[1] ] -= self._coedge_edge_ratios[k][l] \
                                                  #* cmath.exp( -1j * a_integral )
                    #self._keo[ edge[1], edge[0] ] -= self._coedge_edge_ratios[k][l] \
                                                  #* cmath.exp(  1j * a_integral )
                    #self._keo[ edge[1], edge[1] ] += self._coedge_edge_ratios[k][l]
                #l += 1
            #k += 1

        ## transform the matrix into the more efficient CSR format
        #self._keo = self._keo.tocsr()

        #return
    # ==========================================================================
    def _build_edgecoeff_cache( self ):
        '''Build cache for the edge coefficients.
        (in 2D: coedge-edge ratios).'''

        self._edgecoeff_cache = []

        # Calculate the edge contributions cell by cell.
        for cell in self.mesh.cells:
            # Get the edge coordinates.
            num_local_nodes = len( cell.node_indices )
            # We only deal with simplices.
            num_local_edges = num_local_nodes*(num_local_nodes-1) / 2
            local_edge_coords = []
            for e0 in xrange( num_local_nodes ):
                node0 = self.mesh.nodes[cell.node_indices[e0]].coords
                for e1 in xrange( e0+1, num_local_nodes ):
                    node1 = self.mesh.nodes[cell.node_indices[e1]].coords
                    local_edge_coords.append( node1 - node0 )

            # Compute the volume of the simplex.
            if num_local_edges == 3:
                vol = get_triangle_area( local_edge_coords[0],
                                         local_edge_coords[1] )
            elif num_local_edges == 6:
                try:
                    vol = get_tetrahedron_volume( local_edge_coords[0],
                                                  local_edge_coords[1],
                                                  local_edge_coords[2] )
                except:
                    # If computing the volume throws an exception, then the
                    # edges chosen happened to be conplanar. Changing one of
                    # those fixes this.
                    vol = get_tetrahedron_volume( local_edge_coords[0],
                                                  local_edge_coords[1],
                                                  local_edge_coords[3] )
            else:
                raise 'Unknown geometry with %d edges.' % num_local_edges

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
                rhs[i] = vol * np.vdot( local_edge_coords[i], \
                                        local_edge_coords[i] )
                for j in xrange( num_local_edges ):
                    A[i, j] = np.vdot( local_edge_coords[i],  \
                                       local_edge_coords[j] ) \
                            * np.vdot( local_edge_coords[j],  \
                                       local_edge_coords[i] )

            # Append the the resulting coefficients to the coefficient cache.
            # The system is posdef iff the simplex isn't degenerate.
            self._edgecoeff_cache.append( linalg.solve( A, rhs,
                                                        sym_pos = True
                                                      )
                                        )
        return
    # ==========================================================================
    def _build_mvp_edge_cache( self ):
        '''Builds the cache for the magnetic vector potential.'''

        self._mvp_edge_cache = []

        # Loop over the all local edges of all cells.
        for ( cell_index, cell ) in enumerate( self.mesh.cells ):
            self._mvp_edge_cache.append( [] )
            num_local_nodes = len( cell.node_indices )
            for e0 in xrange( num_local_nodes ):
                index0 = cell.node_indices[e0]
                node0 = self.mesh.nodes[index0].coords
                for e1 in xrange( e0+1, num_local_nodes ):
                    index1 = cell.node_indices[e1]
                    node1 = self.mesh.nodes[index1].coords
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
                    mvp = 0.5 * (self._get_mvp(index0) + self._get_mvp(index1))
                    self._mvp_edge_cache[cell_index].append(
                                                   np.vdot( node1 - node0, mvp )
                                                           )
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
                #node0 = self.mesh.nodes[ edge[0] ].coords
                #node1 = self.mesh.nodes[ edge[1] ].coords
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
        def _triangle_circumcenter( x0, x1, x2 ):
            '''
            Compute the circumcenter of a triangle.
            '''
            w = np.cross( x0-x1, x1-x2 )
            omega = 2.0 * np.dot( w, w )

            if abs(omega) < 1.0e-10:
                raise ZeroDivisionError( "Division by 0." )

            alpha = np.dot( x1-x2, x1-x2 ) * np.dot( x0-x1, x0-x2 ) / omega
            beta  = np.dot( x2-x0, x2-x0 ) * np.dot( x1-x2, x1-x0 ) / omega
            gamma = np.dot( x0-x1, x0-x1 ) * np.dot( x2-x0, x2-x1 ) / omega

            return alpha * x0 + beta * x1 + gamma * x2
        # ----------------------------------------------------------------------

        num_nodes = len( self.mesh.nodes )
        self.control_volumes = np.empty( num_nodes, dtype = float )

        for cell in self.mesh.cells:
            #if cell.cell_type != vtk.VTK_TRIANGLE:
                #print "Control volumes can only be constructed consistently " \
                      #"with triangular elements."
                #raise NameError

            # compute the circumcenter of the element
            x0 = self.mesh.nodes[ cell.node_indices[0] ].coords
            x1 = self.mesh.nodes[ cell.node_indices[1] ].coords
            x2 = self.mesh.nodes[ cell.node_indices[2] ].coords
            cc = _triangle_circumcenter( x0, x1, x2 )

            num_local_nodes = len( cell.node_indices )

            # Iterate over pairs of nodes aka local edges.
            for e0 in xrange( num_local_nodes ):
                index0 = cell.node_indices[e0]
                x0 = self.mesh.nodes[ index0 ].coords
                for e1 in xrange( e0+1, num_local_nodes ):
                    index1 = cell.node_indices[e1]
                    x1 = self.mesh.nodes[ index1 ].coords
                    midpoint = 0.5 * ( x0 + x1 )
                    # control volume contributions
                    self.control_volumes[ index0 ] += \
                        get_triangle_area( cc-x0, midpoint-x0 )
                    self.control_volumes[ index1 ] += \
                        get_triangle_area( cc-x1, midpoint-x1 )

        return
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
