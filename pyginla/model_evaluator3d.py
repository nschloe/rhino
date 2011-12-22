#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Solve the Ginzburg--Landau equation.
'''
import vtk
import vtkio
import numpy as np
from scipy import sparse
import math, cmath
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
class model_evaluator3d:
    '''Ginzburg--Landau model evaluator class.
    '''
    # ==========================================================================
    def __init__( self, mu ):
        '''
        Initialization. Set mesh.
        '''
        self.mesh = None
        self._keo = None
        self.control_volumes = None
        self.mu = mu
        self._psi = None
    # ==========================================================================
    def compute_f( self, psi ):
        '''Computes the Ginzburg--Landau residual for a state psi.
        '''
        if self._keo is None:
            self._assemble_kinetic_energy_operator()

        res = - self._keo * psi \
              + self.control_volumes * psi * ( 1.0 - abs( psi )**2 )

        return res
    # ==========================================================================
    def set_current_psi( self, current_psi ):
        '''Sets the current psi.
        '''
        self._psi = current_psi
        return
    # ==========================================================================
    def compute_jacobian( self, psi ):
        '''Defines the matrix vector multiplication scheme for the
        Jacobian operator as in

            A psi + B psi*

        with

            A = K - I * ( 1 - 2*|psi|^2 ),
            B = diag( psi^2 ).
        '''
        if self._keo is None:
            self._assemble_kinetic_energy_operator()

        assert( self._psi is not None )

        return - self._keo * psi \
               + self.control_volumes * ( 1.0 - 2.0*abs(self._psi)**2 ) * psi \
               - self.control_volumes * self._psi**2 * psi.conjugate()
    # ==========================================================================
    def norm( self, psi ):
        '''Compute the discretized L2-norm.
        '''
        alpha = np.vdot( self.control_volumes * psi, psi )

        assert( abs( alpha.imag ) / abs( alpha ) < 1.0e-10 )

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
    def set_mesh( self, mesh ):
        '''Update the mesh.
        '''
        self.mesh = mesh

        self._keo = None
        self.control_volumes = None

        return
    # ==========================================================================
    def write( self, psi, filename ):
        vtkio.write_mesh( filename, self.mesh, psi )
        return
    # ==========================================================================
    def _assemble_kinetic_energy_operator( self ):
        '''Create FVM equation system for Poisson's problem with Dirichlet
        boundary conditions.
        '''
        return self._assemble_kinetic_energy_operator1()
    # ==========================================================================
    def _assemble_kinetic_energy_operator1( self ):
        num_nodes = len( self.mesh.nodes )

        self._keo = sparse.lil_matrix( ( num_nodes, num_nodes ),
                                       dtype = complex
                                     )

        # compute the FVM entities for the mesh
        if self._coedge_edge_ratios is None:
            self._create_fvm_entities()

        # loop over the edges
        for edge in self.mesh.edges:
            # --------------------------------------------------------------
            # Compute the integral
            #
            #    I = \int_{x0}^{xj} (xj-x0)/|xj-x0| . A(x) dx
            #
            # numerically by the midpoint rule, i.e.,
            #
            #    I ~ (xj-x0) . A( 0.5*(xj+x0) ).
            #
            node0 = self.mesh.nodes[ edge[0] ].coords
            node1 = self.mesh.nodes[ edge[1] ].coords

            # Instead of projecting onto the normalized edge and then
            # multiplying with the edge length for the approximation of the
            # integral, just project on the not normalized edge.
            a_integral = np.dot( node1 - node0,
                                 self._magnetic_vector_potential( edge.midpoint.coords )
                               )

            # sum it into the matrix
            self._keo[ edge[0], edge[0] ] += edge.coarea / edge.length
            self._keo[ edge[0], edge[1] ] -= edge.coarea / edge.length \
                                          * cmath.exp( -1j * a_integral )
            self._keo[ edge[1], edge[0] ] -= edge.coarea / edge.length \
                                          * cmath.exp(  1j * a_integral )
            self._keo[ edge[1], edge[1] ] += edge.coarea / edge.length

        # transform the matrix into the more efficient CSR format
        self._keo = self._keo.tocsr()

        return
    # ==========================================================================
    def keo_smallest_eigenvalue_approximation( self ):
        '''Returns
           <v,Av> / <v,v>
        with v = ones and A = KEO - Laplace.
        This is linear approximation for the smallest magnitude eigenvalue
        of KEO.
        '''
        num_nodes = len( self.mesh.nodes )

        # compute the FVM entities for the mesh
        if self._edge_lengths is None or self._coedge_edge_ratios is None:
            self._create_fvm_entities()

        k = 0
        sum = 0.0
        for element in self.mesh.cells:
            # loop over the edges
            l = 0
            for edge in element.edges:
                # --------------------------------------------------------------
                # Compute the integral
                #
                #    I = \int_{x0}^{xj} (xj-x0).A(x) dx
                #
                # numerically by the midpoint rule, i.e.,
                #
                #    I ~ |xj-x0| * (xj-x0) . A( 0.5*(xj+x0) ).
                #
                node0 = self.mesh.nodes[ edge[0] ].coords
                node1 = self.mesh.nodes[ edge[1] ].coords
                midpoint = 0.5 * ( node0 + node1 )

                # Instead of projecting onto the normalized edge and then
                # multiplying with the edge length for the approximation of the
                # integral, just project on the not normalized edge.
                a_integral = np.dot( node1 - node0,
                                     self._magnetic_vector_potential( midpoint )
                                   )

                # sum it in
                sum += 2.0 * self._coedge_edge_ratios[k][l] * \
                       ( 1.0 - math.cos( a_integral ) )
                l += 1
            k += 1

        return sum / len( self.mesh.nodes )
    # ==========================================================================
    def _create_fvm_entities( self ):
        '''Computes the area of the control volumes
        and the lengths of the edges and co-edges.
        '''
        # ----------------------------------------------------------------------
        def _tetrahedron_circumcenter( x0, x1, x2, x3 ):
            '''Computes the center of the circumsphere of a tetrahedron.
            '''
            # http://www.cgafaq.info/wiki/Tetrahedron_Circumsphere
            b = x1 - x0
            c = x2 - x0
            d = x3 - x0

            omega = ( 2 * np.dot( b, np.cross(c,d)) )

            if abs(omega) < 1.0e-10:
                raise ZeroDivisionError( "Division by 0." )
            return (   np.dot(d,d) * np.cross(b,c)
                     + np.dot(c,c) * np.cross(d,b)
                     + np.dot(b,b) * np.cross(c,d)
                   ) / omega
        # ----------------------------------------------------------------------
        def _hexahedron_volume( x0, x1, x2, x3, x4, x5, x6, x7 ):
            '''Computes the volume of a general hexahedron.
            This expects a very specific ordering of the nodes, namely:
            First four nodes of one side of the hexahedron in order,
            then the other four in the same order.
            '''
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            def _tetrahedron_volume( x0, x1, x2, x3 ):
                '''Computes the volumes of a general tetrahedron.
                '''
                # http://en.wikipedia.org/wiki/Tetrahedron#Volume
                return np.dot( x0-x3, np.cross(x1-x3,x2-x3) ) / 6.0
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Divide the hexahedron logically into 5 tetrahedra,
            # and compute the volume of those.
            # http://private.mcnet.ch/baumann/Splitting%20a%20cube%20in%20tetrahedras2.htm
            return   _tetrahedron_volume( x0, x1, x3, x4 )
                   + _tetrahedron_volume( x1, x2, x3, x6 )
                   + _tetrahedron_volume( x1, x3, x4, x6 )
                   + _tetrahedron_volume( x1, x4, x5, x6 )
                   + _tetrahedron_volume( x3, x4, x6, x7 )
        # ----------------------------------------------------------------------
        def _quadrilateral_area( x0, x1, x2, x3 ):
            '''Computes the area of a general quadrilateral.
            Nodes are assumed to be numbered in order.
            '''
            # http://mathworld.wolfram.com/Quadrilateral.html
            p = np.dot( x3-x1, x3-x1 )
            q = np.dot( x2-x0, x2-x0 )
            a = np.dot( x0, x0 )
            b = np.dot( x1, x1 )
            c = np.dot( x2, x2 )
            d = np.dot( x3, x3 )
            return sqrt( 4*p*q - (b+d-a-c)**2) / 4
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
        self.control_volumes = np.zeros( num_nodes )

        # compute the midpoints of all edges
        for edge in self.mesh.edges:
            n0 = self.mesh.nodes[ edge[0] ].coords
            n1 = self.mesh.nodes[ edge[1] ].coords
            edge.midpoint = 0.5 * ( n0 + n1 )
            edge.length   = np.linalg.norm( n1-n0 )
            edge.coarea   = 0.0

        # compute the circumcenters for each face
        for face in self.mesh.faces:
            face.circumcenter = _triangle_circumcenter( face.nodes[0],
                                                        face.nodes[1],
                                                        face.nodes[2] )

        for cell in self.mesh.cells:
            if cell.cell_type != vtk.VTK_TETRA:
                print "Control volumes can only be constructed consistently " \
                      "with tetrahedral elements."
                raise NameError

            # compute the circumcenter of the cell
            x0 = self.mesh.nodes[ cell.nodes[0] ].coords
            x1 = self.mesh.nodes[ cell.nodes[1] ].coords
            x2 = self.mesh.nodes[ cell.nodes[2] ].coords
            x3 = self.mesh.nodes[ cell.nodes[3] ].coords
            cc = _tetrahedron_circumcenter( x0, x1, x2, x3 )

            midpoints = []
            coedge_edge_ratio = []
            for edge in cell.edges:

                # The area of the quadrilateral perpendicular to the edge,
                # crossing the midpoint of the edge.
                # The boundary of the finite volumes are composed of those
                # quadrilaterals.
                coarea = _quadrilateral_area( edge.midpoint.coords,
                                              face.circumcenter,
                                              cc,
                                              face.circumcenter
                                            )
                edge.coarea += corarea

                # Control volume contributions:
                # For each node, the volume of a pyramid of height
                # '0.5*edgelength' and base area 'coarea' is contributed to the
                # volume of the finite volume.
                pyramid_volume = 0.5 * edge.length * coarea / 3.0
                self.control_volumes[ edge[0] ] += pyramid_volume
                self.control_volumes[ edge[1] ] += pyramid_volume

        return
    # ==========================================================================
# #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
