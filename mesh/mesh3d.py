# -*- coding: utf-8 -*-
# ==============================================================================
import numpy as np
from mesh import Mesh
# ==============================================================================
class Mesh3D( Mesh ):
    # --------------------------------------------------------------------------
    def __init__( self,
                  nodes,
                  cellsNodes,
                  cellsEdges = None,
                  edgesNodes = None,
                  edgesCells = None):
        # It would be sweet if we could handle cells and the rest as arrays
        # with fancy dtypes such as
        #
        #     np.dtype([('nodes', (int, num_local_nodes)),
        #               ('edges', (int, num_local_edges))]),
        #
        # but right now there's no (no easy?) way to extend nodes properly
        # for the case that 'edges' aren't given. A simple recreate-and-copy
        #
        #     for k in xrange(num_cells):
        #         new_cells[k]['nodes'] = self.cells[k]['nodes']
        #
        # does not seem to work for whatever reason.
        # Hence, handle cells and friends of dictionaries of np.arrays.
        if not isinstance(nodes,np.ndarray):
           raise TypeError('For performace reasons, build nodes as np.empty(num_nodes, dtype=np.dtype((float, 3)))')

        if not isinstance(cellsNodes,np.ndarray):
           raise TypeError('For performace reasons, build cellsNodes as np.empty(num_nodes, dtype=np.dtype((int, 3)))')

        if cellsEdges is not None and not isinstance(cellsEdges,np.ndarray):
           raise TypeError('For performace reasons, build cellsEdges as np.empty(num_nodes, dtype=np.dtype((int, 3)))')

        if edgesNodes is not None and  not isinstance(edgesNodes,np.ndarray):
           raise TypeError('For performace reasons, build edgesNodes as np.empty(num_nodes, dtype=np.dtype((int, 2)))')

        self.nodes = nodes
        self.edgesNodes = edgesNodes
        self.edgesFaces = None
        self.edgesCells = edgesCells
        self.facesNodes = None
        self.facesEdges = None
        self.facesCells = None
        self.cellsNodes = cellsNodes
        self.cellsEdges = cellsEdges
        self.cellsFaces = None
        self.cell_circumcenters = None
        self.face_circumcenters = None
        self.cellsVolume = None
        self.control_volumes = None
        self.vtk_mesh = None
    # --------------------------------------------------------------------------
    def create_cells_volume(self):
        '''Returns the volume of a tetrahedron given by the nodes.
        '''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cellsVolume = np.empty(num_cells, dtype=float)
        for cell_id, cellNodes in enumerate(self.cellsNodes):
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

            #self.cellsVolume[cell_id] = abs( alpha ) / 6.0

            x = self.nodes[cellNodes]
            self.cellsVolume[cell_id] = \
                abs(vtk.vtkTetra.ComputeVolume(x[0], x[1], x[2], x[3]))
        return
    # --------------------------------------------------------------------------
    def create_adjacent_entities( self ):
        # Take cell #0 as representative.
        num_local_nodes = len(self.cellsNodes[0])
        num_cells = len(self.cellsNodes)

        # Get upper bound for number of edges; trim later.
        max_num_edges = num_local_nodes * num_cells
        self.edgesNodes = np.empty(max_num_edges, dtype=np.dtype((int,2)))
        self.edgesCells = [[] for k in xrange(max_num_edges)]
        self.cellsEdges = np.empty(num_cells, dtype=np.dtype((int,6)))

        # The (sorted) dictionary node_edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        edges = {}
        new_edge_gid = 0
        # Create edges.
        import itertools
        for cell_id in xrange(num_cells):
            # We're treating simplices so loop over all combinations of
            # local nodes.
            # Make sure cellNodes are sorted.
            self.cellsNodes[cell_id] = np.sort(self.cellsNodes[cell_id])
            for k, indices in enumerate(itertools.combinations(self.cellsNodes[cell_id], 2)):
                if indices in edges:
                    # edge already assigned
                    edge_gid = edges[indices]
                    self.edgesCells[edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = edge_gid
                else:
                    # add edge
                    self.edgesNodes[new_edge_gid] = indices
                    self.edgesCells[new_edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = new_edge_gid
                    edges[indices] = new_edge_gid
                    new_edge_gid += 1

        # trim edges
        self.edgesNodes = self.edgesNodes[:new_edge_gid]
        self.edgesCells = self.edgesCells[:new_edge_gid]

        # Create faces.
        max_num_faces = 4 * num_cells
        self.facesNodes = np.empty(max_num_faces, dtype=np.dtype((int,3)))
        self.facesEdges = np.empty(max_num_faces, dtype=np.dtype((int,3)))
        self.facesCells = [[] for k in xrange(max_num_faces)]
        self.edgesFaces = [[] for k in xrange(new_edge_gid)]
        self.cellsFaces = np.empty(num_cells, dtype=np.dtype((int,4)))
        # Loop over all elements.
        new_face_gid = 0
        faces = {}
        for cell_id in xrange(num_cells):
            # Make sure cellNodes are sorted.
            self.cellsNodes[cell_id] = np.sort(self.cellsNodes[cell_id])
            for k in xrange(len(self.cellsNodes[cell_id])):
                # Remove the k-th element. This makes sure that the k-th
                # face is opposite of the k-th node. Useful later in
                # in construction of face normals.
                # TODO np.delete() is slow; come up with something faster.
                indices = tuple(np.delete(self.cellsNodes[cell_id], k))
                if indices in faces:
                    # Face already assigned, just register it with the
                    # current cell.
                    face_gid = faces[indices]
                    self.facesCells[face_gid].append( cell_id )
                    self.cellsFaces[cell_id][k] = face_gid
                else:
                    # Add face.
                    self.facesNodes[new_face_gid] = indices
                    # Register cells.
                    self.facesCells[new_face_gid].append( cell_id )
                    self.cellsFaces[cell_id][k] = new_face_gid
                    # Register edges.
                    for k, node_tuples in enumerate(itertools.combinations(indices, 2)):
                        # Note that node_tuples is also sorted, and thus
                        # is a key in the edges dictionary.
                        edge_id = edges[node_tuples]
                        self.edgesFaces[edge_id].append( new_face_gid )
                        self.facesEdges[new_face_gid][k] = edge_id
                    # Finalize.
                    faces[indices] = new_face_gid
                    new_face_gid += 1

        # trim faces
        self.facesNodes = self.facesNodes[:new_face_gid]
        self.facesEdges = self.facesEdges[:new_face_gid]
        self.facesCells = self.facesCells[:new_face_gid]

        return
    # --------------------------------------------------------------------------
    def create_cell_circumcenters( self ):
        '''Computes the center of the circumsphere of each cell.
        '''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cell_circumcenters = np.empty(num_cells, dtype=np.dtype((float,3)))
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            ## http://www.cgafaq.info/wiki/Tetrahedron_Circumsphere
            #b = x[1] - x[0]
            #c = x[2] - x[0]
            #d = x[3] - x[0]

            #omega = (2.0 * np.dot( b, np.cross(c, d)))

            #if abs(omega) < 1.0e-10:
                #raise ZeroDivisionError( 'Tetrahedron is degenerate.' )
            #self.cell_circumcenters[cell_id] = x[0] + (   np.dot(b, b) * np.cross(c, d)
                            #+ np.dot(c, c) * np.cross(d, b)
                            #+ np.dot(d, d) * np.cross(b, c)
                          #) / omega
            x = self.nodes[cellNodes]
            vtk.vtkTetra.Circumsphere(x[0], x[1], x[2], x[3],
                                      self.cell_circumcenters[cell_id])
        return
    # --------------------------------------------------------------------------
    def create_face_circumcenters( self ):
        '''Computes the center of the circumcircle of each face.
        '''
        import vtk
        num_faces = len(self.facesNodes)
        self.face_circumcenters = np.empty(num_faces, dtype=np.dtype((float,3)))
        for face_id, faceNodes in enumerate(self.facesNodes):
            x = self.nodes[faceNodes]
            # Project triangle to 2D.
            v = np.empty(3, dtype=np.dtype((float, 2)))
            vtk.vtkTriangle.ProjectTo2D(x[0], x[1], x[2],
                                        v[0], v[1], v[2])
            # Get the circumcenter in 2D.
            cc_2d = np.empty(2, dtype=float)
            vtk.vtkTriangle.Circumcircle(v[0], v[1], v[2], cc_2d)
            # Project back to 3D by using barycentric coordinates.
            bcoords = np.empty(3, dtype=float)
            vtk.vtkTriangle.BarycentricCoords(cc_2d, v[0], v[1], v[2], bcoords)
            self.face_circumcenters[face_id] = \
                bcoords[0] * x[0] + bcoords[1] * x[1] + bcoords[2] * x[2]

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
        return
    # --------------------------------------------------------------------------
    def compute_control_volumes(self):
        #self._compute_control_volumes_new()
        self._compute_control_volumes_old()
        return
    # --------------------------------------------------------------------------
    def _compute_control_volumes_new(self):

        if self.edgesNodes is None:
            self.create_adjacent_entities()

        # get cell circumcenters
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()
        cell_ccs = self.cell_circumcenters

        # get face circumcenters
        if self.face_circumcenters is None:
            self.create_face_circumcenters()
        face_ccs = self.face_circumcenters

        # Precompute edge lengths.
        num_edges = len(self.edgesNodes)
        edge_lengths = np.empty(num_edges, dtype=float)
        for edge_id in xrange(num_edges):
            nodes = self.nodes[self.edgesNodes[edge_id]]
            edge_lengths[edge_id] = np.linalg.norm(nodes[1] - nodes[0])

        # Precompute face normals. Do that in such a way that the
        # face normals points in the direction of the cell with the higher
        # cell ID.
        num_faces = len(self.facesNodes)
        normals = np.zeros(num_faces, dtype=np.dtype((float,3)))
        for cell_id, cellFaces in enumerate(self.cellsFaces):
            # Loop over the local faces.
            for k in xrange(4):
                face_id = cellFaces[k]
                # Compute the normal in the direction of the higher cell ID,
                # or if this is a boundary face, to the outside of the domain.
                neighbor_cell_ids = self.facesCells[face_id]
                if cell_id == neighbor_cell_ids[0]:
                    # The current cell is the one with the lower ID.
                    face_nodes = self.nodes[self.facesNodes[face_id]]
                    # Get "other" node (aka the one which is not in the current
                    # face).
                    other_node_id = self.cellsNodes[cell_id][k]
                    # Get any direction other_node -> face.
                    # As reference, any point in face can be taken, e.g.,
                    # the face circumcenter.
                    normals[face_id] = face_ccs[face_id] \
                                     - self.nodes[other_node_id]
                    if face_id == 2:
                        tmp = normals[face_id]
                    # Make it orthogonal to the face by doing Gram-Schmidt
                    # with the two edges of the face.
                    edge_id = self.facesEdges[face_id][0]
                    nodes = self.nodes[self.edgesNodes[edge_id]]
                    # No need to compute the norm of the first edge -- it's
                    # already here!
                    v0 = (nodes[1] - nodes[0]) / edge_lengths[edge_id]
                    edge_id = self.facesEdges[face_id][1]
                    nodes = self.nodes[self.edgesNodes[edge_id]]
                    v1 = nodes[1] - nodes[0]
                    v1 -= np.dot(v1, v0) * v0
                    v1 /= np.linalg.norm(v1)
                    normals[face_id] -= np.dot(normals[face_id], v0) * v0
                    normals[face_id] -= np.dot(normals[face_id], v1) * v1
                    # Normalization.
                    normals[face_id] /= np.linalg.norm(normals[face_id])

        # Compute covolumes and control volumes.
        num_nodes = len(self.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float)

        self.edge_contribs = np.zeros((num_edges,1), dtype = float)
        for edge_id in xrange(num_edges):
            covolume = 0.0
            edge_node_ids = self.edgesNodes[edge_id]
            edge_midpoint = 0.5 * ( self.nodes[edge_node_ids[0]]
                                  + self.nodes[edge_node_ids[1]])
            for face_id in self.edgesFaces[edge_id]:
                face_cc = face_ccs[face_id]
                # Get the circumcenters of the adjacent cells.
                cc = cell_ccs[self.facesCells[face_id]]
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
    # --------------------------------------------------------------------------
    def _compute_control_volumes_old(self):
        # ----------------------------------------------------------------------
        def _triangle_circumcenter(x):
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

            return m
        # ----------------------------------------------------------------------
        def _compute_covolume(edge_node_ids, cc, other_node_ids, verbose=False):
            import math
            covolume = 0.0
            edge_nodes = self.nodes[edge_node_ids]
            edge_midpoint = 0.5 * (edge_nodes[0] + edge_nodes[1])

            other_nodes = self.nodes[other_node_ids]

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
            ccFace0 = _triangle_circumcenter([edge_nodes[0], edge_nodes[1], other_nodes[0]])

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

            ccFace1 = _triangle_circumcenter([edge_nodes[0], edge_nodes[1], other_nodes[1]])

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
        if self.edgesNodes is None:
            self.create_adjacent_entities()

        # get cell circumcenters
        if self.cell_circumcenters is None:
            self.create_cell_circumcenters()
        cell_ccs = self.cell_circumcenters

        num_edges = len(self.edgesNodes)
        edge_lengths = np.empty(num_edges, dtype=float)
        for edge_id in xrange(num_edges):
            nodes = self.nodes[self.edgesNodes[edge_id]]
            edge_lengths[edge_id] = np.linalg.norm(nodes[1] - nodes[0])

        num_nodes = len(self.nodes)
        self.control_volumes = np.zeros((num_nodes,1), dtype = float )

        self.edge_contribs = np.zeros((num_edges,1), dtype = float )
        # Iterate over cells -> edges.
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            for edge_id in self.cellsEdges[cell_id]:
                indices = self.edgesNodes[edge_id]

                other_indices = _without(cellNodes, indices)
                covolume = _compute_covolume(indices,
                                             cell_ccs[cell_id],
                                             other_indices,
                                             verbose = 0 in indices and edge_id == 0 and cell_id == 0
                                             )

                pyramid_volume = 0.5 * edge_lengths[edge_id] * covolume / 3
                # control volume contributions
                self.control_volumes[indices] += pyramid_volume

        return
    # --------------------------------------------------------------------------
# ==============================================================================
