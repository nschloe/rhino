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
        self.vtk_mesh = None
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
# ==============================================================================
