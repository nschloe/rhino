# -*- coding: utf-8 -*-
# ==============================================================================
import numpy as np
# ==============================================================================
class Mesh:
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
    def write( self,
               filename,
               extra_arrays = None
             ):
        '''Writes data together with the mesh to a file.
        '''
        import os
        import vtk

        if self.vtk_mesh is None:
            self.vtk_mesh = self._generate_vtk_mesh(self.nodes, self.cellsNodes)

        # add arrays
        if extra_arrays:
            for key, value in extra_arrays.iteritems():
                self.vtk_mesh.GetPointData().AddArray(
                    self._create_vtkdoublearray(value, key))

        extension = os.path.splitext(filename)[1]
        if extension == ".vtu": # VTK XML format
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif extension == ".pvtu": # parallel VTK XML format
            writer = vtk.vtkXMLPUnstructuredGridWriter()
        elif extension == ".vtk": # classical VTK format
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileTypeToASCII()
        elif extension in [ ".ex2", ".exo", ".e" ]: # Exodus II format
            writer = vtk.vtkExodusIIWriter()
            # If the mesh contains vtkModelData information, make use of it
            # and write out all time steps.
            writer.WriteAllTimeStepsOn()
        else:
            raise IOError( "Unknown file type \"%s\"." % filename )

        writer.SetFileName( filename )

        writer.SetInput( self.vtk_mesh )

        writer.Write()

        return
    # --------------------------------------------------------------------------
    def _generate_vtk_mesh(self, points, cellsNodes, X = None, name = None):
        import vtk
        mesh = vtk.vtkUnstructuredGrid()

        # set points
        vtk_points = vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint( point )
        mesh.SetPoints( vtk_points )

        # set cells
        for cellNodes in cellsNodes:
            pts = vtk.vtkIdList()
            num_local_nodes = len(cellNodes)
            pts.SetNumberOfIds(num_local_nodes)
            # get the connectivity for this element
            k = 0
            # TODO insert the whole thing at once?
            for node_index in cellNodes:
                pts.InsertId(k, node_index)
                k += 1
            if num_local_nodes == 3:
                element_type = vtk.VTK_TRIANGLE
            elif num_local_nodes == 4:
                element_type = vtk.VTK_TETRA
            else:
                raise ValueError('Unknown element.')
            mesh.InsertNextCell(element_type, pts)

        # set values
        if X is not None:
            mesh.GetPointData().AddArray(self._create_vtkdoublearray(X, name))

        return mesh
    # --------------------------------------------------------------------------
    def _create_vtkdoublearray(self, X, name ):
        import vtk

        scalars0 = vtk.vtkDoubleArray()
        scalars0.SetName(name)

        if isinstance( X, float ):
            scalars0.SetNumberOfComponents( 1 )
            scalars0.InsertNextValue( X )
        elif (len( X.shape ) == 1 or X.shape[1] == 1) and X.dtype==float:
            # real-valued array
            scalars0.SetNumberOfComponents( 1 )
            for x in X:
                scalars0.InsertNextValue( x )

        elif (len( X.shape ) == 1 or X.shape[1] == 1) and X.dtype==complex:
            # complex-valued array
            scalars0.SetNumberOfComponents( 2 )
            for x in X:
                scalars0.InsertNextValue( x.real )
                scalars0.InsertNextValue( x.imag )

        elif len( X.shape ) == 2 and X.dtype==float: # 2D float field
            m, n = X.shape
            scalars0.SetNumberOfComponents( n )
            for j in range(m):
                for i in range(n):
                    scalars0.InsertNextValue( X[j, i] )

        elif len( X.shape ) == 2 and X.dtype==complex: # 2D complex field
            scalars0.SetNumberOfComponents( 2 )
            m, n = X.shape
            for j in range(n):
                for i in range(m):
                    scalars0.InsertNextValue( X[j, i].real )
                    scalars0.InsertNextValue( X[j, i].imag )

        elif len( X.shape ) == 3: # vector values
            m, n, d = X.shape
            if X.dtype==complex:
                raise "Can't handle complex-valued vector fields."
            if d != 3:
                raise "Can only deal with 3-dimensional vector fields."
            scalars0.SetNumberOfComponents( 3 )
            for j in range( n ):
                for i in range( m ):
                    for k in range( 3 ):
                        scalars0.InsertNextValue( X[i,j,k] )

        else:
            raise ValueError( "Don't know what to do with array." )

        return scalars0
    # --------------------------------------------------------------------------
    def create_adjacent_entities( self ):

        # check if we have a 2d or a 3d mesh
        if len(self.cellsNodes[0]) == 3: # 2d
            self._create_edges_2d()
        elif len(self.cellsNodes[0]) == 4: # 3d
            self._create_edges_faces_3d()
        else:
            raise RuntimeError('Only 2D and 3D supported.')

        #raise 1

        return
    # --------------------------------------------------------------------------
    def _create_edges_2d(self):
        if self.edgesNodes is not None:
            return

        # Take cell #0 as representative.
        num_local_nodes = len(self.cellsNodes[0])
        num_local_edges = num_local_nodes * (num_local_nodes-1) / 2
        num_cells = len(self.cellsNodes)

        # Get upper bound for number of edges; trim later.
        max_num_edges = num_local_nodes * num_cells
        self.edgesNodes = np.empty(max_num_edges, dtype=np.dtype((int,2)))
        self.edgesCells = [[] for k in xrange(max_num_edges)]

        self.cellsEdges = np.empty(num_cells, dtype=np.dtype((int,num_local_edges)))

        # The (sorted) dictionary edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        edges = {}

        new_edge_gid = 0
        # Loop over all elements.
        for cell_id in xrange(num_cells):
            # We're treating simplices so loop over all combinations of
            # local nodes.
            # Make sure cellNodes are sorted.
            self.cellsNodes[cell_id] = np.sort(self.cellsNodes[cell_id])
            for k in xrange(len(self.cellsNodes[cell_id])):
                # Remove the k-th element. This makes sure that the k-th
                # edge is opposite of the k-th node. Useful later in
                # in construction of edge (face) normals.
                # TODO np.delete() is slow; come up with something faster.
                indices = tuple(np.delete(self.cellsNodes[cell_id], k))
                if indices in edges:
                    edge_gid = edges[indices]
                    self.edgesCells[edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = edge_gid
                else:
                    # add edge
                    self.edgesNodes[new_edge_gid] = indices
                    # edgesCells is also always ordered.
                    self.edgesCells[new_edge_gid].append( cell_id )
                    self.cellsEdges[cell_id][k] = new_edge_gid
                    edges[indices] = new_edge_gid
                    new_edge_gid += 1

        # trim edges
        self.edgesNodes = self.edgesNodes[:new_edge_gid]
        self.edgesCells = self.edgesCells[:new_edge_gid]
        return
    # --------------------------------------------------------------------------
    def _create_edges_faces_3d( self ):
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
    def refine( self ):
        '''Canonically refine a mesh by inserting nodes at all edge midpoints
        and make four triangular elements where there was one.'''
        if self.edgesNodes is None:
            raise RuntimeError("Edges must be defined to do refinement.")

        # Record the newly added nodes.
        num_new_nodes = len(self.edgesNodes)
        new_nodes = np.empty(num_new_nodes, dtype=np.dtype((float,3)))
        new_node_gid = len(self.nodes)

        # After the refinement step, all previous edge-node associations will
        # be obsolete, so record *all* the new edges.
        num_new_edges = 2 * len(self.edgesNodes) + 3 * len(self.cellsNodes)
        new_edgesNodes = np.empty(num_new_edges, dtype=np.dtype((int,2)))
        new_edge_gid = 0

        # After the refinement step, all previous cell-node associations will
        # be obsolete, so record *all* the new cells.
        num_new_cells = 4 * len(self.cellsNodes)
        new_cellsNodes = np.empty(num_new_cells, dtype=np.dtype((int,3)))
        new_cellsEdges = np.empty(num_new_cells, dtype=np.dtype((int,3)))
        new_cell_gid = 0

        num_edges = len(self.edgesNodes)
        is_edge_divided = np.zeros(num_edges, dtype=bool)
        edge_midpoint_gids = np.empty(num_edges, dtype=int)
        dt = np.dtype((int,2))
        edge_newedges_gids = np.empty(num_edges, dtype=dt)
        # Loop over all elements.
        if self.cellsEdges is None or len(self.cellsEdges) != len(self.cellsNodes):
            raise RuntimeError("Edges must be defined for each cell.")
        for cellNodes, cellEdges in zip(self.cellsNodes,self.cellsEdges):
            # Divide edges.
            num_local_edges = len(cellEdges)
            local_edge_midpoint_gids = np.empty(num_local_edges, dtype=int)
            local_edge_newedges = np.empty(num_local_edges, dtype=dt)
            local_neighbor_midpoints = [ [], [], [] ]
            local_neighbor_newedges = [ [], [], [] ]
            for k, local_edge_gid in enumerate(cellEdges):
                edgenodes_gids = self.edgesNodes[local_edge_gid]
                if is_edge_divided[local_edge_gid]:
                    # Edge is already divided. Just keep records
                    # for the cell creation.
                    local_edge_midpoint_gids[k] = \
                        edge_midpoint_gids[local_edge_gid]
                    local_edge_newedges[k] = edge_newedges[local_edge_gid]
                else:
                    # Create new node at the edge midpoint.
                    print new_nodes[new_node_gid]
                    new_nodes[new_node_gid] = \
                        0.5 * (self.nodes[edgenodes_gids[0]] \
                              +self.nodes[edgenodes_gids[1]])
                    local_edge_midpoint_gids[k] = new_node_gid
                    new_node_gid += 1
                    edge_midpoint_gids[local_edge_gid] = \
                        local_edge_midpoint_gids[k]

                    # Divide edge into two.
                    new_edgesNodes[new_edge_gid] = \
                        np.array([edgenodes_gids[0], local_edge_midpoint_gids[k]])
                    new_edge_gid += 1
                    new_edgesNodes[new_edge_gid] = \
                        np.array([local_edge_midpoint_gids[k], edgenodes_gids[1]])
                    new_edge_gid += 1

                    local_edge_newedges[k] = \
                        np.array([new_edge_gid-2, new_edge_gid-1])
                    edge_newedges_gids[local_edge_gid] = \
                        local_edge_newedges[k]
                    # Do the household.
                    is_edge_divided[local_edge_gid] = True
                # Keep a record of the new neighbors of the old nodes.
                # Get local node IDs.
                edgenodes_lids = [cellNodes.index(edgenodes_gids[0]),
                                  cellNodes.index(edgenodes_gids[1])]
                local_neighbor_midpoints[edgenodes_lids[0]] \
                    .append( local_edge_midpoint_gids[k] )
                local_neighbor_midpoints[edgenodes_lids[1]]\
                    .append( local_edge_midpoint_gids[k] )
                local_neighbor_newedges[edgenodes_lids[0]] \
                    .append( local_edge_newedges[k][0] )
                local_neighbor_newedges[edgenodes_lids[1]] \
                    .append( local_edge_newedges[k][1] )

            new_edge_opposite_of_local_node = np.empty(3, dtype=int)
            # New edges: Connect the three midpoints.
            for k in xrange(3):
                new_edgesNodes[new_edge_gid] = local_neighbor_midpoints[k]
                new_edge_opposite_of_local_node[k] = new_edge_gid
                new_edge_gid += 1

            # Create new elements.
            # Center cell:
            new_cellsNodes[new_cell_gid] = local_edge_midpoint_gids
            new_cellsEdges[new_cell_gid] = new_edge_opposite_of_local_node
            new_cell_gid += 1
            # The three corner elements:
            for k in xrange(3):
                new_cellsNodes[new_cell_gid] = \
                    np.array([cell.node_indices[k],
                              local_neighbor_midpoints[k][0],
                              local_neighbor_midpoints[k][1]])
                new_cellsEdges[new_cell_gid] = \
                    np.array([new_edge_opposite_of_local_node[k],
                              local_neighbor_newedges[k][0],
                              local_neighbor_newedges[k][1]])
                new_cell_gid += 1

        np.append(self.nodes, new_nodes)
        self.edgesNodes = new_edgesNodes
        self.cellsNodes = new_cellsNodes
        self.cellsEdges = new_cellsEdges
        return

    # --------------------------------------------------------------------------
    def recreate_cells_with_qhull(self):
        import scipy.spatial

        # Create a Delaunay triangulation of the given points.
        delaunay = scipy.spatial.Delaunay(self.nodes)
        # Use the new cells.
        self.cellsNodes = delaunay.vertices

        return
    # --------------------------------------------------------------------------
# ==============================================================================
