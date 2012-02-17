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
        self.cellsEdges = cellsEdges
        self.edgesNodes = edgesNodes
        self.edgesCells = edgesCells
        self.cellsNodes = cellsNodes
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
    def create_edges( self ):
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

        import itertools
        # The (sorted) dictionary node_edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        num_nodes = len(self.nodes)
        node_neighbors = [{} for k in xrange(num_nodes)]

        new_edge_gid = 0
        cell_id = 0
        # Loop over all elements.
        for cellNodes, cellEdges in zip(self.cellsNodes, self.cellsEdges):
            # We're treating simplices so loop over all combinations of
            # local nodes.
            # Make sure cellNodes are sorted.
            cellNodes = np.sort(cellNodes)
            for k, indices in enumerate(itertools.combinations(cellNodes, 2)):
                if indices[1] in node_neighbors[indices[0]].keys():
                    # edge already assigned
                    edge_gid = node_neighbors[indices[0]][indices[1]]
                    self.edgesCells[edge_gid].append( cell_id )
                    cellEdges[k] = edge_gid
                else:
                    # add edge
                    self.edgesNodes[new_edge_gid] = indices
                    self.edgesCells[new_edge_gid].append( cell_id )
                    cellEdges[k] = new_edge_gid
                    node_neighbors[indices[0]][indices[1]] = new_edge_gid
                    new_edge_gid += 1
            cell_id += 1

        # trim edges
        self.edgesNodes = self.edgesNodes[:new_edge_gid]
        self.edgesCells = self.edgesCells[:new_edge_gid]

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
