# -*- coding: utf-8 -*-
# ==============================================================================
import numpy as np
from mesh import Mesh
# ==============================================================================
class Mesh2D( Mesh ):
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
        self.edgesCells = edgesCells
        self.cellsNodes = cellsNodes
        self.cellsEdges = cellsEdges

        self.cellsVolume = None
        self.cell_circumcenters = None

        self.vtk_mesh = None
    # --------------------------------------------------------------------------
    def create_cells_volume(self):
        '''Returns the area of triangle spanned by the two given edges.'''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cellsVolume = np.empty(num_cells, dtype=float)
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            #edge0 = node0 - node1
            #edge1 = node1 - node2
            #self.cellsVolume[cell_id] = 0.5 * np.linalg.norm( np.cross( edge0, edge1 ) )
            x = self.nodes[cellNodes]
            self.cellsVolume[cell_id] = \
               abs(vtk.vtkTriangle.TriangleArea(x[0], x[1], x[2]))
        return
    # --------------------------------------------------------------------------
    def create_cell_circumcenters( self ):
        '''Computes the center of the circumsphere of each cell.
        '''
        import vtk
        num_cells = len(self.cellsNodes)
        self.cell_circumcenters = np.empty(num_cells, dtype=np.dtype((float,3)))
        for cell_id, cellNodes in enumerate(self.cellsNodes):
            x = self.nodes[cellNodes]
            # Project triangle to 2D.
            v = np.empty(3, dtype=np.dtype((float,2)))
            vtk.vtkTriangle.ProjectTo2D(x[0], x[1], x[2],
                                        v[0], v[1], v[2])
            # Get the circumcenter in 2D.
            cc_2d = np.empty(2, dtype=float)
            vtk.vtkTriangle.Circumcircle(v[0], v[1], v[2],
                                         cc_2d)
            # Project back to 3D by using barycentric coordinates.
            bcoords = np.empty(3, dtype=float)
            vtk.vtkTriangle.BarycentricCoords(cc_2d, v[0], v[1], v[2], bcoords)
            self.cell_circumcenters[cell_id] = \
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
    def create_adjacent_entities( self ):

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
# ==============================================================================
