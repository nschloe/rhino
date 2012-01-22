# -*- coding: utf-8 -*-
# ==============================================================================
class Mesh:
    # --------------------------------------------------------------------------
    def __init__( self,
                  nodes,
                  cells,
                  edges = None ):
        self.nodes = nodes
        self.edges = edges
        self.cells = cells
    # --------------------------------------------------------------------------
    def refine( self ):
        '''Canonically refine a mesh by inserting nodes at all edge midpoints and
        make four triangular elements where there was one.'''
        import numpy as np
        import vtk
        if self.edges is None:
            raise RuntimeError("Edges must be defined to do refinement.")
        new_nodes = self.nodes
        new_elements = []
        new_edges = []
        is_edge_divided = np.zeros(len(self.edges), dtype=bool)
        edge_midpoint_gids = np.empty(len(self.edges), dtype=int)
        dt = np.dtype((int,2))
        edge_newedges_gids = np.empty(len(self.edges), dtype=dt)
        # Loop over all elements.
        for element in self.cells:
            if element.edge_indices is None:
                raise RuntimeError("Edges must be defined for each element.")
            # Divide edges.
            num_local_edges = len(element.edge_indices)
            local_edge_midpoint_gids = np.empty(num_local_edges, dtype=int)
            local_edge_newedges_gids = np.empty(num_local_edges, dtype=dt)
            local_neighbor_midpoints = [ [], [], [] ]
            local_neighbor_newedges = [ [], [], [] ]
            for k, local_edge_gid in enumerate(element.edge_indices):
                edgenodes_gids = self.edges[local_edge_gid]
                if is_edge_divided[local_edge_gid]:
                    # Just keep records for the element creation.
                    local_edge_midpoint_gids[k] = \
                        edge_midpoint_gids[local_edge_gid]
                    local_edge_newedges_gids[k] = \
                        edge_newedges_gids[local_edge_gid]
                else:
                    # Create new node.
                    new_nodes.append(0.5 *
                 (self.nodes[edgenodes_gids[0]] + self.nodes[edgenodes_gids[1]])
                                    )
                    local_edge_midpoint_gids[k] = len(new_nodes) - 1
                    edge_midpoint_gids[local_edge_gid] = local_edge_midpoint_gids[k]
                    # Divide edge into two.
                    new_edges.append([edgenodes_gids[0],
                                      local_edge_midpoint_gids[k]]
                                    )
                    new_edges.append([local_edge_midpoint_gids[k],
                                      edgenodes_gids[1]]
                                    )
                    local_edge_newedges_gids[k] = \
                        [len(new_edges)-2, len(new_edges)-1]
                    edge_newedges_gids[local_edge_gid] = \
                        local_edge_newedges_gids[k]
                    # Do the household.
                    is_edge_divided[local_edge_gid] = True
                # Keep a record of the new neighbors of the old nodes.
                # Get local node IDs.
                edgenodes_lids = np.empty(2, dtype=int)
                edgenodes_lids[0] = element.node_indices \
                     .index( edgenodes_gids[0] )
                edgenodes_lids[1] = element.node_indices \
                     .index( edgenodes_gids[1] )
                local_neighbor_midpoints[edgenodes_lids[0]] \
                    .append( local_edge_midpoint_gids[k] )
                local_neighbor_midpoints[edgenodes_lids[1]]\
                    .append( local_edge_midpoint_gids[k] )
                local_neighbor_newedges[edgenodes_lids[0]] \
                    .append( local_edge_newedges_gids[k][0] )
                local_neighbor_newedges[edgenodes_lids[1]] \
                    .append( local_edge_newedges_gids[k][1] )

            new_edge_opposite_of_local_node = np.empty(3, dtype=int)
            # New edges: Connect the three midpoints.
            for k in xrange(3):
                new_edges.append( local_neighbor_midpoints[k] )
                new_edge_opposite_of_local_node[k] = len(new_edges) - 1

            # Create new elements.
            # Center element:
            new_elements.append( Cell(list(local_edge_midpoint_gids),
                                     vtk.VTK_TRIANGLE,
                                     list(new_edge_opposite_of_local_node)
                                     )
                               )
            # The three corner elements:
            for k in xrange(3):
                new_elements.append( Cell( [element.node_indices[k],
                                           local_neighbor_midpoints[k][0],
                                           local_neighbor_midpoints[k][1]],
                                           vtk.VTK_TRIANGLE,
                                           [ new_edge_opposite_of_local_node[k],
                                             local_neighbor_newedges[k][0],
                                             local_neighbor_newedges[k][1] ]
                                         )
                                  )

        self.nodes = new_nodes
        self.cells = new_elements
        self.edges = new_edges
        return
    # --------------------------------------------------------------------------
# ==============================================================================
class Cell:
    # --------------------------------------------------------------------------
    def __init__( self,
                  node_indices,
                  cell_type,
                  edge_indices = None ):
        self.node_indices = node_indices
        self.edge_indices = edge_indices
        self.cell_type = cell_type
    # --------------------------------------------------------------------------
# ==============================================================================
