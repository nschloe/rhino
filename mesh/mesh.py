# -*- coding: utf-8 -*-
# ==============================================================================
import os
import numpy as np
import vtk
# ==============================================================================
class Cell:
    # --------------------------------------------------------------------------
    def __init__( self,
                  node_indices,
                  edge_indices = None ):
        self.node_indices = node_indices
        self.edge_indices = edge_indices
    # --------------------------------------------------------------------------
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
        self.vtk_mesh = None
    # --------------------------------------------------------------------------
    def write( self,
               filename,
               extra_arrays = None
             ):
        '''Writes data together with the mesh to a file.
        '''

        if self.vtk_mesh is None:
            self.vtk_mesh = self._generate_vtk_mesh( self.nodes, self.cells )

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
    def _generate_vtk_mesh(self, points, elems, X = None, name = None):
        mesh = vtk.vtkUnstructuredGrid()

        # set points
        vtk_points = vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint( point )
        mesh.SetPoints( vtk_points )

        # set cells
        for elem in elems:
            pts = vtk.vtkIdList()
            num_local_nodes = len(elem.node_indices)
            pts.SetNumberOfIds( num_local_nodes )
            # get the connectivity for this element
            k = 0
            for node_index in elem.node_indices:
                pts.InsertId( k, node_index )
                k += 1
            if num_local_nodes == 3:
                element_type = vtk.VTK_TRIANGLE
            elif num_local_nodes == 4:
                element_type = vtk.VTK_TETRA
            else:
                raise ValueError('Unknown element.')
            mesh.InsertNextCell( element_type, pts )

        # set values
        if X is not None:
            mesh.GetPointData().AddArray(self._create_vtkdoublearray(X, name))

        return mesh
    # --------------------------------------------------------------------------
    def _create_vtkdoublearray(self, X, name ):

        scalars0 = vtk.vtkDoubleArray()
        scalars0.SetName ( name )

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
        if self.edges is not None:
            return

        self.edges = []

        import itertools
        # The (sorted) dictionary node_edges keeps track of how nodes and edges
        # are connected.
        # If  node_edges[(3,4)] == 17  is true, then the nodes (3,4) are
        # connected  by edge 17.
        node_edges = {}

        # Loop over all elements.
        for cell_index, cell in enumerate( self.cells ):
            cell.edge_indices = []
            # We're treating simplices so loop over all combinations of
            # local nodes.
            for indices in itertools.combinations(cell.node_indices, 2):
                # Combinations are emitted in lexicographic sort order.
                if indices in node_edges.keys():
                    # edge already assigned
                    cell.edge_indices.append(node_edges[indices])
                else:
                    # add edge
                    new_edge_index = len(self.edges)
                    self.edges.append(indices)
                    node_edges[indices] = new_edge_index
                    cell.edge_indices.append(new_edge_index)

        return
    # --------------------------------------------------------------------------
    def refine( self ):
        '''Canonically refine a mesh by inserting nodes at all edge midpoints and
        make four triangular elements where there was one.'''
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
                                     list(new_edge_opposite_of_local_node)
                                     )
                               )
            # The three corner elements:
            for k in xrange(3):
                new_elements.append( Cell( [element.node_indices[k],
                                           local_neighbor_midpoints[k][0],
                                           local_neighbor_midpoints[k][1]],
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
