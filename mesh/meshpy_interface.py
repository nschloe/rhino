'''Interface to MeshPy.
'''
# ==============================================================================
import numpy as np
# ==============================================================================
MAX_AREA = 0.0
# ==============================================================================
def create_mesh( max_area, roundtrip_points, facets = None ):
    '''Create a mesh.
    '''
    import meshpy.triangle

    global MAX_AREA
    MAX_AREA = max_area

    if facets is not None:
        mesh = create_tetrahedron_mesh( roundtrip_points, facets )
    else:
        mesh = create_triangle_mesh( roundtrip_points )

    # Give some info.
    print "Created mesh with %d nodes and %d elements." % \
          (  len( mesh.points ), len( mesh.elements ) )

    return _construct_mymesh( mesh )
# ==============================================================================
def create_triangle_mesh( roundtrip_points ):
    '''Create a mesh.
    '''
    import meshpy.triangle

    # Set the geometry and build the mesh.
    info = meshpy.triangle.MeshInfo()
    info.set_points( roundtrip_points )
    info.set_facets( _round_trip_connect(0, len(roundtrip_points)-1) )

    meshpy_mesh = meshpy.triangle.build( info,
                                         refinement_func = _needs_refinement
                                       )

    return meshpy_mesh
# ==============================================================================
def create_tetrahedron_mesh( roundtrip_points, facets ):
    '''Create a mesh.
    '''
    import meshpy.tet

    # Set the geometry and build the mesh.
    info = meshpy.tet.MeshInfo()
    info.set_points( roundtrip_points )
    info.set_facets( facets )

    meshpy_mesh = meshpy.tet.build( info,
                                    max_volume = MAX_AREA
                                  )

    return meshpy_mesh
# ==============================================================================
def _round_trip_connect(start, end):
    '''Return pairs of subsequent numbers from start to end.
    '''
    result = []
    for i in range(start, end):
      result.append((i, i+1))
    result.append((end, start))
    return result
# ==============================================================================
def _needs_refinement( vertices, area ):
    '''Refinement function.'''
    return area > MAX_AREA
# ==============================================================================
def _construct_mymesh( meshpy_mesh ):
    '''Create the mesh entity.'''
    import mesh
    import vtk

    # Create the vertices.
    nodes = []
    for point in meshpy_mesh.points:
        if len(point) == 2:
            nodes.append(np.append(point, 0.0))
        elif len(point) == 3:
            nodes.append(point)
        else:
            raise ValueError('Unknown point.')
    # Create the elements (cells).
    elems = []
    for element in meshpy_mesh.elements:
        elems.append(mesh.Cell(element))
    # create the mesh data structure
    return mesh.Mesh( nodes, elems )
# ==============================================================================
