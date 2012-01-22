#!/usr/bin/env python
# ==============================================================================
import argparse
import meshpy_interface
import numpy as np
import mesh, mesh_io
from meshpy.tet import MeshInfo, build
from meshpy.geometry import generate_surface_of_revolution, EXT_OPEN, GeometryBuilder
# ==============================================================================
def _main():

    args = _parse_options()

    radius = 3.0

    points = 10
    dphi = np.pi / points

    def truncate(r):
        if abs(r) < 1e-10:
            return 0
        else:
            return r

    # Build outline for surface of revolution.
    rz = [(truncate(radius * np.sin(i*dphi)), radius * np.cos(i*dphi))
          for i in range(points+1)
         ]

    geob = GeometryBuilder()
    geob.add_geometry( *generate_surface_of_revolution(rz,
                                                       closure=EXT_OPEN,
                                                       radial_subdiv=10)
                     )

    mesh_info = MeshInfo()
    geob.set(mesh_info)
    mesh = build(mesh_info)

    print mesh.points
    print mesh.elements
    #mesh.write_vtk( args.filename )

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct tetrahedrization of a ball.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--refinements', '-r',
                         metavar='NUM_REFINEMENTS',
                         dest='ref_steps',
                         nargs='?',
                         type=int,
                         const=0,
                         default=0,
                         help='number of mesh refinement steps to be performed (default: 0)'
                       )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
