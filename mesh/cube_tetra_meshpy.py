#!/usr/bin/env python
# ==============================================================================
import optparse
import sys
# ==============================================================================
def _main():
    import meshpy_interface
    import numpy as np
    import mesh, mesh_io

    file_name = _parse_options()

    lx = 10
    ly = 10
    lz = 10
    max_volume = 2.0e-3

    # Round trip corner points of the rectangle.
    points = [ ( -0.5*lx, -0.5*ly, -0.5*lz ),
               (  0.5*lx, -0.5*ly, -0.5*lz ),
               (  0.5*lx,  0.5*ly, -0.5*lz ),
               ( -0.5*lx,  0.5*ly, -0.5*lz ),
               ( -0.5*lx, -0.5*ly,  0.5*lz ),
               (  0.5*lx, -0.5*ly,  0.5*lz ),
               (  0.5*lx,  0.5*ly,  0.5*lz ),
               ( -0.5*lx,  0.5*ly,  0.5*lz ) ]
    facets = [ [0,1,2,3],
               [4,5,6,7],
               [0,4,5,1],
               [1,5,6,2],
               [2,6,7,3],
               [3,7,4,0] ]

    # create the mesh
    mymesh = meshpy_interface.create_mesh( max_volume, points, facets )

    # create values
    X = np.empty( len( mymesh.nodes ), dtype = complex )
    for k, x in enumerate( mymesh.nodes ):
        X[k] = complex( 1.0, 0.0 )

    mymesh.write(file_name, {'psi': X})

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    usage = "usage: %prog outfile"

    parser = optparse.OptionParser( usage = usage )

    (options, args) = parser.parse_args()

    if not args  or  len(args) != 1:
        parser.print_help()
        sys.exit( "\nProvide a file to be written to." )

    return args[0]
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
