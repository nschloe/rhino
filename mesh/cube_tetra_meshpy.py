#!/usr/bin/env python
# ==============================================================================
def _main():
    import meshpy_interface
    import numpy as np
    import mesh

    args = _parse_options()

    lx = 10.0
    ly = 10.0
    lz = 10.0
    max_volume = 2.0e-2

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

    mymesh.write(args.filename, {'psi': X})

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct tetrahedrization of a cube.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
