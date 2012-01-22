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
    max_area = 0.02

    # Round trip corner points of the rectangle.
    points = [ ( -0.5*lx, -0.5*ly ),
               (  0.5*lx, -0.5*ly ),
               (  0.5*lx,  0.5*ly ),
               ( -0.5*lx,  0.5*ly ) ]

    # create the mesh
    mymesh = meshpy_interface.create_mesh( points, max_area )

    # create values
    X = np.empty( len( mymesh.nodes ), dtype = complex )
    for k, x in enumerate( mymesh.nodes ):
        X[k] = complex( 1.0, 0.0 )

    # add parameters
    params = { "mu": 0.0,
               "scaling": 1.0
             }

    mesh_io.write_mesh( file_name,
                        mymesh,
                        [X], ["psi"],
                        params
                      )

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