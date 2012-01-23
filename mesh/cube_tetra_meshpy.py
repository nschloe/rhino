#!/usr/bin/env python
# ==============================================================================
def _main():
    import meshpy_interface
    import numpy as np
    import mesh
    import time

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
    print 'Create mesh...',
    start = time.time()
    mymesh = meshpy_interface.create_mesh( max_volume, points, facets )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    num_nodes = len(mymesh.nodes)

    # create values
    print 'Create values...',
    start = time.time()
    import random, cmath
    X = np.empty(num_nodes, dtype = complex)
    for k, node in enumerate(mymesh.nodes):
        #X[k] = cmath.rect( random.random(), 2.0 * pi * random.random() )
        X[k] = complex( 1.0, 0.0 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # Add magnetic vector potential.
    print 'Create mvp...',
    start = time.time()
    A = np.empty( (num_nodes,3), dtype = float )
    import magnetic_vector_potentials
    height0 = 0.1
    height1 = 1.1
    radius = 2.0
    for k, node in enumerate(mymesh.nodes):
        A[k,:] = magnetic_vector_potentials.mvp_z( node )
        #A[k,:] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    mymesh.write(args.filename, {'psi': X, 'A': A})

    print '\n%d nodes, %d elements' % (num_nodes, len(mymesh.cells))

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
