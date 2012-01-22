#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a zig-zag mesh on a rectangle in the x-y-plane.
'''
import sys
import vtk
import mesh_io
import mesh
import numpy as np
from cmath import sin, pi
import time
# ==============================================================================
def _main():

    # get the file name to be written to
    args = _parse_options()

    # dimensions of the rectangle
    cc_radius = 5.0 # circumcircle radius
    lx = np.sqrt(2.0) * cc_radius
    ly = lx

    # Mesh parameters
    # Number of nodes along the length of the strip
    nx = args.nx
    ny = nx

    # Generate suitable ranges for parametrization
    x_range = np.linspace( -0.5*lx, 0.5*lx, nx )
    y_range = np.linspace( -0.5*ly, 0.5*ly, ny )

    # Create the vertices.
    print 'Create nodes...',
    start = time.time()
    num_nodes = nx * ny
    nodes = []
    for x in x_range:
        for y in y_range:
            nodes.append( np.array([x, y, 0]) )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # create the elements (cells)
    print 'Create elements...',
    start = time.time()
    num_elems = 2 * (nx-1) * (ny-1)
    elems = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if (i+j)%2==0:
                elems.append( mesh.Cell( [ i*ny + j, (i + 1)*ny + j + 1,  i     *ny + j + 1 ],
                                         vtk.VTK_TRIANGLE
                                       )
                            )
                elems.append( mesh.Cell( [ i*ny + j, (i + 1)*ny + j    , (i + 1)*ny + j + 1 ],
                                         vtk.VTK_TRIANGLE
                                       )
                            )
            else:
                elems.append( mesh.Cell( [ i    *ny + j, (i+1)*ny + j  , i*ny + j+1 ],
                                         vtk.VTK_TRIANGLE
                                       )
                            )
                elems.append( mesh.Cell( [ (i+1)*ny + j, (i+1)*ny + j+1, i*ny + j+1 ],
                                         vtk.VTK_TRIANGLE
                                       )
                            )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # create values
    print 'Create values...',
    start = time.time()
    import random, cmath
    X = np.empty( num_nodes, dtype = complex )
    for k, node in enumerate(nodes):
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
    for k, node in enumerate(nodes):
        #A[k,:] = magnetic_vector_potentials.mvp_z( node )
        A[k,:] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # create the mesh data structure
    print 'Convert to mesh data structure...',
    start = time.time()
    mymesh = mesh.Mesh( nodes, elems )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # write the mesh
    print 'Write mesh...',
    start = time.time()
    mesh_io.write( args.filename, mymesh, {'psi':X, 'A':A} )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print '\n%d nodes, %d elements' % (num_nodes, len(elems))

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct zigzag triangulation of a rectangle.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    parser.add_argument( '--num-x-points', '-x',
                         metavar='NUM_X_POINTS',
                         dest='nx',
                         nargs='?',
                         type=int,
                         const=10,
                         default=10,
                         help='number of points in x-direction (default: 10)'
                       )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
