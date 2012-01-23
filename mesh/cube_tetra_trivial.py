#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Creates a simplisitc mesh on a cube.
'''
import mesh
import numpy as np
from math import sin, pi, atan, sqrt
import time
# ==============================================================================
def _main():

    # get the file name to be written to
    file_name = _parse_options()

    # dimensions of the cube
    l = [ 10.0, 10.0, 10.0 ]

    # Mesh parameters
    # Number of nodes along the length of the strip
    N = [ 30, 30, 30 ]

    # Generate suitable ranges for parametrization
    x_range = np.linspace( -0.5*l[0], 0.5*l[0], N[0] )
    y_range = np.linspace( -0.5*l[1], 0.5*l[1], N[1] )
    z_range = np.linspace( -0.5*l[2], 0.5*l[2], N[2] )

    print 'Create nodes...',
    start = time.time()
    # Create the vertices.
    num_nodes = N[0] * N[1] * N[2]
    nodes = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                nodes.append( mesh.Node( [ x, y, z ] ) )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print 'Create elements...',
    start = time.time()
    # Create the elements (cells).
    # There is 1 way to split a cube into 5 tetrahedra,
    # and 12 ways to split it into 6 tetrahedra.
    # See <http://private.mcnet.ch/baumann/Splitting%20a%20cube%20in%20tetrahedras2.htm>.
    # Also interesting: <http://en.wikipedia.org/wiki/Marching_tetrahedrons>.
    num_cubes = (N[0]-1) * (N[1]-1) * (N[2]-1)
    num_cells = 5 * num_cubes
    #print type( vtk.VTK_TETRA )
    #dt = np.dtype( [ (int, 4), (int) ] )
    elems = []
    for i in range(N[0] - 1):
        for j in range(N[1] - 1):
            for k in range(N[2] - 1):
                # Switch the element styles to make sure the edges match at
                # the faces of the cubes.
                if ( i+j+k ) % 2 == 0:
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*i     + j   ) + k,
                                               N[2] * ( N[1]*i     + j+1 ) + k,
                                               N[2] * ( N[1]*(i+1) + j   ) + k,
                                               N[2] * ( N[1]*i     + j   ) + k+1
                                             ]
                                           )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*i     + j+1 ) + k,
                                               N[2] * ( N[1]*(i+1) + j+1 ) + k,
                                               N[2] * ( N[1]*(i+1) + j   ) + k,
                                               N[2] * ( N[1]*(i+1) + j+1 ) + k+1
                                             ]
                                           )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*i     + j+1 ) + k,
                                               N[2] * ( N[1]*(i+1) + j   ) + k,
                                               N[2] * ( N[1]*i     + j   ) + k+1,
                                               N[2] * ( N[1]*(i+1) + j+1 ) + k+1
                                             ]
                                           )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*i     + j+1 ) + k,
                                               N[2] * ( N[1]*i     + j   ) + k+1,
                                               N[2] * ( N[1]*i     + j+1 ) + k+1,
                                               N[2] * ( N[1]*(i+1) + j+1 ) + k+1
                                             ],
                                             vtk.VTK_TETRA
                                           )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*(i+1) + j   ) + k,
                                               N[2] * ( N[1]*i     + j   ) + k+1,
                                               N[2] * ( N[1]*(i+1) + j+1 ) + k+1,
                                               N[2] * ( N[1]*(i+1) + j   ) + k+1
                                             ]
                                           )
                                )
                else:
                    # Like the previous one, but flipped along the first
                    # coordinate: i+1 -> i, i -> i+1.
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*(i+1) + j   ) + k,
                                               N[2] * ( N[1]*(i+1) + j+1 ) + k,
                                               N[2] * ( N[1]*i     + j   ) + k,
                                               N[2] * ( N[1]*(i+1) + j   ) + k+1
                                            ]
                                          )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*(i+1) + j+1 ) + k,
                                               N[2] * ( N[1]*i     + j+1 ) + k,
                                               N[2] * ( N[1]*i     + j   ) + k,
                                               N[2] * ( N[1]*i     + j+1 ) + k+1
                                             ]
                                           )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*(i+1) + j+1 ) + k,
                                               N[2] * ( N[1]*i     + j   ) + k,
                                               N[2] * ( N[1]*(i+1) + j   ) + k+1,
                                               N[2] * ( N[1]*i     + j+1 ) + k+1
                                             ]
                                           )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*(i+1) + j+1 ) + k,
                                               N[2] * ( N[1]*(i+1) + j   ) + k+1,
                                               N[2] * ( N[1]*(i+1) + j+1 ) + k+1,
                                               N[2] * ( N[1]*i     + j+1 ) + k+1
                                             ]
                                           )
                                )
                    elems.append( mesh.Cell( [ N[2] * ( N[1]*i     + j   ) + k,
                                               N[2] * ( N[1]*(i+1) + j   ) + k+1,
                                               N[2] * ( N[1]*i     + j+1 ) + k+1,
                                               N[2] * ( N[1]*i     + j   ) + k+1
                                             ]
                                           )
                                )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print 'Create values...',
    start = time.time()
    # create values
    X = np.empty( num_nodes, dtype = complex )
    k = 0
    for x in x_range:
        for y in y_range:
            for z in z_range:
                X[k] = complex( 1.0, 0.0 )
                #X[k] = complex( sin( x/lx * pi ), sin( y/ly * pi ) )
                k += 1
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print 'Create thickness values...',
    start = time.time()
    # add thickness value
    thickness = np.empty( num_nodes, dtype = float )
    alpha = 1.0
    beta = 2.0
    k = 0
    for x in x_range:
        for y in y_range:
            for z in z_range:
                #thickness[k] = alpha + (beta-alpha) * (y/(0.5*ly))**2
                thickness[k] = 1.0
                k += 1
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print 'Create mvp...',
    start = time.time()
    # add magnetic vector potential
    import magnetic_vector_potentials
    A = np.empty( (num_nodes,3), dtype = float )
    k = 0
    # exact corner of a cube
    phi = pi/4.0 # azimuth
    theta = atan( 1.0/sqrt(2.0) ) # altitude
    for x in x_range:
        for y in y_range:
            for z in z_range:
                #A[k,:] = magnetic_vector_potentials.mvp_z( nodes[k].coords )
                A[k,:] = magnetic_vector_potentials.mvp_spherical( nodes[k].coords, phi, theta )
                k += 1
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # create the mesh data structure
    print 'Create mesh data structure...',
    start = time.time()
    mymesh = mesh.Mesh( nodes, elems )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # write the mesh with data
    print 'Write to file...',
    start = time.time()
    mymesh.write(file_name, {'psi': X, 'A': A, 'thickness': thickness})
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct a trival tetrahedrization of a cube.' )


    parser.add_argument( 'filename',
                         metavar = 'FILE',
                         type    = str,
                         help    = 'file to be written to'
                       )

    args = parser.parse_args()

    return args.filename
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
