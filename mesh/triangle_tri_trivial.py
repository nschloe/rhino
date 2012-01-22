#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a zig-zag mesh on a rectangle in the x-y-plane.
'''
import sys
import vtk
import mesh_io
import mesh
import meshpy_interface
import numpy as np
from cmath import sin, pi
import time
# ==============================================================================
def _main():

    # get the command line arguments
    args = _parse_options()

    #Circumcircle radius of the triangle.
    cc_radius = 5.0

    # Create initial nodes/elements.
    #nodes = [ cc_radius * np.array([np.cos((0.5+0.0/3.0)*pi), np.sin((0.5+0.0/3.0)*pi), 0]),
              #cc_radius * np.array([np.cos((0.5+2.0/3.0)*pi), np.sin((0.5+2.0/3.0)*pi), 0]),
              #cc_radius * np.array([np.cos((0.5+4.0/3.0)*pi), np.sin((0.5+4.0/3.0)*pi), 0])
            #]
    nodes = [ cc_radius * np.array([ 0.0, 1.0, 0.0]),
              cc_radius * np.array([-0.5*np.sqrt(3.0), -0.5, 0.0]),
              cc_radius * np.array([ 0.5*np.sqrt(3.0), -0.5, 0.0])
            ]
    edges = [ [0,1], [1,2], [2,0] ]
    elements = [ mesh.Cell([0, 1, 2], vtk.VTK_TRIANGLE, edge_indices=[0,1,2]) ]


    # Create mesh data structure.
    mymesh = mesh.Mesh( nodes, elements, edges )

    # Refine..
    print 'Mesh refinement...',
    start = time.time()
    for k in xrange(args.ref_steps):
        mymesh.refine()
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    num_nodes = len( mymesh.nodes )

    # create values
    print 'Create values...',
    start = time.time()
    import random, cmath
    X = np.empty( num_nodes, dtype = complex )
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
        #A[k,:] = magnetic_vector_potentials.mvp_z( node )
        A[k,:] = magnetic_vector_potentials.mvp_magnetic_dot( node, radius, height0, height1 )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # write the mesh
    print 'Write mesh...',
    start = time.time()
    mesh_io.write( args.filename, mymesh, {'psi':X, 'A':A} )
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    print '\n%d nodes, %d elements' % (num_nodes, len(mymesh.cells))

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct triangulation of a triangle.' )

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
