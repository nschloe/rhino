#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time

import voropy
# ==============================================================================
def _main():

    # get the command line arguments
    args = _parse_options()

    # read the mesh
    print 'Reading mesh...',
    start = time.time()
    mesh, point_data, field_data = voropy.read(args.infile)
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    num_nodes = len(mesh.node_coords)

    # create values
    print 'Creating psi...',
    start = time.time()
    X = np.empty( num_nodes, dtype = complex )
    X[:] = complex( 1.0, 0.0 )
    #for k, node in enumerate(mesh.node_coords):
        #import random, cmath
        #X[k] = cmath.rect( random.random(), 2.0 * pi * random.random() )
        #X[k] = 0.9 * np.cos(0.5 * node[0])
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    # Add magnetic vector potential.
    print 'Creating A...',
    start = time.time()
    A = np.empty((num_nodes, 3), dtype=float)
    import pyginla.magnetic_vector_potentials as mvp
    for k, node in enumerate(mesh.node_coords):
        A[k] = mvp.magnetic_dipole(x = node,
                                   x0 = np.array([0,0,6]),
                                   m = np.array([0,0,1])
                                   )
        #A[k] = mvp.constant_z( node )
        #A[k] = mvp.magnetic_dot(node, radius=2.0, height0=0.1, height1=1.1)
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    ## Add values for thickness:
    #thickness = np.empty(len(mesh.nodes), dtype = float)
    #alpha = 0.5 # thickness at the center of the tube
    #beta = 2.0 # thickness at the boundary
    #t = (beta-alpha) / b**2
    #for k, x in enumerate(mesh.nodes):
        #thickness[k] = alpha + t * x[1]**2

    # write the mesh
    print 'Writing mesh psi and A...',
    start = time.time()
    mesh.write(args.outfile, {'psi': X, 'A': A}, {'mu': 1.0})
    elapsed = time.time()-start
    print 'done. (%gs)' % elapsed

    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Reads a mesh an equips it with psi and A.' )

    parser.add_argument('infile',
                        metavar = 'INFILE',
                        type    = str,
                        help    = 'file that contains the mesh'
                        )
    
    parser.add_argument('outfile',
                        metavar = 'OUFILE',
                        type    = str,
                        help    = 'file to be written to'
                        )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
