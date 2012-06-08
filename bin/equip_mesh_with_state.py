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
    if not args.force_override and 'psi' in point_data:
        psi = point_data['psi']
    else:
        print 'Creating psi...',
        start = time.time()
        psi = np.empty( num_nodes, dtype = complex )
        psi[:] = complex( 1.0, 0.0 )
        #for k, node in enumerate(mesh.node_coords):
            #import random, cmath
            #psi[k] = cmath.rect( random.random(), 2.0 * pi * random.random() )
            #psi[k] = 0.9 * np.cos(0.5 * node[0])
        elapsed = time.time()-start
        print 'done. (%gs)' % elapsed

    if not args.force_override and 'V' in point_data:
        V = point_data['V']
    else:
        # create values
        print 'Creating V...',
        start = time.time()
        V = np.empty(num_nodes)
        V[:] = -1
        #for k, node in enumerate(mesh.node_coords):
            #import random, cmath
            #X[k] = cmath.rect( random.random(), 2.0 * pi * random.random() )
            #X[k] = 0.9 * np.cos(0.5 * node[0])
        elapsed = time.time()-start
        print 'done. (%gs)' % elapsed

    if not args.force_override and 'A' in point_data:
        A = point_data['A']
    else:
        # If this is a 2D mesh, append the z-component 0 to each node
        # to make sure that the magnetic vector potentials can be
        # calculated.
        points = mesh.node_coords.copy()
        if points.shape[1] == 2:
            points = np.column_stack((points, np.zeros(len(points))))
        # Add magnetic vector potential.
        print 'Creating A...',
        start = time.time()
        #A = points # field A(X) = X -- test case
        import pyginla.magnetic_vector_potentials as mvp
        #A = np.zeros((num_nodes,3))
        #B = np.array([np.cos(theta) * np.cos(phi),
                      #np.cos(theta) * np.sin(phi),
                      #np.sin(theta)])
        #A = mvp.constant_field(points, np.array([1,1,1]) / np.sqrt(3))
        A = mvp.magnetic_dipole(points,
                                x0 = np.array([0,0,2]),
                                m = np.array([0,0,1])
                                )
        #A = mvp.magnetic_dot(points, radius=2.0, heights=[0.1, 1.1])
        #A = np.empty((num_nodes, 3), dtype=float)
        #for k, node in enumerate(points):
            #A[k] = mvp.magnetic_dot(node, radius=2.0, height0=0.1, height1=1.1)
        elapsed = time.time()-start
        print 'done. (%gs)' % elapsed

    #if 'thickness' in point_data:
    #    thickness = point_data['thickness']
    #else:
    #    # Add values for thickness:
    #    thickness = np.empty(num_nodes, dtype = float)
    #    alpha = 0.5 # thickness at the center of the tube
    #    beta = 2.0 # thickness at the boundary
    #    t = (beta-alpha) / b**2
    #    for k, x in enumerate(mesh.nodes):
    #        thickness[k] = alpha + t * x[1]**2

    if not args.force_override and 'g' in field_data:
        g = field_data['g']
    else:
        g = 1.0

    if not args.force_override and 'mu' in field_data:
        mu = field_data['mu']
    else:
        mu = 1.0

    # write the mesh
    print 'Writing mesh...',
    start = time.time()
    mesh.write(args.outfile, point_data={'psi': psi, 'V': V, 'A': A}, field_data={'g': g, 'mu': mu})
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

    parser.add_argument('--force-override', '-f',
                        action = 'store_true',
                        default = False,
                        help    = 'override the values present in the input files'
                        )

    args = parser.parse_args()

    return args
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
