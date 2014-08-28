#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of PyNosh.
#
#  PyNosh is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PyNosh is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyNosh.  If not, see <http://www.gnu.org/licenses/>.
#
'''Checker for triangle coefficients.
'''
# ==============================================================================
import numpy as np

#import pynosh.numerical_methods as nm
#import pynosh.ginla_modelevaluator as gm
import voropy
# ==============================================================================
def _main():
    args = _parse_input_arguments()


    # define triangle
    #x0 = np.array([0.0, 0.0])
    #v1 = np.array([1.0, 1.0])
    #v1 = v1 / np.linalg.norm(v1)
    #v2 = np.array([0.0, 1.0])
    #v2 = v2 / np.linalg.norm(v2)
    #alpha1 = np.sqrt(2)*0.1;
    #alpha2 = 0.1;
    #triangle_vertices = np.vstack([x0,
                                  #x0 + alpha1*v1,
                                  #x0 + alpha2*v2
                                  #])
    triangle_vertices = 2 * np.random.rand(3,2) - 1.0

    cc = compute_triangle_cc( triangle_vertices )
    triangle_vol = compute_triangle_vol( triangle_vertices )

    edges = np.array( [triangle_vertices[1] - triangle_vertices[0],
                       triangle_vertices[2] - triangle_vertices[1],
                       triangle_vertices[0] - triangle_vertices[2]] )

    edge_lenghts = np.array([np.linalg.norm(e) for e in edges])

    midpoints = 0.5 * np.array( [triangle_vertices[1] + triangle_vertices[0],
                                 triangle_vertices[2] + triangle_vertices[1],
                                 triangle_vertices[0] + triangle_vertices[2]] )

    if args.show_example:
        _show_example(triangle_vertices, midpoints, cc)
    # --------------------------------------------------------------------------
    # Find the coefficients numerically.
    A = np.dot(edges, edges.T)
    # Careful here! As of NumPy 1.7, np.diag() returns a view.
    rhs = triangle_vol * np.diag(A).copy()
    A = A**2

    weights = np.zeros(3)
    # Append the the resulting coefficients to the coefficient cache.
    # The system is posdef iff the simplex isn't degenerate.
    try:
        weights += np.linalg.solve(A, rhs)
    except np.linalg.linalg.LinAlgError:
        # The matrix A appears to be singular,
        # and the only circumstance that makes this
        # happening is the cell being degenerate.
        # Hence, it has volume 0, and so all the edge
        # coefficients are 0, too.
        # Hence, do nothing.
        pass

    check_weights(weights, edges, triangle_vol)
    # --------------------------------------------------------------------------
    # Qiang's formula.
    #theta = np.array([angle(triangle_vertices[0] - triangle_vertices[2],
                            #triangle_vertices[1] - triangle_vertices[2]),
                      #angle(triangle_vertices[2] - triangle_vertices[0],
                            #triangle_vertices[1] - triangle_vertices[0]),
                      #angle(triangle_vertices[2] - triangle_vertices[1],
                            #triangle_vertices[0] - triangle_vertices[1])
                      #])
    theta = np.array([angle( edges[2],-edges[1]),
                      angle(-edges[2], edges[0]),
                      angle( edges[1],-edges[0])
                      ])
    qweights = 0.5 * np.cos(theta) / np.sin(theta)

    #check_weights(qweights, edges, triangle_vol)

    print 'Compare weights with the previous... ',
    err = np.linalg.norm(qweights - weights)
    if err > 1.0e-14:
        print 'Ah! Diff =', qweights - weights
    else:
        print 'Cool.'

    ## possible extension to qiang's formula
    #u0 = [ rand(2,1); 0 ];
    #u1 = [ rand(2,1); 0 ];
    #p0 = u0'*u1 * triangle_volume( triangle_vertices[0], triangle_vertices[1], triangle_vertices[2] );
    #p1 = cot(theta(1)) * (u0'*edge[0])*(u1'*edge[0]) ...
       #+ cot(theta(2)) * (u0'*e{2})*(u1'*e{2}) ...
       #+ cot(theta(3)) * (u0'*e{3})*(u1'*e{3});
    #p0 - 0.5 * p1
    # --------------------------------------------------------------------------
    # Qiang's formula, passing the angle calculations.
    t = np.array([np.dot( edges[2]/edge_lenghts[2], -edges[1]/edge_lenghts[1]),
                  np.dot(-edges[2]/edge_lenghts[2],  edges[0]/edge_lenghts[0]),
                  np.dot( edges[1]/edge_lenghts[1], -edges[0]/edge_lenghts[0])
                  ])
    qweights = 0.5 * t / np.sqrt(1.0 - t**2)

    #check_weights(qweights, edges, triangle_vol)

    print 'Compare weights with the previous... ',
    err = np.linalg.norm(qweights - weights)
    if err > 1.0e-14:
        print 'Ah! Diff =', qweights - weights
    else:
        print 'Cool.'
    # --------------------------------------------------------------------------
    ## alternative computation of the weights
    #covolumes = np.array([np.linalg.norm(midpoints[0] - cc),
                          #np.linalg.norm(midpoints[1] - cc),
                          #np.linalg.norm(midpoints[2] - cc)
                          #])
    #edge_lenghts = np.array([np.linalg.norm(e) for e in edges])
    #cweights = covolumes / edge_lenghts

    #print 'Compare weights with the previous... ',
    #err = np.linalg.norm(cweights - weights)
    #if err > 1.0e-14:
        #print 'Ah! Diff =', cweights - weights
    #else:
        #print 'Cool.'
    # --------------------------------------------------------------------------


    return
# ==============================================================================
def _show_example(triangle_vertices, midpoints, cc):
    '''Show an example situation.'''
    from matplotlib import pyplot as pp

    # Plot the situation.
    for i in xrange(3):
        for j in xrange(i+1, 3):
            # Edge (i,j).
            pp.plot([triangle_vertices[i][0], triangle_vertices[j][0]],
                    [triangle_vertices[i][1], triangle_vertices[j][1]],
                    'k-' )
            # Line midpoint(edge(i,j))---circumcenter.
            midpoint = 0.5 * (triangle_vertices[i] + triangle_vertices[j])
            pp.plot([midpoint[0], cc[0]],
                    [midpoint[1], cc[1]],
                    color = '0.5' )

    # plot circumcenter
    pp.plot( cc[0], cc[1], 'or' )
    pp.show()

    return
# ==============================================================================
def compute_triangle_cc( node_coords ):
    '''Compute circumcenter.'''
    from vtk import vtkTriangle
    cc = np.empty([2,1])
    vtkTriangle.Circumcircle(node_coords[0], node_coords[1], node_coords[2],
                             cc)
    return cc
# ==============================================================================
def compute_triangle_vol( node_coords ):
    '''Compute triangle volume.'''
    # Shoelace formula.
    return 0.5 * abs( node_coords[0][0] * node_coords[1][1] - node_coords[0][1] * node_coords[1][0]
                    + node_coords[1][0] * node_coords[2][1] - node_coords[1][1] * node_coords[2][0]
                    + node_coords[2][0] * node_coords[0][1] - node_coords[2][1] * node_coords[0][0])
# ==============================================================================
def angle( u, v ):
    '''Computes the angle between two vectors.'''
    return np.arccos(np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v)))
# ==============================================================================
def check_weights( weights, edges, vol, tol=1.0e-14 ):
    '''Check if the given weights are correct.'''

    print 'Checking weights %g, %g, %g...' % (weights[0], weights[1], weights[2]),

    # try out the weight with a bunch of other random vectors
    m = 1000
    found_mismatch = False
    for i in xrange(m):
        u = np.random.rand(2) + 1j * np.random.rand(2)
        v = np.random.rand(2) + 1j * np.random.rand(2)

        control_value = np.vdot(u, v) * vol
        p1 = 0.0
        for j in xrange(3):
            p1 += np.vdot(u, edges[j]) * np.vdot(edges[j], v) * weights[j]

        err = abs(control_value-p1)
        if err > tol:
            found_mismatch = True;
            print 'Found mismatch by %g.\n' % err
            break

    if not found_mismatch:
        print 'Cool.'

    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser( description = 'Test edge coefficients for the triangle.' )

    parser.add_argument('-s', '--show-example',
                        action = 'store_true',
                        default = False,
                        help    = 'Show an example triangle with points highlighted (default: False).'
                        )

    #parser.add_argument('filename',
                        #metavar = 'FILE',
                        #type    = str,
                        #help    = 'ExodusII file containing the geometry and initial state'
                        #)

    #parser.add_argument('--show', '-s',
                        #action = 'store_true',
                        #default = False,
                        #help    = 'show the relative residuals of each linear iteration (default: False)'
                        #)

    return parser.parse_args()
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
