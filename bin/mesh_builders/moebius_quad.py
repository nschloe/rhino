# -*- coding: utf-8 -*-
'''
Creates a simplistic quad mesh on a M\"obius strip.
'''
import vtk
import numpy as np
from math import pi, sin, cos
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # The width of the strip
    width = 1.0

    # Mesh parameters
    # Number of nodes along the length of the strip
    nl = 100
    # Number of nodes along the width of the strip (>= 2)
    nw = 10
    # How many twists are there in the "paper"?
    moebius_index = 1

    # Generate suitable ranges for parametrization
    u_range = np.arange( nl, dtype='d' ) \
            / (nl)*2*pi
    v_range = np.arange( nw, dtype='d' ) \
            / (nw - 1.0)*width \
            - 0.5*width

    # Create the vertices. This is based on the parameterization
    # of the M\"obius strip as given in
    # <http://en.wikipedia.org/wiki/M%C3%B6bius_strip#Geometry_and_topology>
    numpoints = nl * nw
    points = np.zeros( [numpoints, 3] )
    k = 0
    for u in u_range:
        for v in v_range:
            points[k] = [ ( 1 + v*cos(moebius_index * 0.5*u) )*cos(u),
                          ( 1 + v*cos(moebius_index * 0.5*u) )*sin(u),
                                v*sin(moebius_index * 0.5*u)
                        ]
            k += 1

    # create the elements (cells)
    numelems =  nl * (nw-1)
    elems = np.zeros( [numelems, 4], dtype=int )
    elem_types = np.zeros( numelems, dtype=int )
    k = 0
    for i in range(nl - 1):
        for j in range(nw - 1):
            elems[k]   = [ i     *nw + j,
                          (i + 1)*nw + j,
                          (i + 1)*nw + j + 1,
                           i     *nw + j + 1 ]
            elem_types[k] = vtk.VTK_QUAD
            k += 1

    # close the geometry    
    if moebius_index % 2 == 0:
        # Close the geometry upside up (even M\obius fold)
        for j in range(nw - 1):
            elems[k]   = [ (nl - 1)*nw + j,
                                         j,
                                         j + 1 ,
                           (nl - 1)*nw + j + 1 ]
            elem_types[k] = vtk.VTK_QUAD
            k += 1
    else:
        # Close the geometry upside down (odd M\obius fold)
        for j in range(nw - 1):
            elems[k]   = [ (nl-1)*nw +  j    ,
                           (nw-1)    -  j    ,
                           (nw-1)    - (j+1) ,
                           (nl-1)*nw +  j+1  ]
            elem_types[k] = vtk.VTK_QUAD
            k += 1

    # create the mesh
    mesh = create_mesh( points, elems )
    write_mesh( mesh )
# ------------------------------------------------------------------------------
