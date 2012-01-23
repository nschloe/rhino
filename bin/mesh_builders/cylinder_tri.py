#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a simplisitc mesh on a cylinder strip.
'''
import vtkio
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
            points[k] = [ cos(u), sin(u), v ]
            k += 1

    # create the elements (cells)
    numelems = 2 * nl * (nw-1)
    elems = np.zeros( [numelems, 3], dtype=int )
    elem_types = np.zeros( numelems, dtype=int )
    k = 0
    for i in range(nl - 1):
        for j in range(nw - 1):
            elems[k]   = [ i*nw + j, (i + 1)*nw + j + 1,  i     *nw + j + 1 ]
            elems[k+1] = [ i*nw + j, (i + 1)*nw + j    , (i + 1)*nw + j + 1 ]
            elem_types[k]   = vtk.VTK_TRIANGLE
            elem_types[k+1] = vtk.VTK_TRIANGLE
            k += 2

    # close the geometry    
    for j in range(nw - 1):
        elems[k]   = [ (nl - 1)*nw + j, j + 1 , (nl - 1)*nw + j + 1 ]
        elems[k+1] = [ (nl - 1)*nw + j, j     , j + 1  ]
        elem_types[k]   = vtk.VTK_TRIANGLE
        elem_types[k+1] = vtk.VTK_TRIANGLE
        k += 2

    # write the mesh
    mesh_io.write_mesh( "cylinder.vtu", points, elems, elem_types )
# ------------------------------------------------------------------------------
