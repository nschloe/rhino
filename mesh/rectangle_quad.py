#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a simplistic quad mesh on a rectangle.
'''
import vtk
import numpy as np
from math import pi, sin, cos
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # dimensions of the rectangle
    lx = 2.0
    ly = 1.0

    # Mesh parameters
    # Number of nodes along the length of the strip
    nx = 20
    ny = 30

    # Generate suitable ranges for parametrization
    x_range = np.arange( nx, dtype='d' ) \
            / (nx-1) * lx
    y_range = np.arange( ny, dtype='d' ) \
            / (ny-1) * ly

    # Create the vertices. This is based on the parameterization
    # of the M\"obius strip as given in
    # <http://en.wikipedia.org/wiki/M%C3%B6bius_strip#Geometry_and_topology>
    numpoints = nx * ny
    points = np.zeros( [numpoints, 3] )
    k = 0
    for x in x_range:
        for y in y_range:
            points[k] = [ x, y, 0 ]
            k += 1

    # create the elements (cells)
    numelems = (nx-1) * (ny-1)
    elems      = np.zeros( [numelems, 4], dtype=int )
    elem_types = np.zeros( numelems, dtype=int )
    k = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            elems[k]   = [  i     *ny + j, 
                           (i + 1)*ny + j,
                           (i + 1)*ny + j + 1, 
                            i     *ny + j + 1 ]
            elem_types[k] = vtk.VTK_QUAD
            k += 1

    # write mesh
    mesh_io.write_mesh( "quad-rectangle.vtu", points, elems, elem_types )
# ------------------------------------------------------------------------------
