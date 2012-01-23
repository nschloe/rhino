#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a simplisitc mesh on a rectangle.
'''
import vtk
import mesh, mesh_io
import numpy as np
from cmath import sin, pi
# ==============================================================================
def _main():

    # get the file name to be written to
    file_name = _parse_options()

    # dimensions of the rectangle
    lx = 10.0
    ly = 1.0

    # Mesh parameters
    # Number of nodes along the length of the strip
    nx = 2
    ny = 2

    # Generate suitable ranges for parametrization
    x_range = np.linspace( -0.5*lx, 0.5*lx, nx )
    y_range = np.linspace( -0.5*ly, 0.5*ly, ny )

    # Create the vertices.
    num_nodes = nx * ny
    nodes = []
    for x in x_range:
        for y in y_range:
            nodes.append( mesh.Node( [ x, y, 0 ] ) )

    # create the elements (cells)
    num_elems = 2 * (nx-1) * (ny-1)
    elems = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            elems.append( mesh.Cell( [ i*ny + j, (i + 1)*ny + j + 1,  i     *ny + j + 1 ],
                                     [], # edges
                                     [], # faces
                                     vtk.VTK_TRIANGLE
                                   )
                        )
            elems.append( mesh.Cell( [ i*ny + j, (i + 1)*ny + j    , (i + 1)*ny + j + 1 ],
                                     [], # edges
                                     [], # faces
                                     vtk.VTK_TRIANGLE
                                   )
                        )

    # create values
    X = np.empty( num_nodes, dtype = complex )
    k = 0
    for x in x_range:
        for y in y_range:
            X[k] = 1.0
            #X[k] = complex( x, y )
            #X[k] = complex( sin( x/lx * pi ), sin( y/ly * pi ) )
            k += 1

    # add thickness value
    thickness = np.empty( num_nodes, dtype = float )
    alpha = 1.0
    beta = 2.0
    k = 0
    for x in x_range:
        for y in y_range:
            #thickness[k] = alpha + (beta-alpha) * (y/(0.5*ly))**2
            thickness[k] = 1.0
            k += 1

    # add magnetic vector potential
    import magnetic_vector_potentials
    A = np.empty( (num_nodes,3), dtype = float )
    k = 0
    for x in x_range:
        for y in y_range:
            A[k,:] = magnetic_vector_potentials.mvp_z( nodes[k].coords )
            k += 1

    # add parameters
    params = { "mu": 0.0,
               "scaling": 1.0
             }

    # create the mesh data structure
    mymesh = mesh.Mesh( nodes, elems )

    # write the mesh with data
    mesh_io.write_mesh( file_name,
                        mymesh,
                        [X,A,thickness], ["psi","A","thickness"],
                        #[X], ["psi"],
                        params
                      )
    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser( description = 'Construct a trival triangulation of a rectangle.' )


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
