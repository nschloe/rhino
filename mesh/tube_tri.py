#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Creates a simplistic triangular mesh on a M\"obius strip.
'''
import vtk
import mesh, mesh_io
import numpy as np
from math import pi, sin, cos
# ==============================================================================
def _main():
    # get the file name to be written to
    file_name = _parse_options()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # width of the tube
    width = 3.0
    # radius of the tube
    r = 1.0
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Mesh parameters
    # Number of nodes along the length of the strip
    nl = 190
    # Number of nodes along the width of the strip (>= 2)
    nw = int( round( width * nl/(2*pi*r) ) ) # to have approximately square boxes
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Generate suitable ranges for parametrization
    u_range = np.linspace( 0.0, 2*pi, num = nl, endpoint = False )
    v_range = np.linspace( -0.5*width, 0.5*width, num = nw )

    # Create the vertices.
    nodes = []
    for u in u_range:
        for v in v_range:
            nodes.append( mesh.Node( [ r * cos(u), r * sin(u), v ] ) )

    # create the elements (cells)
    elems = []
    for i in range(nl - 1):
        for j in range(nw - 1):
            elem_nodes = [ i*nw + j, (i + 1)*nw + j + 1,  i     *nw + j + 1 ]
            elems.append( mesh.Cell( elem_nodes,
                                     [], # edges
                                     [], # faces
                                     vtk.VTK_TRIANGLE
                                   )
                        )
            elem_nodes = [ i*nw + j, (i + 1)*nw + j    , (i + 1)*nw + j + 1 ]
            elems.append( mesh.Cell( elem_nodes,
                                     [], # edges
                                     [], # faces
                                     vtk.VTK_TRIANGLE
                                   )
                        )
    # close the geometry
    for j in range(nw - 1):
        elem_nodes = [ (nl - 1)*nw + j, j + 1 , (nl - 1)*nw + j + 1 ]
        elems.append( mesh.Cell( elem_nodes,
                                 [], # edges
                                 [], # faces
                                 vtk.VTK_TRIANGLE
                               )
                    )
        elem_nodes = [ (nl - 1)*nw + j, j     , j + 1  ]
        elems.append( mesh.Cell( elem_nodes,
                                 [], # edges
                                 [], # faces
                                 vtk.VTK_TRIANGLE
                               )
                    )

    # add values
    num_nodes = len( nodes )
    X = np.empty( num_nodes, dtype = complex )
    k = 0
    for u in u_range:
        for v in v_range:
            X[k] = complex( 1.0, 0.0 )
            k += 1

    # Add values for thickness:
    # Make it somewhat thicker at the boundaries.
    thickness = np.empty( num_nodes, dtype = float )
    alpha = 0.5 # thickness at the center of the tube
    beta = 2.0 # thickness at the boundary
    t = (beta-alpha) / (0.5*width)**2
    k = 0
    for u in u_range:
        for v in v_range:
            #thickness[k] = alpha + t * nodes[k].coords[2]**2
            thickness[k] = 1.0
            k += 1

    # add parameters
    params = { "mu": 0.0 }

    # create the mesh data structure
    mymesh = mesh.Mesh( nodes, elems )

    # create the mesh
    mesh_io.write_mesh( file_name,
                        mymesh,
                        [X,thickness], ["psi","thickness"],
                        params
                       )
    return
# ==============================================================================
def _parse_options():
    '''Parse input options.'''
    import optparse, sys

    usage = "usage: %prog outfile"

    parser = optparse.OptionParser( usage = usage )

    (options, args) = parser.parse_args()

    if not args  or  len(args) != 1:
        parser.print_help()
        sys.exit( "\nProvide a file to be written to." )

    return args[0]
# ==============================================================================
if __name__ == "__main__":
    _main()
# ==============================================================================
