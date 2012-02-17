#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Solve the Ginzburg--Landau equation.
'''
# ==============================================================================
import mesh.mesh_io
import numpy as np
import vtk
# ==============================================================================
def _main():
    '''Main function.
    '''
    filename = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )
    print 'done.'

    # This approach here is pretty brute-force. For more sophisticated
    # algorithms, see
    # http://en.wikipedia.org/wiki/Delaunay_triangulation#Algorithms.
    
    is_delaunay = True
    for cellNodes in pyginlamesh.cellsNodes:
        # Calculate the circumsphere.
        cc = np.empty(3,float)
        if len(cellNodes) == 3: # triangle
            # Project triangle to 2D.
            v = np.empty(3, dtype=np.dtype((float,2)))
            vtk.vtkTriangle.ProjectTo2D(pyginlamesh.nodes[cellNodes[0]],
                                        pyginlamesh.nodes[cellNodes[1]],
                                        pyginlamesh.nodes[cellNodes[2]],
                                        v[0], v[1], v[2])
            # Get the circumcenter in 2D.
            cc_2d = np.empty(2,dtype=float)
            r_squared = vtk.vtkTriangle.Circumcircle(v[0], v[1], v[2], cc_2d)
            # Project back to 3D by using barycentric coordinates.
            bcoords = np.empty(3,dtype=float)
            vtk.vtkTriangle.BarycentricCoords(cc_2d, v[0], v[1], v[2], bcoords)
            cc = bcoords[0] * pyginlamesh.nodes[cellNodes[0]] \
               + bcoords[1] * pyginlamesh.nodes[cellNodes[1]] \
               + bcoords[2] * pyginlamesh.nodes[cellNodes[2]]

        elif len(cellNodes) == 4: # tet
            r_squared = vtk.vtkTetra.Circumsphere(pyginlamesh.nodes[cellNodes[0]],
                                                  pyginlamesh.nodes[cellNodes[1]],
                                                  pyginlamesh.nodes[cellNodes[2]],
                                                  pyginlamesh.nodes[cellNodes[3]],
                                                  cc)
        else:
            raise RuntimeError('Can only handle triangles and tets.')

        # Check if any node sits inside the circumsphere.
        for node in pyginlamesh.nodes:
            d = cc - node
            alpha = np.dot(d, d)
            # Add a bit of a tolerance here to make sure that the current cell's
            # nodes aren't counted in.
            if alpha < r_squared - 1.0e-10:
                print 'The point', node, 'sits inside the circumsphere of the cell given by cell', cellNodes, '.', np.sqrt(alpha) - np.sqrt(r_squared)
                is_delaunay = False

    if is_delaunay:
        print 'The given mesh is a Delaunay mesh.'

    return
# ==============================================================================
def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse

    parser = argparse.ArgumentParser(description = 'Check if a mesh is a Delaunay mesh.')

    parser.add_argument('filename',
                        metavar = 'FILE',
                        type    = str,
                        help    = 'ExodusII file containing the geometry'
                        )

    args = parser.parse_args()

    return args.filename
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
