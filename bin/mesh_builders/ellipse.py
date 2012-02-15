# ==============================================================================
def _main():
    from math import pi, cos, sin, sqrt
    import numpy as np
    from scipy import special
    import mesh, mesh_io, meshpy_interface

    file_name = _parse_options()

    n_phi = 200
    # lengths of major and minor axes
    a = 15.0
    b = 15.0

    # Choose the maximum area of a triangle equal to the area of
    # an equilateral triangle on the boundary.
    # For circumference of an ellipse, see
    # http://en.wikipedia.org/wiki/Ellipse#Circumference
    eccentricity = sqrt( 1.0 - (b/a)**2 )
    length_boundary = float( 4 * a * special.ellipe( eccentricity ) )
    a_boundary = length_boundary / n_phi
    max_area = a_boundary**2 * sqrt(3) / 4

    # generate points on the circle
    Phi = np.linspace(0, 2*pi, n_phi, endpoint = False)
    for phi in Phi:
        points.append((a * cos(phi), b * sin(phi)))

    # create the mesh
    mymesh = meshpy_interface.create_mesh(max_area, points)

    # create values
    X = np.empty( len( mymesh.nodes ), dtype = complex )
    for k, x in enumerate( mymesh.nodes ):
        X[k] = complex( 1.0, 0.0 )

    # Add values for thickness:
    thickness = np.empty( len(mymesh.nodes), dtype = float )
    alpha = 0.5 # thickness at the center of the tube
    beta = 2.0 # thickness at the boundary
    t = (beta-alpha) / b**2
    for k, x in enumerate( mymesh.nodes ):
        thickness[k] = alpha + t * x.coords[1]**2

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
