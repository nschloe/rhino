'''Module that provides magnetic vector potentials.'''
import numpy


def constant_field(X, B):
    '''Converts a spatially constant magnetic field B at X
    into a corresponding potential.
    '''
    # This is one particular choice that works.
    return 0.5 * numpy.cross(B, X)


def magnetic_dipole(x, x0, m):
    '''Magnetic vector potential for the static dipole at x0 with orientation
    m.
    '''
    r = x - x0
    # npsum(...) = ||r||^3 row-wise;
    # numpy.cross acts on rows by default;
    # The ".T" magic makes sure that each row of numpy.cross(m, r)
    # gets divided by the corresponding entry in ||r||^3.
    return (numpy.cross(m, r).T /
            numpy.sum(numpy.abs(r)**2, axis=-1)**(3./2)
            ).T


def magnetic_dot(X, radius, heights):
    '''Magnetic vector potential corresponding to the field that is induced by
    a cylindrical magnetic dot, centered at (0,0,0.5*(height0+height1)), with
    the radius `radius` for objects in the x-y-plane.  The potential is derived
    by interpreting the dot as an infinitesimal collection of magnetic dipoles,
    hence

       A(x) = \int_{dot} A_{dipole}(x-r) dr.

    Support for input valued (x,y,z), z!=0, is pending.
    '''
    # Span a cartesian grid over the sample, and integrate over it.
    # For symmetry, choose a number that is divided by 4.
    n_phi = 100
    # Choose such that the quads at radius/2 are approximately squares.
    n_radius = int(round(n_phi / numpy.pi))

    dr = radius / n_radius

    A = numpy.zeros((len(X), 3))

    # What we want to have is the value of
    #
    #    I(X) := \int_{dot} \|X-XX\|^{-3/2} (m\times(X-XX)) dXX
    #
    # with
    #
    #    X := (x, y, z)^T,
    #    XX := (xx, yy, zz)^T
    #
    # The integral in zz-direction (height) can be calculated analytically,
    # such that
    #
    #    I = \int_{disk}
    #          [ - (z-zz) / (r2D*sqrt(r3D)) ]_{zz=h_0}^{h_1}
    #          ( -(y-yy), x-xx, 0)^T dxx dyy.
    #
    # The integral over the disk is then approximated numerically by
    # the summation over little disk segments.
    # An alternative is to use cylindrical coordinates.
    #
    X_dist = numpy.empty((X.shape[0], 2))
    for i_phi in range(n_phi):
        beta = 2.0 * numpy.pi / n_phi * i_phi
        sin_beta = numpy.sin(beta)
        cos_beta = numpy.cos(beta)
        for i_radius in range(n_radius):
            rad = radius / n_radius * (i_radius + 0.5)
            # r = squared distance between grid point X to the
            #     point (x,y) on the magnetic dot
            X_dist[:, 0] = X[:, 0] - rad * cos_beta
            X_dist[:, 1] = X[:, 1] - rad * sin_beta

            # r = x_dist * x_dist + y_dist * y_dist
            # Note that X_dist indeed only has two components.
            R = numpy.sum(X_dist**2, axis=1)
            ind = numpy.nonzero(R > 1.0e-15)

            # 3D distance to point on lower edge (xi,yi,height0)
            # and upper edge ( (xi,yi,height1), respectively
            R_3D = [numpy.sqrt(R[ind] + heights[0]**2),
                    numpy.sqrt(R[ind] + heights[1]**2)
                    ]
            # Volume of circle segment = pi*angular_width * r^2,
            # so the volume of a building brick of the discretization is
            #   V = pi/n_phi * [(r+dr/2)^2 - (r-dr/2)^2]
            #     = pi/n_phi * 2 * r * dr.
            Alpha = (heights[1]/R_3D[1] - heights[0]/R_3D[0]) / R[ind] \
                * numpy.pi / n_phi * (2.0*rad*dr)  # volume
            # ax += y_dist * alpha
            # ay -= x_dist * alpha
            A[ind, 0] += X_dist[ind, 1] * Alpha
            A[ind, 1] -= X_dist[ind, 0] * Alpha
    return A
