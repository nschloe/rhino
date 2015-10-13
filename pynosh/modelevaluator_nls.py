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
'''
Provide information around the nonlinear Schrödinger equations.
'''
import numpy
from scipy import sparse, linalg
import warnings
import krypy


class NlsModelEvaluator(object):
    '''Nonlinear Schrödinger model evaluator class.
    Incorporates

       * Nonlinear Schrödinger: :math:`g=1.0, V=0.0, A=0.0`.
       * Gross--Pitaevskii: :math:`g=1.0`, :math:`V` given, :math:`A=0.0`.
       * Ginzburg--Landau: :math:`g=1.0, V=-1.0`,
         and some magnetic potential :math:`A`.
    '''
    def __init__(self,
                 mesh,
                 V=None,
                 A=None,
                 preconditioner_type='none',
                 num_amg_cycles=numpy.inf
                 ):
        '''Initialization. Set mesh.
        '''
        self.dtype = complex
        self.mesh = mesh
        n = len(mesh.node_coords)
        if V is None:
            self._V = numpy.zeros(n)
        else:
            self._V = V
        if A is None:
            self._raw_magnetic_vector_potential = numpy.zeros((n, 3))
        else:
            self._raw_magnetic_vector_potential = A
        self._keo_cache = None
        self._keo_cache_mu = 0.0
        self._edgecoeff_cache = None
        self.tot_amg_cycles = []
        self.cv_variant = 'voronoi'
        self._preconditioner_type = preconditioner_type
        self._num_amg_cycles = num_amg_cycles
        return

    def compute_f(self, x, mu, g):
        '''Computes the nonlinear Schrödinger residual

        .. math::
            GP(\\psi) = K\\psi + (V + g |\\psi|^2) \\psi
        '''
        keo = self._get_keo(mu)
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes(variant=self.cv_variant)
        res = (keo * x) / self.mesh.control_volumes.reshape(x.shape) \
            + (self._V.reshape(x.shape) + g * abs(x)**2) * x
        return res

    def get_jacobian(self, x, mu, g):
        '''Returns a LinearOperator object that defines the matrix-vector
        multiplication scheme for the Jacobian operator as in

        .. math::
            A \\phi + B \\phi^*

        with

        .. math::
            A &= K + I (V + g \\cdot 2|\\psi|^2),\\\\
            B &= g \\cdot  diag( \\psi^2 ).
        '''
        def _apply_jacobian(phi):
            if len(phi.shape) == 1:
                shape = phi.shape
            elif len(phi.shape) == 2:
                # phi may be a vector of shape (n, k).
                shape = (phi.shape[0], 1)
            else:
                raise ValueError('Illegal phi.')
            y = (keo * phi) / self.mesh.control_volumes.reshape(shape) \
                + alpha.reshape(shape) * phi \
                + gPsi0Squared.reshape(shape) * phi.conj()
            return y

        assert x is not None

        keo = self._get_keo(mu)

        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes(variant=self.cv_variant)
        alpha = self._V.reshape(x.shape) + g * 2.0*(x.real**2 + x.imag**2)
        gPsi0Squared = g * x**2

        num_unknowns = len(self.mesh.node_coords)

        return krypy.utils.LinearOperator((num_unknowns, num_unknowns),
                                          self.dtype,
                                          dot=_apply_jacobian,
                                          dot_adj=_apply_jacobian
                                          )

    def get_jacobian_blocks(self, x, mu, g):
        '''Returns

        .. math::
            A &= K + I  (V + g \\cdot 2|\\psi|^2),\\\\
            B &= g \\cdot diag( \\psi^2 ).
        '''
        assert x is not None

        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes(variant=self.cv_variant)

        A = self._get_keo(mu).copy()
        diag = A.diagonal()
        alpha = self._V.reshape(x.shape) + g * 2.0 * (x.real**2 + x.imag**2)
        diag += alpha.reshape(diag.shape) \
            * self.mesh.control_volumes.reshape(x.shape)
        A.setdiag(diag)

        num_nodes = len(self.mesh.node_coords)
        from scipy.sparse import spdiags
        B = spdiags(g * x**2 * self.mesh.control_volumes.reshape(x.shape),
                    [0],
                    num_nodes, num_nodes)
        return A, B

    def get_preconditioner(self, x, mu, g):
        '''Return the preconditioner.
        '''
        if self._preconditioner_type == 'none':
            return None
        if self._preconditioner_type == 'cycles':
            warnings.warn(
                'Preconditioner inverted approximately with '
                '%d AMG cycles, so get_preconditioner() isn\'t exact.'
                % self._num_amg_cycles
                )

        def _apply_precon(phi):
            return (keo * phi) / self.mesh.control_volumes.reshape(phi.shape) \
                + alpha.reshape(phi.shape) * phi
                # + beta.reshape(phi.shape) * phi.conj()

        assert x is not None

        keo = self._get_keo(mu)

        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes(variant=self.cv_variant)

        if g > 0.0:
            alpha = g * 2.0 * (x.real**2 + x.imag**2)
            # beta = g * x**2
        else:
            alpha = numpy.zeros(len(x))
        num_unknowns = len(self.mesh.node_coords)
        return krypy.utils.LinearOperator((num_unknowns, num_unknowns),
                                          self.dtype,
                                          dot=_apply_precon
                                          )

    def get_preconditioner_inverse(self, x, mu, g):
        '''Use AMG to invert M approximately.
        '''
        if self._preconditioner_type == 'none':
            return None
        import pyamg

        num_unknowns = len(x)

        def _apply_inverse_prec_exact(phi):
            assert(len(phi.shape) == 2)
            assert(len(self.mesh.control_volumes.shape) == 1)
            rhs = numpy.empty(phi.shape, dtype=phi.dtype)
            sol = numpy.empty(phi.shape, dtype=phi.dtype)
            for i in range(phi.shape[1]):
                rhs = self.mesh.control_volumes * phi[:, i]
                linear_system = krypy.linsys.LinearSystem(
                    prec,
                    rhs,
                    M=amg_prec,
                    self_adjoint=True,
                    positive_definite=True
                    )
                x_init = numpy.zeros((num_unknowns, 1), dtype=complex)
                out = krypy.linsys.Cg(linear_system,
                                      x0=x_init,
                                      tol=1.0e-13,
                                      # explicit_residual = False
                                      )
                sol[:, i] = out.xk[:, 0]
                # Forget about the cycle used to gauge the residual norm.
                self.tot_amg_cycles += [len(out.resnorms) - 1]
            return sol

        def _apply_inverse_prec_cycles(phi):
            rhs = self.mesh.control_volumes.reshape((phi.shape[0], 1)) * phi
            x_init = numpy.zeros((num_unknowns, 1), dtype=complex)
            x = numpy.empty(phi.shape, dtype=complex)
            residuals = []
            for i in range(rhs.shape[1]):
                x[:, i] = prec_amg_solver.solve(rhs[:, i],
                                                x0=x_init,
                                                maxiter=self._num_amg_cycles,
                                                tol=0.0,
                                                accel=None,
                                                residuals=residuals
                                                )
            # Alternative for one cycle:
            # amg_prec = prec_amg_solver.aspreconditioner( cycle='V' )
            # x = amg_prec * rhs
            self.tot_amg_cycles += [self._num_amg_cycles]
            return x

        keo = self._get_keo(mu)

        if g > 0.0:
            if self.mesh.control_volumes is None:
                self.mesh.compute_control_volumes(variant=self.cv_variant)
            # don't use .setdiag,
            # cf. https://github.com/scipy/scipy/issues/3501
            alpha = g * 2.0 * (x.real**2 + x.imag**2) \
                * self.mesh.control_volumes.reshape(x.shape)
            prec = keo \
                + sparse.spdiags(alpha[:, 0], [0], num_unknowns, num_unknowns)
        else:
            prec = keo

        # The preconditioner assumes the eigenvalue 0 iff mu=0 and psi=0.
        # This may lead to problems if mu=0 and the Newton iteration
        # converges to psi=0 for psi0 != 0.
        # import scipy.sparse.linalg
        # lambd, v = scipy.sparse.linalg.eigs(prec, which='SM')
        # assert all(abs(lambd.imag) < 1.0e-15)
        # print '||psi||^2 = %g' % numpy.linalg.norm(absPsi0Squared)
        # print 'lambda =', lambd.real

        prec_amg_solver = \
            pyamg.smoothed_aggregation_solver(
                prec,
                strength=('evolution',
                          {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}
                          ),
                smooth=('energy',
                        {'weighting': 'local',
                         'krylov': 'cg',
                         'degree': 2,
                         'maxiter': 3
                         }),
                improve_candidates=None,
                aggregate='standard',
                presmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric', 'iterations': 1}
                             ),
                postsmoother=('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 1}
                              ),
                max_levels=25,
                coarse_solver='splu'
                )

        # print 'operator complexity', prec_amg_solver.operator_complexity()
        # print 'cycle complexity', prec_amg_solver.cycle_complexity('V')

        if self._preconditioner_type == 'cycles':
            if self._num_amg_cycles == numpy.inf:
                raise ValueError('Invalid number of cycles.')
            return krypy.utils.LinearOperator((num_unknowns, num_unknowns),
                                              self.dtype,
                                              dot=_apply_inverse_prec_cycles
                                              )
        elif self._preconditioner_type == 'exact':
            amg_prec = prec_amg_solver.aspreconditioner(cycle='V')
            return krypy.utils.LinearOperator((num_unknowns, num_unknowns),
                                              dtype=self.dtype,
                                              dot=_apply_inverse_prec_exact
                                              )
        else:
            raise ValueError('Unknown preconditioner type ''%s''.'
                             % self._preconditioner_type
                             )

    def _get_preconditioner_inverse_directsolve(self, x, mu, g):
        '''Use a direct solver for M^{-1}.
        '''
        from scipy.sparse.linalg import spsolve

        def _apply_inverse_prec(phi):
            return spsolve(prec, phi)
        prec = self.get_preconditioner(x, mu, g)
        num_unknowns = len(x)
        return krypy.Utils.LinearOperator(
            (num_unknowns, num_unknowns),
            _apply_inverse_prec,
            dtype=self.dtype
            )

    def inner_product(self, phi0, phi1):
        '''The natural inner product of the problem.
        '''
        assert phi0.shape[0] == phi1.shape[0], \
            ('Input vectors not matching.', phi0.shape, phi1.shape)
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes(variant=self.cv_variant)
        if len(phi0.shape) == 1:
            scaledPhi0 = self.mesh.control_volumes * phi0
        elif len(phi0.shape) == 2:
            scaledPhi0 = \
                self.mesh.control_volumes.reshape((phi0.shape[0], 1)) * phi0
        # numpy.vdot only works for vectors, so use numpy.dot(....T.conj())
        # here.
        return numpy.dot(scaledPhi0.T.conj(), phi1).real

    def energy(self, psi):
        '''Compute the Gibbs free energy.
        Not really a norm, but a good measure for our purposes here.
        '''
        if self.mesh.control_volumes is None:
            self.mesh.compute_control_volumes(variant=self.cv_variant)
        alpha = -self.inner_product(psi**2, psi**2)
        return alpha.real / self.mesh.control_volumes.sum()

    def _get_keo(self, mu):
        '''Assemble the kinetic energy operator.'''

        if self._keo_cache is None or self._keo_cache_mu != mu:
            # Create the matrix structure.
            num_nodes = len(self.mesh.node_coords)

            mvp_edge_cache = self._build_mvp_edge_cache(mu)

            # Build caches.
            if self._edgecoeff_cache is None:
                self._build_edgecoeff_cache()
            if self.mesh.edges is None:
                self.mesh.create_adjacent_entities()

            n_edges = len(self.mesh.edges['nodes'])
            row = numpy.zeros(4*n_edges, dtype=int)
            col = numpy.zeros(4*n_edges, dtype=int)
            data = numpy.zeros(4*n_edges, dtype=complex)

            # loop over all edges
            for k, node_indices in enumerate(self.mesh.edges['nodes']):
                # Fetch the cached values.
                alpha = self._edgecoeff_cache[k]
                alphaExp0 = self._edgecoeff_cache[k] \
                    * numpy.exp(1j * mvp_edge_cache[k])
                # Sum them into the matrix.
                row[4*k:4*k+4] = [node_indices[0], node_indices[0],
                                  node_indices[1], node_indices[1]]
                col[4*k:4*k+4] = [node_indices[0], node_indices[1],
                                  node_indices[0], node_indices[1]]
                data[4*k:4*k+4] = [alpha, -alphaExp0.conj(),
                                   -alphaExp0, alpha]

            self._keo_cache = sparse.csr_matrix((data, (row, col)),
                                                (num_nodes, num_nodes))
            # transform the matrix into the more efficient CSR format
            self._keo_cache_mu = mu
        return self._keo_cache

    def _build_edgecoeff_cache(self):
        '''Build cache for the edge coefficients.
        (in 2D: coedge-edge ratios).
        '''
        # make sure the mesh has edges
        if self.mesh.edges is None:
            self.mesh.create_adjacent_entities()

        num_edges = len(self.mesh.edges)
        self._edgecoeff_cache = numpy.zeros(num_edges, dtype=float)

        if self.mesh.cell_volumes is None:
            self.mesh.create_cell_volumes()
        vols = self.mesh.cell_volumes

        # Precompute edges.
        edges = self.mesh.node_coords[self.mesh.edges['nodes'][:, 1]] \
            - self.mesh.node_coords[self.mesh.edges['nodes'][:, 0]]

        # Calculate the edge contributions cell by cell.
        for vol, cell in zip(vols, self.mesh.cells):
            cell_edge_gids = cell['edges']
            # Build the equation system:
            # The equation
            #
            # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>
            #
            # has to hold for all vectors u in the plane spanned by the edges,
            # particularly by the edges themselves.
            A = numpy.dot(edges[cell_edge_gids], edges[cell_edge_gids].T)
            # Careful here! As of NumPy 1.7, numpy.diag() returns a view.
            rhs = vol * numpy.diag(A).copy()
            A = A**2

            # Append the the resulting coefficients to the coefficient cache.
            # The system is posdef iff the simplex isn't degenerate.
            try:
                self._edgecoeff_cache[cell_edge_gids] += \
                    linalg.solve(A, rhs, sym_pos=True)
            except numpy.linalg.linalg.LinAlgError:
                # The matrix A appears to be singular,
                # and the only circumstance that makes this
                # happening is the cell being degenerate.
                # Hence, it has volume 0, and so all the edge
                # coefficients are 0, too.
                # Hence, do nothing.
                pass
        return

    def _build_mvp_edge_cache(self, mu):
        '''Builds the cache for the magnetic vector potential.'''

        # make sure the mesh has edges
        if self.mesh.edges is None:
            self.mesh.create_adjacent_entities()

        # Approximate the integral
        #
        #    I = \int_{x0}^{xj} (xj-x0)/|xj-x0| . A(x) dx
        #
        # numerically by the midpoint rule, i.e.,
        #
        #    I ~ (xj-x0) . A( 0.5*(xj+x0) )
        #      ~ (xj-x0) . 0.5*( A(xj) + A(x0) )
        #
        # The following computes the dot-products of all those
        # edges[i], mvp[i], and put the result in the cache.
        edges = self.mesh.node_coords[self.mesh.edges['nodes'][:, 1]] \
            - self.mesh.node_coords[self.mesh.edges['nodes'][:, 0]]
        mvp = 0.5 * (self._get_mvp(mu, self.mesh.edges['nodes'][:, 1]) +
                     self._get_mvp(mu, self.mesh.edges['nodes'][:, 0])
                     )
        return numpy.sum(edges * mvp, 1)

    def _get_mvp(self, mu, index):
        return mu * self._raw_magnetic_vector_potential[index]

    # def keo_smallest_eigenvalue_approximation(self):
    #     '''Returns
    #        <v,Av> / <v,v>
    #     with v = ones and A = KEO - Laplace.
    #     This is linear approximation for the smallest magnitude eigenvalue
    #     of KEO.
    #     '''
    #     num_nodes = len(self.mesh.nodes)

    #     # compute the FVM entities for the mesh
    #     if self._edge_lengths is None or self._coedge_edge_ratios is None:
    #         self._create_fvm_entities()

    #     k = 0
    #     sum = 0.0
    #     for element in self.mesh.cells:
    #         # loop over the edges
    #         l = 0
    #         for edge in element.edges:
    #             # -----------------------------------------------------------
    #             # Compute the integral
    #             #
    #             #    I = \int_{x0}^{xj} (xj-x0).A(x) dx
    #             #
    #             # numerically by the midpoint rule, i.e.,
    #             #
    #             #    I ~ |xj-x0| * (xj-x0) . A( 0.5*(xj+x0) ).
    #             #
    #             node0 = self.mesh.nodes[edge[0]]
    #             node1 = self.mesh.nodes[edge[1]]
    #             midpoint = 0.5 * (node0 + node1)

    #             # Instead of projecting onto the normalized edge and then
    #             # multiplying with the edge length for the approximation of
    #             # the integral, just project on the not normalized edge.
    #             a_integral = numpy.dot(
    #                 node1 - node0,
    #                 self._magnetic_vector_potential(midpoint)
    #                 )

    #             # sum it in
    #             sum += 2.0 * self._coedge_edge_ratios[k][l] * \
    #                 (1.0 - math.cos(a_integral))
    #             l += 1
    #         k += 1

    #     return sum / len(self.mesh.nodes)
