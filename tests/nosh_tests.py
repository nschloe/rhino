import voropy
import pynosh.modelevaluator_nls as nls
import numpy as np
import unittest


class TestF(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, filename, mu, control_values):
        # read the mesh
        mesh, point_data, field_data = voropy.reader.read(filename)

        # build the model evaluator
        modeleval = nls.NlsModelEvaluator(mesh,
                                          V=point_data['V'],
                                          A=point_data['A']
                                          )

        # compute the ginzburg-landau residual
        psi = point_data['psi'][:, 0] + 1j * point_data['psi'][:, 1]
        r = modeleval.compute_f(psi, mu, 1.0)

        # scale with D for compliance with the Nosh (C++) tests
        if mesh.control_volumes is None:
            mesh.compute_control_volumes()
        r *= mesh.control_volumes.reshape(r.shape)

        tol = 1.0e-13
        # For C++ Nosh compatibility:
        # Compute 1-norm of vector (Re(psi[0]), Im(psi[0]), Re(psi[1]), ... )
        alpha = np.linalg.norm(r.real, ord=1) \
            + np.linalg.norm(r.imag, ord=1)
        self.assertAlmostEqual(control_values['one'],
                               alpha,
                               delta=tol
                               )
        self.assertAlmostEqual(control_values['two'],
                               np.linalg.norm(r, ord=2),
                               delta=tol
                               )
        # For C++ Nosh compatibility:
        # Compute inf-norm of vector (Re(psi[0]), Im(psi[0]), Re(psi[1]), ... )
        alpha = max(np.linalg.norm(r.real, ord=np.inf),
                    np.linalg.norm(r.imag, ord=np.inf))
        self.assertAlmostEqual(control_values['inf'],
                               alpha,
                               delta=tol
                               )
        return

    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        control_values = {'one': 0.50126061034211067,
                          'two': 0.24749434381636057,
                          'inf': 0.12373710977782607
                          }
        self._run_test(filename, mu, control_values)
        return

    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        control_values = {'one': 0.71366475047893463,
                          'two': 0.12552206259336218,
                          'inf': 0.055859319123267033
                          }
        self._run_test(filename, mu, control_values)
        return

    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        control_values = {'one': 8.3541623156163313e-05,
                          'two': 2.9536515963905867e-05,
                          'inf': 1.0468744547749431e-05
                          }
        self._run_test(filename, mu, control_values)
        return

    def test_brick(self):
        filename = 'brick-w-hole.e'
        mu = 1.0e-2
        control_values = {'one': 1.8084716102419285,
                          'two': 0.15654267585120338,
                          'inf': 0.03074423493622647
                          }
        self._run_test(filename, mu, control_values)
        return


class TestKeo(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, filename, mu, control_values):
        # read the mesh
        mesh, point_data, field_data = voropy.reader.read(filename)

        # build the model evaluator
        modeleval = nls.NlsModelEvaluator(mesh,
                                          V=point_data['V'],
                                          A=point_data['A']
                                          )

        # Assemble the KEO.
        keo = modeleval._get_keo(mu)

        tol = 1.0e-13

        # Check that the matrix is Hermitian.
        KK = keo - keo.H
        self.assertAlmostEqual(0.0,
                               KK.sum(),
                               delta=tol
                               )

        # Check the matrix sum.
        self.assertAlmostEqual(control_values['sum real'],
                               keo.sum(),
                               delta=tol
                               )

        # Check the 1-norm of the matrix |Re(K)| + |Im(K)|.
        # This equals the 1-norm of the matrix defined by the block
        # structure
        #   Re(K) -Im(K)
        #   Im(K)  Re(K).
        K = abs(keo.real) + abs(keo.imag)
        self.assertAlmostEqual(control_values['norm 1'],
                               np.max(K.sum(0)),
                               delta=tol
                               )
        return

    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        control_values = {'sum real': 0.0063121712308067401,
                          'norm 1': 10.224658806561596
                          }
        self._run_test(filename, mu, control_values)

    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        control_values = {'sum real': 0.37044264296585938,
                          'norm 1': 10.000520856079092
                          }
        self._run_test(filename, mu, control_values)

    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        control_values = {'sum real': 8.3541623155714007e-05,
                          'norm 1': 10.058364522531498
                          }
        self._run_test(filename, mu, control_values)

    def test_brick(self):
        filename = 'brick-w-hole.e'
        mu = 1.0e-2
        control_values = {'sum real': 0.16763276012920181,
                          'norm 1': 15.131119904340618
                          }
        self._run_test(filename, mu, control_values)


class TestJacobian(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, filename, mu, actual_values):
        # read the mesh
        mesh, point_data, field_data = voropy.reader.read(filename)
        psi = point_data['psi'][:, 0] \
            + 1j * point_data['psi'][:, 1]
        num_unknowns = len(psi)
        psi = psi.reshape(num_unknowns, 1)

        # build the model evaluator
        modeleval = nls.NlsModelEvaluator(mesh,
                                          V=point_data['V'],
                                          A=point_data['A']
                                          )

        # Get the Jacobian
        J = modeleval.get_jacobian(psi, mu, 1.0)

        tol = 1.0e-12

        # [1+i, 1+i, 1+i, ... ]
        phi = (1 + 1j) * np.ones(num_unknowns, dtype=complex)
        val = np.vdot(phi, mesh.control_volumes.reshape(phi.shape) * (J*phi)) \
            .real
        self.assertAlmostEqual(actual_values[0], val, delta=tol)

        # [1, 1, 1, ... ]
        phi = np.ones((num_unknowns, 1), dtype=complex)
        val = np.vdot(phi, mesh.control_volumes[:, None] * (J*phi)).real
        self.assertAlmostEqual(actual_values[1], val, delta=tol)

        # [i, i, i, ... ]
        phi = 1j * np.ones((num_unknowns, 1), dtype=complex)
        val = np.vdot(phi, mesh.control_volumes[:, None] * (J*phi)).real
        self.assertAlmostEqual(actual_values[2], val, delta=tol)
        return

    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        actual_values = [20.0126243424616,
                         20.0063121712308,
                         0.00631217123080606
                         ]
        self._run_test(filename, mu, actual_values)

    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        actual_values = [605.78628672795264,
                         605.41584408498682,
                         0.37044264296586299
                         ]
        self._run_test(filename, mu, actual_values)

    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        actual_values = [20.000167083246311,
                         20.000083541623155,
                         8.3541623155658495e-05
                         ]
        self._run_test(filename, mu, actual_values)

    def test_brick(self):
        filename = 'brick-w-hole.e'
        mu = 1.0e-2
        actual_values = [777.70784890954064,
                         777.54021614941144,
                         0.16763276012921419
                         ]
        self._run_test(filename, mu, actual_values)

    def test_tet(self):
        filename = 'tetrahedron.e'
        mu = 1.0e-2
        actual_values = [128.31647020288861,
                         128.3082636471523,
                         0.0082065557362998032
                         ]
        self._run_test(filename, mu, actual_values)

    def test_tetsmall(self):
        filename = 'tet.e'
        mu = 1.0e-2
        actual_values = [128.31899139655067,
                         128.30952517579789,
                         0.0094662207527960365
                         ]
        self._run_test(filename, mu, actual_values)


class TestInnerProduct(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, filename, control_values):
        # read the mesh
        mesh, point_data, field_data = voropy.reader.read(filename)

        # build the model evaluator
        modeleval = nls.NlsModelEvaluator(mesh,
                                          V=point_data['V'],
                                          A=point_data['A']
                                          )
        tol = 1.0e-12

        # For C++ Nosh compatibility:
        # Compute 1-norm of vector (Re(psi[0]), Im(psi[0]), Re(psi[1]), ... )
        N = len(mesh.node_coords)
        phi0 = 1.0 * np.ones((N, 1), dtype=complex)
        phi1 = 1.0 * np.ones((N, 1), dtype=complex)
        alpha = modeleval.inner_product(phi0, phi1)[0][0]
        self.assertAlmostEqual(control_values[0],
                               alpha,
                               delta=tol
                               )

        phi0 = np.empty((N, 1), dtype=complex)
        phi1 = np.empty((N, 1), dtype=complex)
        for k, node in enumerate(mesh.node_coords):
            phi0[k] = np.cos(np.pi * node[0]) + 1j * np.sin(np.pi * node[1])
            phi1[k] = np.sin(np.pi * node[0]) + 1j * np.cos(np.pi * node[1])
        alpha = modeleval.inner_product(phi0, phi1)[0][0]
        self.assertAlmostEqual(control_values[1],
                               alpha,
                               delta=tol
                               )

        phi0 = np.empty((N, 1), dtype=complex)
        phi1 = np.empty((N, 1), dtype=complex)
        for k, node in enumerate(mesh.node_coords):
            phi0[k] = np.dot(node, node)
            phi1[k] = np.exp(1j * np.dot(node, node))
        alpha = modeleval.inner_product(phi0, phi1)[0][0]
        self.assertAlmostEqual(control_values[2],
                               alpha,
                               delta=tol
                               )
        return

    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        control_values = [10.0,
                          0.0,
                          250.76609861896702
                          ]
        self._run_test(filename, control_values)
        return

    def test_pacman(self):
        filename = 'pacman.e'
        control_values = [302.52270072101049,
                          8.8458601556211267,
                          1261.5908800348018
                          ]
        self._run_test(filename, control_values)
        return

    def test_cubesmall(self):
        filename = 'cubesmall.e'
        control_values = [10.0,
                          0.0,
                          237.99535357630012
                          ]
        self._run_test(filename, control_values)
        return

    def test_brick(self):
        filename = 'brick-w-hole.e'
        control_values = [388.68629169464111,
                          30.434181122856277,
                          -24.459076553128803
                          ]
        self._run_test(filename, control_values)
        return

if __name__ == '__main__':
    unittest.main()
