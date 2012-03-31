import voropy
import pyginla.ginla_modelevaluator as gm
import numpy as np
import unittest
from scipy.sparse import spdiags
# ==============================================================================
class TestF(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, mu, control_values):
        # read the mesh
        mesh, point_data, field_data = voropy.read( filename )

        # build the model evaluator
        ginla_modeleval = gm.GinlaModelEvaluator(mesh, point_data['A'], mu)

        # compute the ginzburg-landau residual
        r = ginla_modeleval.compute_f(point_data['psi'])

        # scale with D for compliance with the Ginla (C++) tests
        if mesh.control_volumes is None:
            mesh.compute_control_volumes()
        r *= mesh.control_volumes[:,None]

        tol = 1.0e-13
        # For C++ Ginla compatibility:
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
        # For C++ Ginla compatibility:
        # Compute inf-norm of vector (Re(psi[0]), Im(psi[0]), Re(psi[1]), ... )
        alpha = max(np.linalg.norm(r.real, ord=np.inf),
                    np.linalg.norm(r.imag, ord=np.inf))
        self.assertAlmostEqual(control_values['inf'],
                               alpha,
                               delta=tol
                               )
        return
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        control_values = {'one': 0.50126061034211067,
                          'two': 0.24749434381636057,
                          'inf': 0.12373710977782607
                          }
        self._run_test(filename, mu, control_values)
        return
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        control_values = {'one': 0.713664749303348,
                          'two': 0.12552206461432219,
                          'inf': 0.055859321274632785
                          }
        self._run_test(filename, mu, control_values)
        return
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        control_values = {'one': 0.28999063035759653,
                          'two': 0.15062204533498347,
                          'inf': 0.095254500561777741
                          }
        self._run_test(filename, mu, control_values)
        return
    # --------------------------------------------------------------------------
    def test_brick(self):
        filename = 'brick-w-hole.e'
        mu = 1.0e-2
        control_values = {'one': 1.8084716162552725,
                          'two': 0.15654267591639234,
                          'inf': 0.030744236169795706
                          }
        self._run_test(filename, mu, control_values)
        return
# ==============================================================================
class TestKeo(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, mu, actual_control_sum_real):
        # read the mesh
        mesh, point_data, field_data = voropy.read( filename )

        # build the model evaluator
        ginla_modeleval = gm.GinlaModelEvaluator(mesh, point_data['A'], mu)

        # Assemble the KEO.
        ginla_modeleval._assemble_keo()

        tol = 1.0e-13

        # Check that the matrix is Hermitian.
        KK = ginla_modeleval._keo  - ginla_modeleval._keo.H
        self.assertAlmostEqual( 0.0,
                                KK.sum(),
                                delta=tol )

        # Check the matrix sum.
        self.assertAlmostEqual( actual_control_sum_real,
                                ginla_modeleval._keo.sum(),
                                delta=tol )
        return
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        actual_control_sum_real = 0.0063121712308067401
        self._run_test(filename, mu, actual_control_sum_real)
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        actual_control_sum_real = 0.37044264471562194
        self._run_test(filename, mu, actual_control_sum_real)
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        actual_control_sum_real = 0.0042221425209372221
        self._run_test(filename, mu, actual_control_sum_real)
    # --------------------------------------------------------------------------
    def test_brick(self):
        filename = 'brick-w-hole.e'
        mu = 1.0e-2
        actual_control_sum_real = 0.16763276011469475
        self._run_test(filename, mu, actual_control_sum_real)
# ==============================================================================
class TestJacobian(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, mu, actual_values ):
        # read the mesh
        mesh, point_data, field_data = voropy.read( filename )
        psi = point_data['psi'][:,0] \
            + 1j * point_data['psi'][:,1]
        num_unknowns = len(psi)
        psi = psi.reshape(num_unknowns,1)

        # build the model evaluator
        ginla_modeleval = gm.GinlaModelEvaluator(mesh, point_data['A'], mu)

        # Get the Jacobian
        J = ginla_modeleval.get_jacobian(psi)

        tol = 1.0e-12

        # [1+i, 1+i, 1+i, ... ]
        phi = (1+1j) * np.ones((num_unknowns,1), dtype=complex)
        val = np.vdot( phi, mesh.control_volumes[:,None] * (J*phi)).real
        self.assertAlmostEqual( actual_values[0], val, delta=tol )

        # [1, 1, 1, ... ]
        phi = np.ones((num_unknowns,1), dtype=complex)
        val = np.vdot( phi, mesh.control_volumes[:,None] * (J*phi)).real
        self.assertAlmostEqual( actual_values[1], val, delta=tol )

        # [i, i, i, ... ]
        phi = 1j * np.ones((num_unknowns,1), dtype=complex)
        val = np.vdot( phi, mesh.control_volumes[:,None] * (J*phi)).real
        self.assertAlmostEqual( actual_values[2], val, delta=tol )

        return
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        actual_values = [ -20.0126243424616,
                          -20.0063121712308,
                          -0.00631217123080606 ]
        self._run_test(filename, mu, actual_values)
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        actual_values = [ -605.786286731452,
                          -605.415844086736,
                          -0.370442644715631 ]
        self._run_test(filename, mu, actual_values)
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        actual_values = [ -20.0084442850419,
                          -20.0042221425209,
                          -0.00422214252093753 ]
        self._run_test(filename, mu, actual_values)
    # --------------------------------------------------------------------------
    def test_brick(self):
        filename = 'brick-w-hole.e'
        mu = 1.0e-2
        actual_values = [ -777.70784890951165,
                          -777.54021614939688,
                          -0.16763276011468597 ]
        self._run_test(filename, mu, actual_values)
    # --------------------------------------------------------------------------
    def test_tet(self):
        filename = 'tetrahedron.e'
        mu = 1.0e-2
        actual_values = [ -128.31647020294287,
                          -128.30826364717944,
                          -0.0082065557634346201 ]
        self._run_test(filename, mu, actual_values)
    # --------------------------------------------------------------------------
    def test_tetsmall(self):
        filename = 'tet.e'
        mu = 1.0e-2
        actual_values = [ -128.31899139647672,
                          -128.30952517576091,
                          -0.0094662207158164313 ]
        self._run_test(filename, mu, actual_values)
    # --------------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
