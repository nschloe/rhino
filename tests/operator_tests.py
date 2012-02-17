import mesh.mesh_io
import pyginla.ginla_modelevaluator as gm
import numpy as np
import unittest
from scipy.sparse import spdiags
# ==============================================================================
class TestKeo(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, mu, actual_control_sum_real):
        # read the mesh
        pyginla_mesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )

        # build the model evaluator
        ginla_modeleval = gm.GinlaModelEvaluator(pyginla_mesh, A, mu)

        # Assemble the KEO.
        ginla_modeleval._assemble_keo()

        tol = 1.0e-14

        # Check that the matrix is Hermitian.
        KK = ginla_modeleval._keo  - ginla_modeleval._keo.H
        self.assertAlmostEqual( 0.0,
                                KK.sum(),
                                delta=tol )

        # Check the matrix sum.
        self.assertAlmostEqual( actual_control_sum_real,
                                ginla_modeleval._keo.sum(),
                                delta=tol )
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
# ==============================================================================
class TestJacobian(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, mu, actual_values ):
        # read the mesh
        pyginla_mesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )

        # build the model evaluator
        ginla_modeleval = gm.GinlaModelEvaluator(pyginla_mesh, A, mu)

        # Get the Jacobian
        J = ginla_modeleval.get_jacobian( psi )

        tol = 1.0e-12

        n = len( ginla_modeleval.control_volumes )
        D = spdiags(ginla_modeleval.control_volumes.T, [0], n, n)

        # [1+i, 1+i, 1+i, ... ]
        phi = (1+1j) * np.ones((len(psi),1), dtype=complex)

        val = np.vdot( phi, D*(J*phi) ).real
        self.assertAlmostEqual( actual_values[0], val, delta=tol )

        # [1, 1, 1, ... ]
        phi = np.ones((len(psi),1), dtype=complex)
        val = np.vdot( phi, D*(J*phi) ).real
        self.assertAlmostEqual( actual_values[1], val, delta=tol )

        # [i, i, i, ... ]
        phi = 1j * np.ones((len(psi),1), dtype=complex)
        val = np.vdot( phi, D*(J*phi) ).real
        self.assertAlmostEqual( actual_values[2], val, delta=tol )
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
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
