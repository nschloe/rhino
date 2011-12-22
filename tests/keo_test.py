import mesh_io
import ginla_modelevaluator
import numpy as np
import unittest
# ==============================================================================
class TestKeo(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, mu, actual_control_sum_real):
        # read the mesh
        mesh, psi, A, field_data = mesh_io.read_mesh( filename )

        # build the model evaluator
        self.ginla_modeleval = \
            ginla_modelevaluator.GinlaModelEvaluator(mesh, A, mu)

        # Assemble the KEO.
        self.ginla_modeleval._assemble_keo()

        tol = 1.0e-14

        # Check that the matrix is Hermitian.
        A = self.ginla_modeleval._keo \
          - self.ginla_modeleval._keo.transpose().conjugate()
        self.assertAlmostEqual( 0.0,
                                A.sum(),
                                delta=tol )

        # Check the matrix sum.
        self.assertAlmostEqual( actual_control_sum_real,
                                self.ginla_modeleval._keo.sum(),
                                delta=tol )
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        actual_control_sum_real = 0.00631217123080605
        self._run_test(filename, mu, actual_control_sum_real)
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        actual_control_sum_real = 0.370442644715617
        self._run_test(filename, mu, actual_control_sum_real)
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        actual_control_sum_real = 0.0042221425209367988
        self._run_test(filename, mu, actual_control_sum_real)
# ==============================================================================
class TestJacobian(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, mu, actual_values ):
        # read the mesh
        mesh, psi, A, field_data = mesh_io.read_mesh( filename )

        # build the model evaluator
        self.ginla_modeleval = \
            ginla_modelevaluator.GinlaModelEvaluator(mesh, A, mu)

        # Set current psi.
        self.ginla_modeleval.set_current_psi( psi )

        tol = 1.0e-13

        # [1+i, 1+i, 1+i, ... ]
        phi = (1+1j) * np.ones( len(psi), dtype=complex )
        Jphi = self.ginla_modeleval.apply_jacobian( phi )
        val = np.vdot( phi, Jphi ).real
        self.assertAlmostEqual( actual_values[0], val, delta=tol )

        # [1, 1, 1, ... ]
        phi = np.ones( len(psi), dtype=complex )
        Jphi = self.ginla_modeleval.apply_jacobian( phi )
        val = np.vdot( phi, Jphi ).real
        self.assertAlmostEqual( actual_values[1], val, delta=tol )

        # [i, i, i, ... ]
        phi = 1j * np.ones( len(psi), dtype=complex )
        Jphi = self.ginla_modeleval.apply_jacobian( phi )
        val = np.vdot( phi, Jphi ).real
        self.assertAlmostEqual( actual_values[2], val, delta=tol )
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        mu = 1.0e-2
        actual_values = [ -20.0126243424616,
                          -20.0063121712308,
                          -0.00631217123080606 ]
        self._run_test(filename, mu, actual_values )
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        mu = 1.0e-2
        actual_values = [ -605.786286731452,
                          -605.415844086736,
                          -0.370442644715631 ]
        self._run_test(filename, mu, actual_values )
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        mu = 1.0e-2
        actual_values = [ -20.0084442850419,
                          -20.0042221425209,
                          -0.00422214252093753 ]
        self._run_test(filename, mu, actual_values )
# ==============================================================================
class TestMesh(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, actual_values ):
        # read the mesh
        mesh, psi, A, field_data = mesh_io.read_mesh( filename )

        # build the model evaluator
        mu = 0.0
        self.ginla_modeleval = \
            ginla_modelevaluator.GinlaModelEvaluator(mesh, A, mu)

        # Compute the control volumes.
        self.ginla_modeleval._compute_control_volumes()

        tol = 1.0e-15

        # Get their norms
        norm = np.linalg.norm( self.ginla_modeleval.control_volumes, ord=1 )
        self.assertAlmostEqual( actual_values[0], norm, delta=tol )
        norm = np.linalg.norm( self.ginla_modeleval.control_volumes, ord=2 )
        self.assertAlmostEqual( actual_values[1], norm, delta=tol )
        norm = np.linalg.norm( self.ginla_modeleval.control_volumes, ord=np.Inf)
        self.assertAlmostEqual( actual_values[2], norm, delta=tol )
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        actual_values = [ 10.0,
                          5.0,
                          2.5 ]
        self._run_test(filename, actual_values )
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        actual_values = [ 302.52270072101,
                          15.3857579093391,
                          1.12779746704366 ]
        self._run_test(filename, actual_values )
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        actual_values = [ 1000.0,
                          6.23840106458169,
                          0.0410021145682357 ]
        self._run_test(filename, actual_values )
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
