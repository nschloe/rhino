import mesh_io
import ginla_modelevaluator
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

        tol = 1.0e-13

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
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
