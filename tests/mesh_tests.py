import mesh.mesh_io
import pyginla.ginla_modelevaluator as gm
import numpy as np
import unittest
# ==============================================================================
class TestMesh(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, actual_values ):
        # read the mesh
        pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )

        # build the model evaluator
        mu = 0.0
        self.ginla_modeleval = \
            gm.GinlaModelEvaluator(pyginlamesh, A, mu)

        # Compute the control volumes.
        self.ginla_modeleval._compute_control_volumes()

        tol = 1.0e-12

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
        actual_values = [ 10.0,
                          3.53553390593274,
                          1.25 ]
        self._run_test(filename, actual_values)
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
