import mesh.mesh_io
import numpy as np
import unittest
# ==============================================================================
class TestMesh(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, pyginlamesh, actual_values ):
        # Compute the control volumes.
        if pyginlamesh.control_volumes is None:
            pyginlamesh.compute_control_volumes()

        tol = 1.0e-12

        # Get their norms
        norm = np.linalg.norm( pyginlamesh.control_volumes, ord=1 )
        self.assertAlmostEqual( actual_values[0], norm, delta=tol )
        norm = np.linalg.norm( pyginlamesh.control_volumes, ord=2 )
        self.assertAlmostEqual( actual_values[1], norm, delta=tol )
        norm = np.linalg.norm( pyginlamesh.control_volumes, ord=np.Inf)
        self.assertAlmostEqual( actual_values[2], norm, delta=tol )

        return
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )
        actual_values = [ 10.0,
                          5.0,
                          2.5 ]
        self._run_test(pyginlamesh, actual_values )
        return
    # --------------------------------------------------------------------------
    def test_arrow3d(self):
        num_nodes = 5
        nodes = np.array([[0.0,  0.0, 0.0],
                          [2.0, -1.0, 0.0],
                          [2.0,  0.0, 0.0],
                          [2.0,  1.0, 0.0],
                          [0.5,  0.0, -1.0],
                          [0.5,  0.0, 1.0]])
        cellsNodes = np.array([[1,2,4,5],
                              [2,3,4,5],
                              [0,1,4,5],
                              [0,3,4,5]])
        from mesh.mesh3d import Mesh3D
        mymesh = Mesh3D(nodes, cellsNodes)

        #if mymesh.edgesNodes is None:
            #mymesh.create_adjacent_entities()
        #mymesh.show_edge(5)

        actual_values = [ 1.3333333333333333,
                          0.66386172963165968,
                          0.54166666666666652 ]
        self._run_test(mymesh, actual_values)
        return
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )
        actual_values = [ 302.52270072101,
                          15.3857579093391,
                          1.12779746704366 ]
        self._run_test(pyginlamesh, actual_values )
        return
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )
        actual_values = [ 10.0,
                          3.53553390593274,
                          1.25 ]
        self._run_test(pyginlamesh, actual_values)
        return
    # --------------------------------------------------------------------------
    def test_brick(self):
        filename = 'brick-w-hole.e'
        pyginlamesh, psi, A, field_data = mesh.mesh_io.read_mesh( filename )
        actual_values = [ 388.68629169464117,
                          16.661401941985677,
                          1.4684734547497671 ]
        self._run_test(pyginlamesh, actual_values)
        return
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
