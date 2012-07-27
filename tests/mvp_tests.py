import voropy
import pynosh.magnetic_vector_potentials as mvp
import numpy as np
import unittest
from scipy.sparse import spdiags
# ==============================================================================
class TestF(unittest.TestCase):
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _run_test(self, filename, control_values):
        '''Test $\int_{\Omega} A^2$.'''

        # read the mesh
        mesh, point_data, field_data = voropy.read( filename )
        if mesh.control_volumes is None:
            mesh.compute_control_volumes()

        tol = 1.0e-13

        A = mvp.constant_field(mesh.node_coords, np.array([0,0,1]))
        integral = np.sum(mesh.control_volumes * np.sum(A**2, axis=1))
        self.assertAlmostEqual(control_values['z'], integral, delta = tol)

        # If this is a 2D mesh, append the z-component 0 to each node
        # to make sure that the magnetic vector potentials can be
        # calculated.
        points = mesh.node_coords.copy()
        if points.shape[1] == 2:
            points = np.column_stack((points, np.zeros(len(points))))

        A = mvp.magnetic_dipole(points,
                                x0 = np.array([0,0,10]),
                                m = np.array([0,0,1])
                                )
        integral = np.sum(mesh.control_volumes * np.sum(A**2, axis=1))
        self.assertAlmostEqual(control_values['dipole'], integral, delta = tol)


        #import time
        #start = time.time()
        A = mvp.magnetic_dot(mesh.node_coords, 2.0, [10.0, 11.0])
        #A = np.empty((len(points), 3), dtype=float)
        #for k, node in enumerate(points):
            #A[k] = mvp.magnetic_dot(node[0], node[1], 2.0, 10.0, 11.0)
        #end = time.time()
        #print end-start
        integral = np.sum(mesh.control_volumes * np.sum(A**2, axis=1))
        self.assertAlmostEqual(control_values['dot'], integral, delta = tol)

        return
    # --------------------------------------------------------------------------
    def test_rectanglesmall(self):
        filename = 'rectanglesmall.e'
        control_values = {'z': 63.125,
                          'dipole': 0.00012850741240854054,
                          'dot': 0.015062118041804408
                          }
        self._run_test(filename, control_values)
        return
    # --------------------------------------------------------------------------
    def test_pacman(self):
        filename = 'pacman.e'
        control_values = {'z': 3730.2268660993054,
                          'dipole': 0.0037630906971841487,
                          'dot': 0.46680832033437036
                          }
        self._run_test(filename, control_values)
        return
    # --------------------------------------------------------------------------
    def test_cubesmall(self):
        filename = 'cubesmall.e'
        control_values = {'z': 1.25,
                          'dipole': 0.00015098959555300608,
                          'dot': 0.00052723843169109191
                          }
        self._run_test(filename, control_values)
        return
    # --------------------------------------------------------------------------
    def test_brick(self):
        filename = 'brick-w-hole.e'
        control_values = {'z': 1687.6928071551067,
                          'dipole': 0.014339810567783946,
                          'dot': 0.4275090788990229
                          }
        self._run_test(filename, control_values)
        return
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
