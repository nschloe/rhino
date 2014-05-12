#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of pynosh.
#
#  pynosh is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  pynosh is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with pynosh.  If not, see <http://www.gnu.org/licenses/>.
'''Solve the Ginzburg--Landau equation.
'''

import pynosh.numerical_methods as nm
import pynosh.modelevaluator_nls as gm
import mesh.mesh_io
import numpy as np
import matplotlib.pyplot as pp
#import matplotlib2tikz


def _main():
    '''Main function.
    '''
    args = _parse_input_arguments()

    # read the mesh
    print 'Reading the mesh...',
    pynoshmesh, psi, A, field_data = mesh.mesh_io.read_mesh(args.filename)
    print 'done.'

    # build the model evaluator
    mu = 8.0e-2
    ginla_modelval = gm.ModelEvaluator(pynoshmesh, A, mu)

    # initial guess
    num_nodes = len(pynoshmesh.nodes)
    psi0 = np.ones((num_nodes, 1),
                   dtype=complex
                   )

    nums_deflation_vectors = range(50)
    last_resvecs = []
    for num_deflation_vectors in nums_deflation_vectors:
        print('Performing Newton iteration with %d deflation vectors...'
              % num_deflation_vectors)
        # perform newton iteration
        newton_out = nm.newton(psi0,
                               ginla_modelval,
                               linear_solver=nm.minres,
                               nonlinear_tol=1.0e-10,
                               eta0=1.0e-13,
                               forcing_term='constant',
                               use_preconditioner=True,
                               deflate_ix=True,
                               num_deflation_vectors=num_deflation_vectors
                               )
        print ' done.'
        assert newton_out[1] == 0, 'Newton did not converge.'

        last_resvecs.append(newton_out[3][-1])

    multiplot_data_series(last_resvecs)
    pp.title('Residual curves for the last Newton step for %s. Darker=More deflation vectors.'
             % args.filename
             )
    pp.show()
    #matplotlib2tikz.save('w-defl.tex')
    return


def multiplot_data_series(list_of_data_vectors):
    '''Plot a list of data vectors with increasing black value.
    '''
    num_plots = len(list_of_data_vectors)
    for k, relresvec in enumerate(list_of_data_vectors):
        pp.semilogy(relresvec, color=str(1.0 - float(k+1)/num_plots))
    pp.xlabel('MINRES step')
    pp.ylabel('||r||/||b||')
    return


def _parse_input_arguments():
    '''Parse input arguments.
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Find solutions to the Ginzburg--Landau equation.')
    parser.add_argument('filename',
                        metavar='FILE',
                        type=str,
                        help='ExodusII file containing the geometry and initial state'
                        )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _main()
