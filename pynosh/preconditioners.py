#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Preconditioners for the Jacobian of the Ginzburg--Landau problem.
'''
from scipy import sparse
from scipy.sparse.linalg import splu, spilu
import pynosh.numerical_methods as nm


class Preconditioners(object):
    '''
    Ginzburg--Landau model evaluator class.
    '''
    def __init__(self, model_evaluator):
        '''Initialization.
        '''
        self._modeleval = model_evaluator
        self._keo_lu = None
        self._keo_ilu = None
        self._keo_ilu_droptol = None
        self._keo_symmetric_ilu = None
        self._keo_symmetric_ilu_droptol = None
        self._keoai_lu = None
        self._keo_amg_solver = None
        return

    def diagonal(self, psi):
        '''The equivalent of a diagonal preconditioner.
        Solves the equation system with

            self._keo.diagonal() * x \
            - ( 1.0 - 2.0*abs(current_psi)**2 ) * x \
            + current_psi**2 * x.conjugate()
            = psi.
        '''
        if self._modeleval._keo is None:
            self._modeleval._assemble_kinetic_energy_operator()

        # Mind that all operations below are executed elementwise.
        a = self._modeleval._keo.diagonal() \
            - (1.0-self._modeleval._temperature -
               2.0*abs(self._modeleval._psi)**2
               )
        b = self._modeleval._psi**2

        # One needs to solve
        #    a*z + b z.conj = psi
        # for each "diagonal" entry z. The solution is
        #   z = ( a.conj * psi - b * psi.conj ) / ( |a|^2 - |b|^2 )
        alpha = abs(a)**2 - abs(b)**2
        assert((abs(alpha) > 1.0e-10).all)
        return (a.conjugate() * psi - b * psi.conjugate()) / alpha

    def keo_cgapprox(self, psi):
        '''Solves a system with the kinetic energy operator only using
        ordinary CG.
        '''
        if self._modeleval._keo is None:
            self._modeleval._assemble_kinetic_energy_operator()

        sol, info = nm.cg_wrap(self._modeleval._keo,
                               psi,
                               x0=None,
                               inner_product=self._modeleval.inner_product,
                               tol=1.0e-10,
                               M=None,
                               callback=None
                               )
        # make sure it could be solved
        assert info == 0
        return sol

    def keo_amg(self, psi):
        '''Algebraic multigrid solve.
        '''
        import pyamg
        if self._keo_amg_solver is None:
            if self._modeleval._keo is None:
                self._modeleval._assemble_keo()
            self._keo_amg_solver = \
                pyamg.smoothed_aggregation_solver(self._modeleval._keo)
        return self._keo_amg_solver.solve(psi,
                                          tol=1e-12,
                                          accel=None
                                          )

    def keo_lu(self, psi):
        '''Solves a system with the kinetic energy operator only via ordinary
        CG.
        '''
        if self._modeleval._keo is None:
            self._modeleval._assemble_kinetic_energy_operator()

        # From http://crd.lbl.gov/~xiaoye/SuperLU/faq.html#sym-problem:
        # SuperLU cannot take advantage of symmetry, but it can still solve the
        # linear system as long as you input both the lower and upper parts of
        # the matrix A. If off-diagonal pivoting does not occur, the U matrix
        # in A = L*U is equivalent to D*L'.
        # In many applications, matrix A may be diagonally dominant or nearly
        # so. In this case, pivoting on the diagonal is sufficient for
        # stability and is preferable for sparsity to off-diagonal pivoting. To
        # do this, the user can set a small (less-than-one) diagonal pivot
        # threshold (e.g., 0.0, 0.01, ...) and choose an (A' + A)-based column
        # permutation algorithm. We call this setting Symmetric  Mode. To use
        # this (in serial SuperLU), you need to set:
        #
        #    options.SymmetricMode = YES;
        #    options.ColPerm = MMD_AT_PLUS_A;
        #    options.DiagPivotThresh = 0.001; /* or 0.0, 0.01, etc. */
        #
        if self._keo_lu is None:
            self._keo_lu = splu(self._modeleval._keo,
                                options={'SymmetricMode': True},
                                permc_spec='MMD_AT_PLUS_A',  # minimum deg
                                diag_pivot_thresh=0.0
                                )
        return self._keo_lu.solve(psi)

    def keo_symmetric_ilu2(self, psi):
        return self.keo_symmetric_ilu(psi, 1.0e-2)

    def keo_symmetric_ilu4(self, psi):
        return self.keo_symmetric_ilu(psi, 1.0e-4)

    def keo_symmetric_ilu6(self, psi):
        return self.keo_symmetric_ilu(psi, 1.0e-6)

    def keo_symmetric_ilu8(self, psi):
        return self.keo_symmetric_ilu(psi, 1.0e-8)

    def keo_symmetric_ilu(self, psi, droptol):
        if self._modeleval._keo is None:
            self._modeleval._assemble_kinetic_energy_operator()
        if self._keo_symmetric_ilu is None \
           or self._keo_symmetric_ilu_droptol is None \
           or droptol != self._keo_symmetric_ilu_droptol:
            self._keo_symmetric_ilu = spilu(self._modeleval._keo,
                                            drop_tol=droptol,
                                            fill_factor=10,
                                            drop_rule=None,
                                            # see remark above for splu
                                            options={'SymmetricMode': True},
                                            permc_spec='MMD_AT_PLUS_A',
                                            diag_pivot_thresh=0.0,
                                            relax=None,
                                            panel_size=None
                                            )
            self._keo_symmetric_ilu_droptol = droptol
        return self._keo_symmetric_ilu.solve(psi)

    def keo_ilu4(self, psi):
        return self.keo_ilu(psi, 1.0e-4)

    def keo_ilu6(self, psi):
        return self.keo_ilu(psi, 1.0e-6)

    def keo_ilu(self, psi, droptol):
        if self._modeleval._keo is None:
            self._modeleval._assemble_kinetic_energy_operator()
        if self._keo_ilu is None \
           or self._keo_ilu_droptol is None \
           or droptol != self._keo_ilu_droptol:
            self._keo_ilu = spilu(self._modeleval._keo,
                                  drop_tol=droptol,
                                  fill_factor=10,
                                  drop_rule=None,
                                  relax=None,
                                  panel_size=None
                                  )
            self._keo_ilu_droptol = droptol
        return self._keo_ilu.solve(psi)

    def keoi(self, psi):
        if self._modeleval._keo is None:
            self._modeleval._assemble_kinetic_energy_operator()
        if self._keoai_lu is None:
            self._keoai_lu = splu(
                self._modeleval._keo + 1.0e-2 * sparse.identity(len(psi))
                )
        return self._keoai_lu.solve(psi)

    def set_parameter(self, mu):
        '''This function is actually somewhat funny in that it is not used to
        actually set mu (which lives in the model evaluator anyway), but to
        make sure that all the corresponding objects (the factorizations) are
        recreated next time they are needed.
        '''
        self._modeleval.set_parameter(mu)
        self._keo_lu = None
        self._keo_ilu = None
        self._keo_ilu_droptol = None
        self._keo_symmetric_ilu = None
        self._keo_symmetric_ilu_droptol = None
        self._keoai_lu = None
        self._keo_amg_solver = None
        return

    def set_mesh(self, mesh):
        '''This function is actually somewhat funny in that it is not used to
        actually set the mesh (which lives in the model evaluator anyway), but
        to make sure that all the corresponding objects (the factorizations)
        are recreated next time they are needed.
        '''
        self._modeleval.set_mesh(mesh)
        self._keo_lu = None
        self._keo_ilu = None
        self._keo_ilu_droptol = None
        self._keo_symmetric_ilu = None
        self._keo_symmetric_ilu_droptol = None
        self._keoai_lu = None
        self._keo_amg_solver = None
        return
