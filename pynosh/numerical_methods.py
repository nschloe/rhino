#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Collection of numerical algorithms.
'''
import numpy
import krypy


class ForcingConstant(object):
    def __init__(self, eta0):
        self.eta0 = eta0
        return

    def get(self, eta_previous, resval_previous, F0, F_1):
        return self.eta0


class Forcing_EW1(object):
    '''Linear tolerance is given by

    "Choosing the Forcing Terms in an Inexact Newton Method (1994)"
    -- Eisenstat, Walker
    http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.15.3196

    See also
    "NITSOL: A Newton Iterative Solver for Nonlinear Systems"
    http://epubs.siam.org/sisc/resource/1/sjoce3/v19/i1/p302_s1?isAuthorized=no
    '''
    def __init__(self, eta_min=1.0e-6, eta_max=1.0e-2):
        self.eta_min = eta_min
        self.eta_max = eta_max
        return

    def get(self, eta_previous, resval_previous, F0, F_1):
        from scipy.constants import golden
        # linear_relresvec[-1] \approx tol, so this could be replaced.
        eta = abs(F0 - resval_previous) / F_1
        eta = max(eta, eta_previous**golden, self.eta_min)
        eta = min(eta, self.eta_max)
        return eta


class Forcing_EW2(object):
    def __init__(self, eta_min=1.0e-6, eta_max=1.0e-2, alpha=1.5, gamma=0.9):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.alpha = alpha
        self.gamma = gamma
        return

    def get(self, eta_previous, resval_previous, F0, F_1):
        eta = self.gamma * (F0 / F_1)**self.alpha
        eta = max(eta, self.gamma * eta_previous**self.alpha, self.eta_min)
        eta = min(eta, self.eta_max)
        return eta


def newton(x0,
           model_evaluator,
           nonlinear_tol=1.0e-10,
           newton_maxiter=20,
           RecyclingSolver=krypy.recycling.RecyclingMinres,
           recycling_solver_kwargs=None,
           vector_factory_generator=None,
           compute_f_extra_args={},
           eta0=1.0e-10,
           forcing_term='constant',
           debug=False,
           yaml_emitter=None
           ):
    '''Newton's method with different forcing terms.
    '''

    # Default forcing term.
    if forcing_term == 'constant':
        forcing_term = ForcingConstant(eta0)

    if recycling_solver_kwargs is None:
        recycling_solver_kwargs = {}

    # Some initializations.
    # Set the default error code to 'failure'.
    error_code = 1
    k = 0

    x = x0.copy()
    Fx = model_evaluator.compute_f(x, **compute_f_extra_args)
    Fx_norms = [numpy.sqrt(model_evaluator.inner_product(Fx, Fx))]
    eta_previous = None
    linear_relresvecs = []

    # get recycling solver
    recycling_solver = RecyclingSolver()

    # no solution in before first iteration if Newton
    out = None

    if debug:
        import yaml
        if yaml_emitter is None:
            yaml_emitter = yaml.YamlEmitter()
            yaml_emitter.begin_doc()
        yaml_emitter.begin_seq()

    while Fx_norms[-1] > nonlinear_tol and k < newton_maxiter:
        if debug:
            yaml_emitter.add_comment('Newton step %d' % (k+1))
            yaml_emitter.begin_map()
            yaml_emitter.add_key_value('Fx_norm', Fx_norms[-1][0][0])

        # Get tolerance for next linear solve.
        if k == 0:
            eta = eta0
        else:
            eta = forcing_term.get(eta_previous, out.resnorms[-1],
                                   Fx_norms[-1], Fx_norms[-2]
                                   )
        eta_previous = eta

        # Setup linear problem.
        jacobian = model_evaluator.get_jacobian(x, **compute_f_extra_args)

        M = model_evaluator.get_preconditioner(x, **compute_f_extra_args)
        Minv = \
            model_evaluator.get_preconditioner_inverse(x,
                                                       **compute_f_extra_args
                                                       )

        # get vector factory
        if vector_factory_generator is not None:
            vector_factory = vector_factory_generator(x)
        else:
            vector_factory = None

        # Create the linear system.
        linear_system = krypy.linsys.TimedLinearSystem(
            jacobian, -Fx, M=Minv, Minv=M, ip_B=model_evaluator.inner_product,
            normal=True, self_adjoint=True
            )

        out = recycling_solver.solve(linear_system,
                                     vector_factory,
                                     tol=eta,
                                     **recycling_solver_kwargs
                                     )

        if debug:
            yaml_emitter.add_key_value('relresvec', out.resnorms)
            #yaml_emitter.add_key_value('relresvec[-1]', out['relresvec'][-1])
            yaml_emitter.add_key_value('num_iter', len(out.resnorms)-1)
            yaml_emitter.add_key_value('eta', eta)

        # save the convergence history
        linear_relresvecs.append(out.resnorms)

        # perform the Newton update
        x += out.xk

        # do the household
        k += 1
        Fx = model_evaluator.compute_f(x, **compute_f_extra_args)
        Fx_norms.append(numpy.sqrt(model_evaluator.inner_product(Fx, Fx)))

        # run garbage collector in order to prevent MemoryErrors from being
        # raised
        import gc
        gc.collect()

        if debug:
            yaml_emitter.end_map()

    if Fx_norms[-1] < nonlinear_tol:
        error_code = 0

    if debug:
        yaml_emitter.begin_map()
        yaml_emitter.add_key_value('Fx_norm', Fx_norms[-1])
        yaml_emitter.end_map()
        yaml_emitter.end_seq()
        if Fx_norms[-1] > nonlinear_tol:
            yaml_emitter.add_comment(
                'Newton solver did not converge '
                '(residual = %g > %g = tol)' % (Fx_norms[-1], nonlinear_tol)
                )

    return {'x': x,
            'info': error_code,
            'Newton residuals': Fx_norms,
            'linear relresvecs': linear_relresvecs,
            'recycling_solver': recycling_solver
            }


def poor_mans_continuation(x0,
                           model_evaluator,
                           initial_parameter_value,
                           initial_step_size=1.0e-2,
                           minimal_step_size=1.0e-6,
                           maximum_step_size=1.0e-1,
                           max_steps=1000,
                           nonlinear_tol=1.0e-10,
                           max_newton_iters=5,
                           adaptivity_aggressiveness=1.0
                           ):
    '''Poor man's parameter continuation. With adaptive step size.
    If the previous step was unsuccessful, the step size is cut in half,
    but if the step was successful this strategy increases the step size based
    on the number of nonlinear solver iterations required in the previous step.
    In particular, the new step size :math:`\Delta s_{new}` is given by

    .. math::
       \Delta s_{new} = \Delta s_{old}\left(1 + a\left(
         \\frac{N_{max} - N}{N_{max}}\\right)^2\\right).
    '''

    # write header of the statistics file
    stats_file = open('continuationData.dat', 'w')
    stats_file.write('# step    parameter     norm            Newton iters\n')
    stats_file.flush()

    parameter_value = initial_parameter_value
    x = x0

    current_step_size = initial_step_size

    for k in range(max_steps):
        print('Continuation step %d (parameter=%e)...' % (k, parameter_value))

        # Try to converge to a solution and adapt the step size.
        converged = False
        while current_step_size > minimal_step_size:
            x_new, error_code, iters = newton(x,
                                              model_evaluator,
                                              nonlinear_tol=nonlinear_tol,
                                              max_iters=max_newton_iters
                                              )
            if error_code != 0:
                current_step_size *= 0.5
                print(('Continuation step failed (error code %d). '
                       'Setting step size to %e.')
                      % (error_code, current_step_size)
                      )
            else:
                current_step_size *= 1.0 + adaptivity_aggressiveness * \
                    (float(max_newton_iters-iters)/max_newton_iters)**2
                converged = True
                x = x_new
                print('Continuation step success!')
                break

        if not converged:
            print('Could not find a solution although '
                  'the step size was %e. Abort.' % current_step_size
                  )
            break

        stats_file.write('  %4d    %.5e   %.5e    %d\n' %
                         (k, parameter_value, model_evaluator.energy(x), iters)
                         )
        stats_file.flush()
        #model_evaluator.write( x, "step" + str(k) + ".vtu" )

        parameter_value += current_step_size
        model_evaluator.set_parameter(parameter_value)

    stats_file.close()

    print('done.')
    return
