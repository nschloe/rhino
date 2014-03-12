#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Collection of numerical algorithms.
'''
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.sputils import upcast
import numpy as np
import warnings
import scipy

import krypy


def _apply(A, x):
    '''Implement A*x for different types of linear operators.'''
    if A is None:
        return x
    elif isinstance(A, np.ndarray):
        return np.dot(A, x)
    elif scipy.sparse.isspmatrix(A):
        return A * x
    elif isinstance(A, scipy.sparse.linalg.LinearOperator):
        return A * x
    else:
        raise ValueError('Unknown operator type "%s".' % type(A))


def _ipstd(X, Y):
    '''Euclidean inner product

    np.vdot only works for vectors and np.dot does not use the conjugate
    transpose. In Octave/MATLAB notation _ipstd(X,Y) == X'*Y.

    Arguments:
        X:  array of shape [N,m]
        Y:  array of shape [N,n]

    Returns:
        ip: array of shape [m,n] with X^H * Y
    '''
    return np.dot(X.T.conj(), Y)


def _norm_squared(x,
                  Mx=None,
                  inner_product=_ipstd
                  ):
    '''Compute the norm^2 w.r.t. to a given scalar product.
    '''
    assert(len(x.shape) == 2)
    assert(x.shape[1] == 1)
    if Mx is None:
        rho = inner_product(x, x)[0, 0]
    else:
        assert(len(Mx.shape) == 2)
        assert(Mx.shape[1] == 1)
        rho = inner_product(x, Mx)[0, 0]

    #    if rho.imag != 0.0: #abs(rho.imag) > abs(rho) * 1.0e-10:
    if abs(rho.imag) > abs(rho) * 1e-10:
        raise ValueError('M not positive definite?')

    rho = rho.real
    if rho < 0.0:
        raise ValueError('<x,Mx> = %g. M not positive definite?' % rho)
    return rho


def _norm(x,
          Mx=None,
          inner_product=_ipstd
          ):
    '''Compute the norm w.r.t. to a given scalar product.'''
    return np.sqrt(_norm_squared(x,
                                 Mx=Mx,
                                 inner_product=inner_product
                                 ))


def newton(x0,
           model_evaluator,
           nonlinear_tol=1.0e-10,
           newton_maxiter=20,
           linear_solver=krypy.linsys.Gmres,
           linear_solver_maxiter=None,
           linear_solver_extra_args={},
           compute_f_extra_args={},
           forcing_term='constant',
           eta0=1.0e-1,
           eta_min=1.0e-6,
           eta_max=1.0e-2,
           alpha=1.5,  # only used by forcing_term='type 2'
           gamma=0.9,  # only used by forcing_term='type 2'
           #deflation_generators=[],
           #num_deflation_vectors=0,
           debug=False,
           yaml_emitter=None
           ):
    '''Newton's method with different forcing terms.
    '''
    from scipy.constants import golden

    # Some initializations.
    # Set the default error code to 'failure'.
    error_code = 1
    k = 0

    x = x0.copy()
    Fx = model_evaluator.compute_f(x, **compute_f_extra_args)
    Fx_norms = [_norm(Fx, inner_product=model_evaluator.inner_product)]
    eta_previous = None
    #W = np.zeros((len(x), 0))
    linear_relresvecs = []

    # get recycling solver
    recycling_solver = krypy.recycling.RecyclingGmres()
    # get vector factory
    vector_factory = krypy.recycling.factories.RitzFactorySimple(
        n_vectors=3,
        which='smallest_res'
        )

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
            yaml_emitter.add_key_value('Fx_norm', Fx_norms[-1])
        # Linear tolerance is given by
        #
        # "Choosing the Forcing Terms in an Inexact Newton Method (1994)"
        # -- Eisenstat, Walker
        # http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.15.3196
        #
        # See also
        # "NITSOL: A Newton Iterative Solver for Nonlinear Systems"
        # http://epubs.siam.org/sisc/resource/1/sjoce3/v19/i1/p302_s1?isAuthorized=no
        if eta_previous is None or forcing_term == 'constant':
            eta = eta0
        elif forcing_term == 'type 1':
            # linear_relresvec[-1] \approx tol, so this could be replaced.
            eta = abs(Fx_norms[-1] - out.resvals[-1]) / Fx_norms[-2]
            eta = max(eta, eta_previous**golden, eta_min)
            eta = min(eta, eta_max)
        elif forcing_term == 'type 2':
            eta = gamma * (Fx_norms[-1] / Fx_norms[-2])**alpha
            eta = max(eta, gamma * eta_previous**alpha, eta_min)
            eta = min(eta, eta_max)
        else:
            raise ValueError('Unknown forcing term \'%s\'. Abort.'
                             % forcing_term
                             )
        eta_previous = eta

        # Setup linear problem.
        jacobian = model_evaluator.get_jacobian(x, **compute_f_extra_args)
        #initial_guess = np.zeros((len(x), 1))
        # The .copy() is redundant as Python copies on "-Fx" anyways,
        # but leave it here for clarity.
        rhs = -Fx.copy()

        M = model_evaluator.get_preconditioner(x, **compute_f_extra_args)
        Minv = \
            model_evaluator.get_preconditioner_inverse(x,
                                                       **compute_f_extra_args
                                                       )

        #def Minner_product(x, y):
        #    return model_evaluator.inner_product(_apply(M, x), y)

        ## Conditionally deflate the nearly-null vector i*x or others.
        ## TODO Get a more sensible guess for the dtype here.
        ##      If x is real-values, deflation_generator(x) could indeed
        ##      be complex-valued.
        ##      Maybe get the lambda function along with its dtype
        ##      as input argument?
        #U = np.empty((len(x), len(deflation_generators)), dtype=x.dtype)
        #for i, deflation_generator in enumerate(deflation_generators):
        #    U[:, [i]] = deflation_generator(x)

        ## Gather up all deflation vectors.
        #WW = np.c_[W, U]

        ## Brief sanity test for the deflation vectors.
        ## If one of them is (too close to) 0, then qr below will cowardly bail
        ## out, so remove them from the list.
        #del_k = []
        #for col_index, w in enumerate(WW.T):
        #    alpha = Minner_product(w[:, None], w[:, None])
        #    if abs(alpha) < 1.0e-14:
        #        warnings.warn('Deflation vector dropped due to low norm; '
        #                      '<v, v> = %g.' % alpha
        #                      )
        #        del_k.append(col_index)
        #if del_k:
        #    WW = np.delete(WW, del_k, 1)

        ## Attention:
        ## If the preconditioner is later solved inexactly then W will be
        ## orthonormal w.r.t. a different inner product! This may affect the
        ## computation of Ritz pairs and their residuals.
        #W, _ = krypy.utils.qr(WW, ip_B=Minner_product)

        #if W.shape[1] > 0:
        #    AW = jacobian * W
        #    P, x0new = krypy.deflation.get_deflation_data(
        #        W, AW, rhs,
        #        x0=initial_guess,
        #        inner_product=model_evaluator.inner_product
        #        )
        #    if debug:
        #        yaml_emitter.add_key_value('dim of deflation space',
        #                                   W.shape[1]
        #                                   )
        #        yaml_emitter.add_key_value(
        #            '||I-ip(W,W)||',
        #            np.linalg.norm(np.eye(W.shape[1]) - Minner_product(W, W))
        #            )
        #        ix_normalized = 1j*x / np.sqrt(Minner_product(1j*x, 1j*x))
        #        ixW = Minner_product(W, ix_normalized)
        #        from scipy.linalg import svd
        #        yaml_emitter.add_key_value(
        #            'principal angle',
        #            np.arccos(min(svd(ixW, compute_uv=False)[0], 1.0))
        #            )
        #else:
        #    AW = np.zeros((len(x), 0))

        #if num_deflation_vectors > 0:
        #    # limit to 1 GB memory for Vfull/Pfull (together)
        #    from math import floor
        #    maxmem = 2**30  # 1 GB
        #    maxmem_maxiter = int(floor(maxmem/(2*16*len(x))))
        #    if linear_solver_maxiter > maxmem_maxiter:
        #        warnings.warn('limiting linear_solver_maxiter to %d instead of %d to fulfill memory constraints (%g GB)' % (maxmem_maxiter, linear_solver_maxiter, maxmem / float(2**30)))
        #        linear_solver_maxiter = maxmem_maxiter

        # Create the linear system.
        # TODO check Minv, M
        linear_system = krypy.linsys.LinearSystem(
            jacobian, rhs, M=Minv, Minv=M, ip_B=model_evaluator.inner_product
            )

        out = recycling_solver.solve(linear_system,
                                     vector_factory,
                                     tol=eta,
                                     maxiter=linear_solver_maxiter,
                                     #**linear_solver_extra_args
                                     )

        ## Solve the linear system.
        #out = linear_solver(jacobian,
        #                    rhs,
        #                    x0new,
        #                    maxiter=linear_solver_maxiter,
        #                    Mr=P,
        #                    M=Minv,
        #                    tol=eta,
        #                    ip_B=model_evaluator.inner_product,
        #                    store_arnoldi=return_basis,
        #                    **linear_solver_extra_args
        #                    )

        if debug:
            yaml_emitter.add_key_value('relresvec', out.resnorms)
            #yaml_emitter.add_key_value('relresvec[-1]', out['relresvec'][-1])
            yaml_emitter.add_key_value('num_iter', len(out.resnorms)-1)
            yaml_emitter.add_key_value('eta', eta)
            #print 'Linear solver \'%s\' performed %d iterations with final residual %g (tol %g).' %(linear_solver.__name__, len(out['relresvec'])-1, out['relresvec'][-1], eta)

        #np.set_printoptions(linewidth=150)
        #if ('Vfull' in out) and ('Hfull' in out):
        #    if debug:
        #        MVfull = out['Pfull'] if ('Pfull' in out) else out['Vfull']
        #        yaml_emitter.add_key_value(
        #            '||ip(Vfull,W)||',
        #            np.linalg.norm(model_evaluator.inner_product(MVfull, W))
        #            )
        #        yaml_emitter.add_key_value(
        #            '||I-ip(Vfull,Vfull)||',
        #            np.linalg.norm(np.eye(out['Vfull'].shape[1])
        #                - model_evaluator.inner_product(MVfull, out['Vfull']))
        #            )
        #        # next one is time-consuming, uncomment if needed
        #        #print '||Minv*A*P*V - V_*H|| = %g' % \
        #        #    np.linalg.norm(_apply(Minv, _apply(jacobian, _apply(P, out['Vfull'][:,0:-1]))) - np.dot(out['Vfull'], out['Hfull']) )

        #    if num_deflation_vectors > 0:
        #        ritz_vals, W = get_p_harmonic_ritz(
        #            jacobian, W, AW, out['Vfull'], out['Hfull'],
        #            Minv=Minv,
        #            M=M,
        #            p=num_deflation_vectors,
        #            mode='SM',
        #            inner_product=model_evaluator.inner_product
        #            )
        #        if debug:
        #            yaml_emitter.add_key_value(
        #                '||I-ip(Wnew,Wnew)||',
        #                np.linalg.norm(np.eye(W.shape[1])-Minner_product(W, W))
        #                )
        #    else:
        #        W = np.zeros((len(x), 0))
        #else:
        #    W = np.zeros((len(x), 0))

        # save the convergence history
        linear_relresvecs.append(out.resnorms)

        # perform the Newton update
        x += out.xk

        # do the household
        k += 1
        Fx = model_evaluator.compute_f(x, **compute_f_extra_args)
        Fx_norms.append(_norm(Fx, inner_product=model_evaluator.inner_product))

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
            'linear relresvecs': linear_relresvecs
            }


def jacobi_davidson(A,
                    v0,  # starting vector
                    tol=1e-5,
                    maxiter=None,
                    M=None,
                    inner_product=_ipstd
                    ):
    '''Jacobi-Davidson for the largest-magnitude eigenvalue of a
    self-adjoint operator.
    '''
    xtype = upcast(A.dtype, v0.dtype)
    num_unknowns = len(v0)
    if maxiter is None:
        maxiter = num_unknowns
    t = v0
    # Set up fields.
    V = np.empty((num_unknowns, maxiter), dtype=xtype)
    AV = np.empty((num_unknowns, maxiter), dtype=xtype)
    B = np.empty((maxiter, maxiter), dtype=float)

    resvec = []
    info = 1
    for m in range(maxiter):
        # orthgonalize t w.r.t. to the basis V
        t = orth_vec(t, V[:, 0:m], inner_product=inner_product)

        # normalize
        norm_t = np.sqrt(inner_product(t, t))[0, 0]
        assert norm_t > 1.0e-10, '||t|| = 0. Breakdown.'

        V[:, [m]] = t / norm_t
        AV[:, [m]] = _apply(A, V[:, [m]])

        # B = <V,AV>.
        # Only fill the lower triangle of B.
        for i in range(m+1):
            alpha = inner_product(V[:, [i]], AV[:, [m]])[0, 0]
            assert alpha.imag < 1.0e-10, 'A not self-adjoint?'
            B[m, i] = alpha.real

        # Compute the largest eigenpair of B.
        from scipy.linalg import eigh
        Theta, S = eigh(B[0:m+1, 0:m+1], lower=True)

        # Extract the largest-magnitude one.
        index = np.argmax(abs(Theta))
        theta = Theta[index]
        s = S[:, [index]]
        # normalize s in the inner product
        norm_s = np.sqrt(inner_product(s, s))[0, 0]
        assert norm_s > 1.0e-10, '||s|| = 0. Breakdown.'
        s /= norm_s

        # Get u, Au.
        u = np.dot(V[:, 0:m+1], s)
        Au = np.dot(AV[:, 0:m+1], s)

        # Compute residual.
        res = Au - theta*u
        resvec.append(np.sqrt(inner_product(res, res)[0, 0]))

        if resvec[-1] < tol:
            info = 0
            break
        else:
            # (Approximately) solve for t\ortho u from
            # (I-uu*)(A-theta I)(I-uu*) t = -r.
            def _shifted_projected_operator(A, u, theta):
                def _apply_proj(phi):
                    return phi - u * inner_product(u, phi)

                def _apply_shifted_projected_operator(phi):
                    return _apply_proj(A*_apply_proj(phi) - theta*_apply_proj(phi))

                return LinearOperator((num_unknowns, num_unknowns),
                                      _apply_shifted_projected_operator,
                                      dtype=A.dtype
                                      )
            assert abs(inner_product(u, res)) < 1.0e-10
            out = minres(_shifted_projected_operator(A, u, theta),
                         -res,
                         x0=np.zeros((num_unknowns,1)),
                         tol=1.0e-8,
                         M=M,
                         #Minv=None,
                         #Ml=_proj(u),
                         #Mr=_proj(u),
                         maxiter=num_unknowns,
                         inner_product=inner_product
                         )
            assert out[1] == 0, 'MINRES did not converge.'
            t = out[0]
            assert abs(inner_product(t, u)[0, 0]) < 1.0e-10, abs(inner_product(t, u))[0,0]
    return theta, u, info, resvec


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
    If the previous step was unsucessful, the step size is cut in half,
    but if the step was sucessful this strategy increases the step size based
    on the number of nonlinear solver iterations required in the previous step.
    In particular, the new step size \f$\Delta s_{new}\f$ is given by

       \Delta s_{new} = \Delta s_{old}\left(1 + a\left(\frac{N_{max} - N}{N_{max}}\right)^2\right).
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
                print('Continuation step failed (error code %d). Setting step size to %e.'
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
