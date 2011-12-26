#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Collection of numerical algorithms.
'''
# ==============================================================================
from scipy.sparse.linalg import LinearOperator, arpack
import numpy as np
import scipy
from math import sqrt
# ==============================================================================
def l2_condition_number( linear_operator ):
    small_eigenval = arpack.eigen( linear_operator,
                                   k = 1,
                                   sigma = None,
                                   which = 'SM',
                                   return_eigenvectors = False
                                 )
    large_eigenval = arpack.eigen( linear_operator,
                                   k = 1,
                                   sigma = None,
                                   which = 'LM',
                                   return_eigenvectors = False
                                 )

    return large_eigenval[0] / small_eigenval[0]
# ==============================================================================
def _norm_squared( x, M = None, inner_product = np.vdot ):
    '''Compute the norm^2 w.r.t. to a given scalar product.'''
    Mx = _apply( M, x )
    rho = inner_product( x, Mx )

    if rho.imag != 0.0: #abs(rho.imag) > abs(rho) * 1.0e-10:
        raise ValueError( 'M not positive definite?' )

    return rho.real, Mx
# ==============================================================================
def _norm( x, M = None, inner_product = np.vdot ):
    '''Compute the norm w.r.t. to a given scalar product.'''
    norm2, Mx = _norm_squared( x, M = M, inner_product = inner_product )
    return sqrt(norm2), Mx
# ==============================================================================
def _apply( A, x ):
    '''Implement A*x for different types of linear operators.'''
    if A is None:
        return x
    elif isinstance( A, np.ndarray ):
        return np.dot( A, x )
    elif isinstance( A, scipy.sparse.dia_matrix ) \
      or isinstance( A, scipy.sparse.csr_matrix ):
        return A * x
    elif isinstance( A, scipy.sparse.linalg.LinearOperator ):
        return A * x
    else:
        raise ValueError( 'Unknown operator type "%s".' % type(A) )
# ==============================================================================
def cg_wrap( linear_operator,
             rhs,
             x0,
             tol = 1.0e-5,
             maxiter = 1000,
             M = None,
             explicit_residual = False,
             exact_solution = None,
             inner_product = np.vdot
           ):
    '''Wrapper around the CG method to get a vector with the relative residuals
    as return argument.
    '''
    # --------------------------------------------------------------------------
    def _callback( x, relative_residual ):
        relresvec.append( relative_residual )
        if exact_solution is not None:
            error = exact_solution - x
            nrm, _ = _norm(error, inner_product=inner_product)
            errorvec.append( nrm )
    # --------------------------------------------------------------------------

    relresvec = []
    errorvec = []

    sol, info = cg( linear_operator,
                    rhs,
                    x0,
                    tol = tol,
                    maxiter = maxiter,
                    M = M,
                    callback = _callback,
                    explicit_residual = explicit_residual,
                    inner_product = inner_product
                  )

    return sol, info, relresvec, errorvec
# ==============================================================================
def cg( A,
        rhs,
        x0,
        tol = 1.0e-5,
        maxiter = None,
        M = None,
        callback = None,
        explicit_residual = False,
        inner_product = np.vdot
      ):
    '''Conjugate gradient method with different inner product.
    '''
    x = x0.copy()
    r = rhs - _apply(A, x)

    rho_old, z = _norm_squared( r, M=M, inner_product = inner_product )

    if M is not None:
        p = z.copy()
    else:
        p = r.copy()

    if maxiter is None:
        maxiter = len(rhs)

    info = maxiter

    # store the initial residual
    rho0 = rho_old

    if callback is not None:
        callback( x, sqrt(rho_old / rho0) )

    for k in xrange( maxiter ):
        Ap = _apply(A, p)

        # update current guess and residial
        alpha = rho_old / inner_product( p, Ap )
        x += alpha * p
        if explicit_residual:
            r = rhs - _apply(A, x)
        else:
            r -= alpha * Ap

        rho_new, z = _norm_squared( r, M=M, inner_product = inner_product )

        relative_rho = sqrt(rho_new / rho0)
        if relative_rho < tol:
            # Compute exact residual
            r = rhs - _apply(A, x)
            rho_new, z = _norm_squared( r, M=M, inner_product = inner_product )
            relative_rho = sqrt(rho_new / rho0)
            if relative_rho < tol:
                info = 0
                if callback is not None:
                    callback( x, relative_rho )
                return x, info

        if callback is not None:
            callback( x, relative_rho )

        # update the search direction
        if M is not None:
            p = z + rho_new/rho_old * p
        else:
            p = r + rho_new/rho_old * p

        rho_old = rho_new

    return x, info
# ==============================================================================
def minres_wrap( linear_operator,
                 b,
                 x0,
                 tol = 1.0e-5,
                 maxiter = None,
                 M = None,
                 inner_product = np.vdot,
                 explicit_residual = False,
                 exact_solution = None
               ):
    '''
    Wrapper around the MINRES method to get a vector with the relative residuals
    as return argument.
    '''
    # --------------------------------------------------------------------------
    def _callback( norm_rel_residual, x ):
        relresvec.append( norm_rel_residual )
        if exact_solution is not None:
            error = exact_solution - x
            nrm, _ = _norm(error, inner_product=inner_product)
            errorvec.append( nrm )
    # --------------------------------------------------------------------------

    relresvec = []
    errorvec = []

    sol, info = minres( linear_operator,
                        b,
                        x0,
                        tol = tol,
                        maxiter = maxiter,
                        M = M,
                        inner_product = inner_product,
                        callback = _callback,
                        explicit_residual = explicit_residual
                      )

    return sol, info, relresvec, errorvec
# ==============================================================================
def minres( A,
            b,
            x0,
            tol = 1e-5,
            maxiter = None,
            M = None,
            inner_product = np.vdot,
            x_cor = None,
            r0_proj = None,
            proj = None,
            callback = None,
            explicit_residual = False
            ):
    # --------------------------------------------------------------------------
    info = 0
    maxiter = len(b)

    N = len(b)

    r0 = b - _apply(A, x0)

    xk = x0.copy()
    # --------------------------------------------------------------------------
    # Init Lanczos and MINRES
    r0 = _apply(r0_proj, r0)

    norm_Mr0, Mr0 = _norm(r0, M=M, inner_product = inner_product)

    # Compute M-norm of b.
    # Note: stopping criterion is ||M\(b-A*xk)||_M / ||M\b||_M < tol
    # If a projection is applied we obtain with b-A*xcor(xk)=Proj(b-A*xk) also
    # ||M\Proj(b-A*xk)||_M / ||M\b||_M = ||M\(b-A*xcor(xk))||_M / ||M\b||_M < tol
    norm_Mb, _ = _norm(b, M=M, inner_product = inner_product)

    norm_rel_residual = norm_Mr0 / norm_Mb
    # --------------------------------------------------------------------------

    # Allocate and initialize the 'large' memory blocks.
    # Last and current Lanczos vector:
    V = [np.empty(N), Mr0 / norm_Mr0]
    # M*v[i] = P[1], M*v[i-1] = P[0]
    P = [np.zeros(N), r0 / norm_Mr0]
    # Necessary for efficient update of xk:
    W = [np.zeros(N), np.zeros(N)]
    # some small helpers
    ts = 0.0           # (non-existing) first off-diagonal entry (corresponds to pi1)
    y  = [norm_Mr0, 0] # first entry is (updated) residual
    G2 = np.eye(2)     # old givens rotation
    G1 = np.eye(2)     # even older givens rotation ;)
    k = 0

    if callback is not None:
        callback( norm_rel_residual, xk )

    # --------------------------------------------------------------------------
    # Lanczos + MINRES iteration
    # --------------------------------------------------------------------------
    while norm_rel_residual > tol and k <= maxiter:
        # ---------------------------------------------------------------------
        # Lanczos
        tsold = ts
        z  = _apply(A, V[1])
        z  = _apply(proj, z)
        # tsold = inner_product(V[0], z)
        z  = z - tsold * P[0]
        # Should be real! (diagonal element):
        td = inner_product(V[1], z)
        assert abs(td.imag) < 1.0e-15
        td = td.real
        z  = z - td * P[1]

        ## local reorthogonalization
        #tsold2 = inner_product(V[0], z)
        #z   = z - tsold2 * P[0]
        #td2 = inner_product(V[1], z)
        #td  = td + td2
        #z   = z - td2*P[1]
        #tsold = tsold + tsold2

        # needed for QR-update:
        R = _apply(G1, [0, tsold])
        R = np.append(R, [0.0, 0.0])

        # Apply the preconditioner.
        v  = _apply(M, z)
        alpha = inner_product(z, v)
        assert alpha.imag == 0.0
        alpha = alpha.real
        assert alpha > 0.0
        ts = sqrt( alpha )

        P  = [P[1], z / ts]
        V  = [V[1], v / ts]

        # ----------------------------------------------------------------------
        # (implicit) update of QR-factorization of Lanczos matrix
        R[2:4] = [td, ts]
        R[1:3] = _apply(G2, R[1:3])
        G1 = G2
        # compute new givens rotation.
        gg = np.linalg.norm( R[2:4] )
        gc = R[2] / gg
        gs = R[3] / gg
        G2 = np.array([ [gc,  gs],
                        [-gs, gc] ])
        R[2] = gg
        R[3] = 0.0
        y = _apply(G2, y)

        # ----------------------------------------------------------------------
        # update solution
        z  = (V[0] - R[0]*W[0] - R[1]*W[1]) / R[2]
        W  = [W[1], z]
        xk = xk + y[0] * z
        y  = [y[1], 0]

        # ----------------------------------------------------------------------
        # update residual
        if explicit_residual:
            xkcor = _apply(x_cor, xk)
            r = b - _apply(A, xkcor)
            norm_res_exact, _ = _norm(r, M, inner_product=inner_product)
            norm_rel_residual = norm_res_exact / norm_Mb
        else:
            norm_rel_residual = abs(y[0]) / norm_Mb

        # Compute residual explicitly if updated residual is below tolerance.
        if norm_rel_residual <= tol:
            # Compute the exact residual norm.
            xkcor = _apply(x_cor, xk)
            r = b - _apply(A, xkcor)
            norm_res_exact, _ = _norm(r, M, inner_product=inner_product)
            norm_rel_res_exact = norm_res_exact / norm_Mb
            if norm_rel_res_exact > tol:
                print ( 'Info (iter %d): Updated residual is below tolerance, '
                      + 'explicit residual is NOT!\n  (resEx=%g > tol=%g >= '
                      + 'resup=%g)\n' \
                      ) % (k, norm_rel_res_exact, tol, norm_rel_residual)
            norm_rel_residual = norm_rel_res_exact

        if callback is not None:
            callback( norm_rel_residual, xk )

        k += 1
    # end MINRES iteration
    # --------------------------------------------------------------------------

    # Ultimate convergence test.
    if norm_rel_residual > tol:
        print 'No convergence after iter %d (res=%e > tol=%e)' % \
              (k-1, norm_rel_residual, tol)
        xkcor = _apply(x_cor, xk)
        info = 1

    return xkcor, info
# ==============================================================================
def newton( x0,
            model_evaluator,
            nonlinear_tol = 1.0e-10,
            max_iters = 20
          ):
    '''Poor man's Newton method.
    '''
    # --------------------------------------------------------------------------
    def _newton_jacobian( dx ):
        '''Create the linear operator object to be used for CG.'''
        model_evaluator.set_current_psi( x )
        return model_evaluator.compute_jacobian( dx )
    # --------------------------------------------------------------------------
    # some initializations
    error_code = 0
    iters = 0

    # create the linear operator object
    jacobian = LinearOperator( (len(x0), len(x0)),
                               _newton_jacobian,
                               dtype = complex
                             )

    x = x0
    res = model_evaluator.compute_f( x )
    rho = model_evaluator.norm( res )
    print "Norm(res) =", rho
    while abs(rho) > nonlinear_tol:
        # solve the Newton system using cg
        x_update, info = cg( jacobian,
                             -res,
                             x0 = x,
                             tol = 1.0e-10
                           )

        # make sure the solution is alright
        assert( info == 0 )

        # perform the Newton update
        x = x + x_update

        # do the household
        iters += 1
        res = model_evaluator.compute_f( x )
        rho = model_evaluator.norm( res )
        print "Norm(res) =", abs(rho)
        if iters == max_iters:
            error_code = 1
            break

    return x, error_code, iters
# ==============================================================================
def poor_mans_continuation( x0,
                            model_evaluator,
                            initial_parameter_value,
                            initial_step_size = 1.0e-2,
                            minimal_step_size = 1.0e-6,
                            maximum_step_size = 1.0e-1,
                            max_steps = 1000,
                            nonlinear_tol = 1.0e-10,
                            max_newton_iters = 5,
                            adaptivity_aggressiveness = 1.0
                          ):
    '''Poor man's parameter continuation. With adaptive step size.
    If the previous step was unsucessful, the step size is cut in half,
    but if the step was sucessful this strategy increases the step size based
    on the number of nonlinear solver iterations required in the previous step.
    In particular, the new step size \f$\Delta s_{new}\f$ is given by

       \Delta s_{new} = \Delta s_{old}\left(1 + a\left(\frac{N_{max} - N}{N_{max}}\right)^2\right).
    '''

    # write header of the statistics file
    stats_file = open( 'continuationData.dat', 'w' )
    stats_file.write( '# step    parameter     norm            Newton iters\n' )
    stats_file.flush()

    parameter_value = initial_parameter_value
    x = x0

    current_step_size = initial_step_size

    for k in range( max_steps ):
        print "Continuation step %d (parameter=%e)..." % ( k, parameter_value )

        # Try to converge to a solution and adapt the step size.
        converged = False
        while current_step_size > minimal_step_size:
            x_new, error_code, iters = newton( x,
                                               model_evaluator,
                                               nonlinear_tol = nonlinear_tol,
                                               max_iters = max_newton_iters
                                             )
            if error_code != 0:
                current_step_size *= 0.5
                print "Continuation step failed (error code %d). Setting step size to %e." \
                      % ( error_code, current_step_size )
            else:
                current_step_size *= 1.0 + adaptivity_aggressiveness * \
                                           (float(max_newton_iters-iters)/max_newton_iters)**2
                converged = True
                x = x_new
                print "Continuation step success!"
                break

        if not converged:
            print "Could not find a solution although the step size was %e. Abort." % current_step_size
            break


        stats_file.write( '  %4d    %.5e   %.5e    %d\n' %
                          ( k, parameter_value, model_evaluator.energy(x), iters )
                        )
        stats_file.flush()
        #model_evaluator.write( x, "step" + str(k) + ".vtu" )

        parameter_value += current_step_size
        model_evaluator.set_parameter( parameter_value )
        
    stats_file.close()
    
    print "done."
    return
# ==============================================================================
