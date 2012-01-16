#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Collection of numerical algorithms.
'''
# ==============================================================================
from scipy.sparse.linalg import LinearOperator, arpack
from scipy.sparse.sputils import upcast
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
def _norm_squared( x, inner_product = np.vdot ):
    '''Compute the norm^2 w.r.t. to a given scalar product.'''
    rho = inner_product( x, x )

    if rho.imag != 0.0: #abs(rho.imag) > abs(rho) * 1.0e-10:
        raise ValueError( 'inner_product not a proper inner product?' )

    return rho.real
# ==============================================================================
def _norm( x, inner_product = np.vdot ):
    '''Compute the norm w.r.t. to a given scalar product.'''
    norm2 = _norm_squared( x, inner_product = inner_product )
    if norm2 < 0.0:
        raise ValueError( '<x,x> = %g. Improper inner product?' % norm2 )
    return sqrt(norm2)
# ==============================================================================
def _normM_squared( x, M, inner_product = np.vdot ):
    '''Compute the norm^2 w.r.t. to a given scalar product.'''
    Mx = _apply( M, x )
    rho = inner_product( x, Mx )

    if rho.imag != 0.0: #abs(rho.imag) > abs(rho) * 1.0e-10:
        raise ValueError( 'M not positive definite?' )

    return rho.real, Mx
# ==============================================================================
def _normM( x, M, inner_product = np.vdot ):
    '''Compute the norm w.r.t. to a given scalar product.'''
    norm2, Mx = _normM_squared( x, M, inner_product = inner_product )
    if norm2 < 0.0:
        raise ValueError( '<x,Mx> = %g. M not positive definite?' % norm2 )
    return sqrt(norm2), Mx
# ==============================================================================
def _apply( A, x ):
    '''Implement A*x for different types of linear operators.'''
    if A is None:
        return x
    elif isinstance( A, np.ndarray ):
        return np.dot( A, x )
    elif isinstance( A, scipy.sparse.dia_matrix ) \
      or isinstance( A, scipy.sparse.csr_matrix ) \
      or isinstance( A, scipy.sparse.csc_matrix ):
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
            errorvec.append( _norm(error, inner_product=inner_product) )
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
    xtype = upcast( A.dtype, rhs.dtype, x0.dtype )
    if M:
        xtype = upcast( xtype, M.dtype )

    x = xtype(x0.copy())
    # If len(x)==1, then xtype strips off the np.array frame around the value.
    # This is needed for _apply, though.
    if len(x0) == 1:
        x = np.array( [x] )

    r = rhs - _apply(A, x)

    rho_old, Mr = _normM_squared( r, M, inner_product = inner_product )
    p = Mr.copy()

    if maxiter is None:
        maxiter = len(rhs)

    info = maxiter

    # Store rho0 = ||rhs||_M.
    rho0, _ = _normM_squared( rhs, M, inner_product = inner_product )

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

        rho_new, Mr = _normM_squared( r, M, inner_product = inner_product )

        relative_rho = sqrt(rho_new / rho0)
        if relative_rho < tol:
            # Compute exact residual
            r = rhs - _apply(A, x)
            rho_new, Mr = _normM_squared( r, M, inner_product = inner_product )
            relative_rho = sqrt(rho_new / rho0)
            if relative_rho < tol:
                info = 0
                if callback is not None:
                    callback( x, relative_rho )
                return x, info

        if callback is not None:
            callback( x, relative_rho )

        # update the search direction
        p = Mr + rho_new/rho_old * p

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
            errorvec.append( _norm(error, inner_product=inner_product) )
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
    if maxiter is None:
        maxiter = len(b)

    N = len(b)

    r0 = b - _apply(A, x0)

    xk = x0.copy()
    # --------------------------------------------------------------------------
    # Init Lanczos and MINRES
    r0 = _apply(r0_proj, r0)

    norm_Mr0, Mr0 = _normM(r0, M, inner_product = inner_product)

    # Compute M-norm of b.
    # Note: stopping criterion is ||M\(b-A*xk)||_M / ||M\b||_M < tol
    # If a projection is _applyied we obtain with b-A*xcor(xk)=Proj(b-A*xk) also
    # ||M\Proj(b-A*xk)||_M / ||M\b||_M = ||M\(b-A*xcor(xk))||_M / ||M\b||_M < tol
    norm_Mb, _ = _normM(b, M, inner_product = inner_product)

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
        assert abs(td.imag) < 1.0e-12
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
            norm_Fx_exact, _ = _normM(r, M, inner_product=inner_product)
            norm_rel_residual = norm_Fx_exact / norm_Mb
        else:
            norm_rel_residual = abs(y[0]) / norm_Mb

        # Compute residual explicitly if updated residual is below tolerance.
        if norm_rel_residual <= tol:
            # Compute the exact residual norm.
            xkcor = _apply(x_cor, xk)
            r = b - _apply(A, xkcor)
            norm_Fx_exact, _ = _normM(r, M, inner_product=inner_product)
            norm_rel_res_exact = norm_Fx_exact / norm_Mb
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
def gmres_wrap( linear_operator,
                b,
                x0,
                tol = 1.0e-5,
                maxiter = None,
                Mleft = None,
                Mright = None,
                inner_product = np.vdot,
                explicit_residual = False,
                exact_solution = None
              ):
    '''Wrapper around the GMRES method to get a vector with the relative
    residuals as return argument.
    '''
    # --------------------------------------------------------------------------
    def _callback( norm_rel_residual, x ):
        relresvec.append( norm_rel_residual )
        if exact_solution is not None:
            error = exact_solution - x
            errorvec.append( _norm(error, inner_product=inner_product) )
    # --------------------------------------------------------------------------

    relresvec = []
    errorvec = []

    sol, info = gmres( linear_operator,
                       b,
                       x0,
                       tol = tol,
                       maxiter = maxiter,
                       Mleft = Mleft,
                       Mright = Mright,
                       inner_product = inner_product,
                       callback = _callback,
                       explicit_residual = explicit_residual
                     )

    return sol, info, relresvec, errorvec
# ==============================================================================
def gmres( A,
           b,
           x0,
           tol = 1e-5,
           maxiter = None,
           Mleft = None,
           Mright = None,
           inner_product = np.vdot,
           callback = None,
           explicit_residual = False
         ):
    '''Preconditioned GMRES, pretty standard.
    memory consumption is about maxiter+1 vectors for the Arnoldi basis.
    Solves   Ml*A*Mr*y = Ml*b,  x=Mr*y. '''
    # --------------------------------------------------------------------------
    def _compute_explicit_xk():
        '''Compute approximation xk to the solution.'''
        yy = np.linalg.solve(H[:k+1, :k+1], y[:k+1])
        u  = _apply(Mright, _apply(V[:, :k+1], yy))
        xk = x0 + u
        return xk
    # --------------------------------------------------------------------------
    def _compute_explicit_residual( xk ):
        '''Compute residual explicitly.'''
        if xk is None:
            xk = _compute_explicit_xk()
        rk  = b - _apply(A, xk)
        rk  = _apply(Mleft, rk)
        return rk, xk
    # --------------------------------------------------------------------------
    xtype = upcast( A.dtype, b.dtype, x0.dtype )
    if Mleft:
        xtype = upcast( xtype, Mleft )
    if Mright:
        xtype = upcast( xtype, Mright )

    N = len(b)
    if not maxiter:
        maxiter = N

    info = 0

    # get memory for working variables
    V = np.zeros([N, maxiter+1], dtype=xtype) # Arnoldi basis
    H = np.zeros([maxiter+1, maxiter], dtype=xtype) # Hessenberg matrix

    # initialize working variables
    MleftB = _apply(Mleft, b)
    norm_MleftB = _norm(MleftB, inner_product=inner_product)
    # This may only save us the application of Ml to the same vector again if
    # x0 is the zero vector.
    norm_x0 = _norm(x0, inner_product=inner_product)
    if norm_x0 > np.finfo(float).eps:
        r    = b - _apply(A, x0)
        r    = _apply(Mleft, r)
        norm_r = _norm( r, inner_product=inner_product)
    else:
        x0 = np.zeros( N )
        r    = MleftB
        norm_r = norm_MleftB

    V[:, 0] = r / norm_r
    norm_rel_residual  = norm_r / norm_MleftB
    # Right hand side of projected system:
    y = np.zeros( maxiter+1, dtype=xtype )
    y[0] = norm_r
    # Givens rotations:
    G = []
    xk = x0
    k = 0

    if callback is not None:
        callback(norm_rel_residual, xk)

    while norm_rel_residual > tol and k < maxiter:
        xk = None
        rk = None
        # Apply operator Ml*A*Mr
        V[:, k+1] = _apply(Mleft, _apply(A, _apply(Mright, V[:, k])))

        # orthogonalize (MGS)
        for i in xrange(k+1):
            H[i, k] = inner_product(V[:, i], V[:, k+1])
            V[:, k+1] = V[:, k+1] - H[i, k] * V[:, i]
        H[k+1, k] = _norm(V[:, k+1], inner_product=inner_product)
        V[:, k+1] = V[:, k+1] / H[k+1, k]

        # Apply previous Givens rotations.
        for i in xrange(k):
            H[i:i+2, k] = _apply(G[i], H[i:i+2, k])

        # Compute and apply new Givens rotation.
        G.append(_givens(H[k, k], H[k+1, k]))
        H[k:k+2, k] = _apply(G[k], H[k:k+2, k])
        y[k:k+2] = _apply(G[k], y[k:k+2])

        # Update residual norm.
        if explicit_residual:
            if rk is None:
                rk, xk = _compute_explicit_residual( xk )
            norm_r = _norm(rk, inner_product=inner_product)
            norm_rel_residual = norm_r / norm_MleftB
        else:
            norm_rel_residual = abs(y[k+1]) / norm_MleftB

        # convergence of updated residual or maxiter reached?
        if norm_rel_residual < tol or k+1 == maxiter:
            norm_ur = norm_rel_residual
            if rk is None:
                rk, xk = _compute_explicit_residual( xk )
            norm_r = _norm(rk, inner_product=inner_product)
            norm_rel_residual = norm_r / norm_MleftB

            # No convergence of expl. residual?
            if norm_rel_residual >= tol:
                # Was this the last iteration?
                if k+1 == maxiter:
                    print 'No convergence! expl. res = %e >= tol =%e in last it. %d (upd. res = %e)' % (norm_rel_residual, tol, k, norm_ur)
                    info = 1
                else:
                    print 'Expl. res = %e >= tol = %e > upd. res = %e in it. %d' % (norm_rel_residual, tol, norm_ur, k)

        if callback is not None:
            if xk is None:
                xk = _compute_explicit_xk()
            callback(norm_rel_residual, xk)

        k += 1

    return xk, info
# ==============================================================================
def _givens(a, b):
    '''Givens rotation
    [   c       s    ] * [a] = [r]
    [-conj(s) conj(c)]   [b]   [0]
    r real and non-negative.'''
    if abs(b) == 0:
        r = abs(a)
        c = a.conjugate() / r
        s = 0
    elif abs(a) == 0:
        r = abs(b)
        c = 0
        s = b.conjugate() / r
    elif abs(b) > abs(a):
        absb = abs(b)
        t = a.conjugate() / absb
        u = sqrt(1 + t.real**2 + t.imag**2)
        c = t / u
        s = (b.conjugate()/absb) / u
        r = absb * u
    else:
        absa = abs(a)
        t = b.conjugate()/absa
        u = sqrt(1 + t.real**2 + t.imag**2)
        c = (a.conjugate()/absa)/u
        s = t/u
        r = absa*u
    return np.array([[c, s],
                     [-s.conjugate(), c.conjugate()]])
# ==============================================================================
def newton( x0,
            model_evaluator,
            nonlinear_tol = 1.0e-10,
            maxiter = 20,
            linear_solver = cg_wrap,
            forcing_term = 'constant',
            eta0 = 1.0e-1,
            eta_min = 1.0e-6,
            eta_max = 1.0e-2,
            alpha = 1.5, # only used by forcing_term='type 2'
            gamma = 0.9  # only used by forcing_term='type 2'
          ):
    '''Newton's method with different forcing terms.
    '''
    from scipy.constants import golden

    # some initializations
    error_code = 0
    k = 0

    # Create Jacobian as linear operator object.
    jacobian = LinearOperator( (len(x0), len(x0)),
                               model_evaluator.apply_jacobian,
                               dtype = model_evaluator.dtype
                             )

    ## Create Jacobian as linear operator object.
    #preconditioner = LinearOperator( (len(x0), len(x0)),
                               #None,
                               #dtype = model_evaluator.dtype
                             #)

    x = x0
    Fx = model_evaluator.compute_f( x )
    Fx_norms = [ _norm( Fx, inner_product=model_evaluator.inner_product ) ]
    eta_previous = None
    while Fx_norms[-1] > nonlinear_tol and k < maxiter:
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
            eta = abs(Fx_norms[-1] - linear_relresvec[-1]) / Fx_norms[-2]
            eta = max( eta, eta_previous**golden, eta_min )
            eta = min( eta, eta_max )
        elif forcing_term == 'type 2':
            eta = gamma * (Fx_norms[-1] / Fx_norms[-2])**alpha
            eta = max( eta, gamma * eta_previous**alpha, eta_min )
            eta = min( eta, eta_max )
        else:
            print 'Unknown forcing term \'%s\'. Abort.'
            return
        eta_previous = eta

        # Solve the linear system.
        model_evaluator.set_current_x( x )
        x_update, info, linear_relresvec, _ = linear_solver( jacobian,
                                                             -Fx,
                                                             np.zeros( len(x0) ),
                                                             tol = eta,
                                                             inner_product = model_evaluator.inner_product
                                                           )
        # make sure the solution is alright
        assert( info == 0 )

        # perform the Newton update
        x += x_update

        # do the household
        k += 1
        Fx = model_evaluator.compute_f( x )
        Fx_norms.append( _norm( Fx, inner_product=model_evaluator.inner_product ) )

    if k == maxiter:
        error_code = 1

    return x, error_code, Fx_norms
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
