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
def _ipstd( X, Y ):
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
# ==============================================================================
def _norm_squared( x, Mx = None, inner_product = _ipstd ):
    '''Compute the norm^2 w.r.t. to a given scalar product.'''
    assert( len(x.shape)==2 )
    assert( x.shape[1]==1 )
    if Mx is None:
        rho = inner_product(x, x)[0,0]
    else:
        assert( len(Mx.shape)==2 )
        assert( Mx.shape[1]==1 )
        rho = inner_product(x, Mx)[0,0]

#    if rho.imag != 0.0: #abs(rho.imag) > abs(rho) * 1.0e-10:
    if abs(rho.imag) > abs(rho) *1e-10:
        raise ValueError( 'M not positive definite?' )

    rho = rho.real

    if rho < 0.0:
        raise ValueError( '<x,Mx> = %g. M not positive definite?' % norm2 )

    return rho.real
# ==============================================================================
def _norm( x, Mx = None, inner_product = _ipstd ):
    '''Compute the norm w.r.t. to a given scalar product.'''
    return np.sqrt(_norm_squared( x, Mx = Mx, inner_product = inner_product ) )
# ==============================================================================
def _apply( A, x ):
    '''Implement A*x for different types of linear operators.'''
    if A is None:
        return x
    elif isinstance( A, np.ndarray ):
        return np.dot( A, x )
    elif scipy.sparse.isspmatrix(A):
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
             inner_product = _ipstd
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
        inner_product = _ipstd
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

    Mr = _apply(M, r)
    rho_old = _norm_squared(r, Mr, inner_product = inner_product)
    p = Mr.copy()

    if maxiter is None:
        maxiter = len(rhs)

    info = maxiter

    # Store rho0 = ||rhs||_M.
    Mrhs = _apply(M, rhs)
    rho0 = _norm_squared( rhs, Mrhs, inner_product = inner_product )

    if callback is not None:
        callback( x, np.sqrt(rho_old / rho0) )

    for k in xrange( maxiter ):
        Ap = _apply(A, p)

        # update current guess and residial
        alpha = rho_old / inner_product( p, Ap )
        x += alpha * p
        if explicit_residual:
            r = rhs - _apply(A, x)
        else:
            r -= alpha * Ap

        Mr = _apply(M, r)
        rho_new = _norm_squared( r, Mr, inner_product = inner_product )

        relative_rho = np.sqrt(rho_new / rho0)
        if relative_rho < tol:
            # Compute exact residual
            r = rhs - _apply(A, x)
            Mr = _apply(M, r)
            rho_new = _norm_squared( r, Mr, inner_product = inner_product )
            relative_rho = np.sqrt(rho_new / rho0)
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
                 inner_product = _ipstd,
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
            Ml = None,
            Mr = None,
            inner_product = _ipstd,
            explicit_residual = False,
            return_lanczos = False,
            full_reortho = False,
            exact_solution = None
            ):
    # --------------------------------------------------------------------------
    # This MINRES solves M*Ml*A*Mr*y = M*Ml*b,  x=Mr*y
    # where Ml and Mr have to be such that Ml*A*Mr is self-adjoint in the 
    # inner_product. M has to be self-adjoint and positive-definite w.r.t.
    # inner_product. 
    # 
    # Details:
    # The Lanczos procedure is used with the operator M*Ml*A*Mr and the 
    # inner product defined by inner_product(M^{-1}x,y). The initial vector 
    # for Lanczos is r0 = M*Ml*(b - A*x0) -- note that Mr is not used for
    # the initial vector!
    #
    # Stopping criterion is 
    # ||M*Ml*(b-A*(x0+Mr*yk))||_{M^{-1}} / ||M*Ml*b||_{M^{-1}} <= tol

    info = 0
    N = len(b)

    xtype = upcast( A.dtype, b.dtype, x0.dtype )
    if M:
        xtype = upcast( xtype, M )
    if Ml:
        xtype = upcast( xtype, Ml )
    if Mr:
        xtype = upcast( xtype, Mr )

    if maxiter is None:
        maxiter = N

    # Compute M-norm of M*Ml*b.
    Mlb = _apply(Ml, b)
    MMlb = _apply(M, Mlb)
    norm_MMlb = _norm(Mlb, MMlb, inner_product = inner_product)

    # --------------------------------------------------------------------------
    # Init Lanczos and MINRES
    r0 = b - _apply(A, x0)
    Mlr0 = _apply(Ml, r0)
    MMlr0 = _apply(M, Mlr0)
    norm_MMlr0 = _norm(Mlr0, MMlr0, inner_product = inner_product)

    # initial relative residual norm 
    relresvec = [norm_MMlr0 / norm_MMlb]
    
    # compute error?
    if exact_solution is not None:
        errvec = [_norm(exact_solution - x0, inner_product = inner_product)]

    # --------------------------------------------------------------------------
    # Allocate and initialize the 'large' memory blocks.
    if return_lanczos or full_reortho:
        Vfull = np.c_[MMlr0 / norm_MMlr0, np.zeros([N,maxiter], dtype=xtype)]
        Pfull = np.c_[Mlr0 / norm_MMlr0, np.zeros([N,maxiter], dtype=xtype)]
        Tfull = scipy.sparse.lil_matrix( (maxiter+1,maxiter) )
    # Last and current Lanczos vector:
    V = np.c_[np.zeros(N), MMlr0 / norm_MMlr0]
    # M*v[i] = P[1], M*v[i-1] = P[0]
    P = np.c_[np.zeros(N), Mlr0 / norm_MMlr0]
    # Necessary for efficient update of yk:
    W = np.c_[np.zeros(N), np.zeros(N)]
    # some small helpers
    ts = 0.0           # (non-existing) first off-diagonal entry (corresponds to pi1)
    y  = [norm_MMlr0, 0] # first entry is (updated) residual
    G2 = np.eye(2)     # old givens rotation
    G1 = np.eye(2)     # even older givens rotation ;)
    k = 0
    
    # resulting approximation is xk = x0 + Mr*yk
    yk = np.zeros((N,1))

    # --------------------------------------------------------------------------
    # Lanczos + MINRES iteration
    # --------------------------------------------------------------------------
    while relresvec[-1] > tol and k < maxiter:
        # ---------------------------------------------------------------------
        # Lanczos
        tsold = ts
        z  = _apply(Mr, V[:,[1]])
        z  = _apply(A, z)
        z  = _apply(Ml, z)

        # full reortho?
        if full_reortho:
            for i in range(0,k-1):
                ip = inner_product(Vfull[:,[i]], z)[0,0]
                if abs(ip) > 1.0e-9:
                    raise ValueError('abs(ip) = %g > 1.0e-9: The Krylov basis has become linearly dependent. Maxiter too large?' % abs(ip))
                z = z - ip * Pfull[:,[i]]

        # tsold = inner_product(V[0], z)
        z  = z - tsold * P[:,[0]]
        # Should be real! (diagonal element):
        td = inner_product(V[:,[1]], z)[0,0]
        if abs(td.imag) > 1.0e-12:
            raise ValueError('abs(td.imag) = %g > 1.0e-12' % abs(td.imag))
        td = td.real
        z  = z - td * P[:,1:2]

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
        alpha = inner_product(z, v)[0,0]
        assert alpha.imag == 0.0
        alpha = alpha.real
        assert alpha >= 0.0
        ts = np.sqrt( alpha )

        if ts>0.0:
            P  = np.c_[P[:,[1]], z / ts]
            V  = np.c_[V[:,[1]], v / ts]
        else:
            P  = np.c_[P[:,[1]], np.zeros(N)]
            V  = np.c_[V[:,[1]], np.zeros(N)]

        
        # store new vectors in full basis
        if return_lanczos or full_reortho:
            if ts>0.0:
                Vfull[:,[k+1]] = v / ts
                Pfull[:,[k+1]] = z / ts
            Tfull[k,k] = td        # diagonal
            Tfull[k+1,k] = ts      # subdiagonal
            if k+1 < maxiter:
                Tfull[k,k+1] = ts  # superdiagonal

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
        z  = (V[:,0:1] - R[0]*W[:,0:1] - R[1]*W[:,1:2]) / R[2]
        W  = np.c_[W[:,1:2], z]
        yk = yk + y[0] * z
        y  = [y[1], 0]

        # ----------------------------------------------------------------------
        # update residual
        if exact_solution is not None:
            xk = x0 + _apply(Mr, yk)
            errvec.append(_norm(exact_solution - xk, inner_product=inner_product))
        if explicit_residual:
            xk = x0 + _apply(Mr, yk)
            r_exp = b - _apply(A, xk)
            r_exp = _apply(Ml, r_exp)
            Mr_exp = _apply(M, r_exp)
            norm_r_exp = _norm(r_exp, Mr_exp, inner_product=inner_product)
            relresvec.append(norm_r_exp / norm_MMlb)
        else:
            relresvec.append(abs(y[0]) / norm_MMlb)

        # Compute residual explicitly if updated residual is below tolerance.
        if relresvec[-1] <= tol or k+1 == maxiter:
            norm_r_upd = relresvec[-1]
            # Compute the exact residual norm (if not yet done above)
            if not explicit_residual:
                xk = x0 + _apply(Mr, yk)
                r_exp = b - _apply(A, xk)
                r_exp = _apply(Ml, r_exp)
                Mr_exp = _apply(M, r_exp)
                norm_r_exp = _norm(r_exp, Mr_exp, inner_product=inner_product)
                relresvec[-1] = norm_r_exp / norm_MMlb
            # No convergence of explicit residual?
            if relresvec[-1] > tol:
                # Was this the last iteration?
                if k+1 == maxiter:
                    print 'No convergence! expl. res = %e >= tol =%e in last it. %d (upd. res = %e)' \
                        % (relresvec[-1], tol, k, norm_r_upd)
                    info = 1
                else:
                    print ( 'Info (iter %d): Updated residual is below tolerance, '
                          + 'explicit residual is NOT!\n  (resEx=%g > tol=%g >= '
                          + 'resup=%g)\n' \
                          ) % (k, relresvec[-1], tol, norm_r_upd)

        k += 1
    # end MINRES iteration
    # --------------------------------------------------------------------------

    ret = (xk, info, relresvec)
    if exact_solution is not None:
        ret = ret + (errvec,)

    if return_lanczos:
        Vfull = Vfull[:,0:k+1]
        Pfull = Pfull[:,0:k+1]
        Tfull = Tfull[0:k+1,0:k]
        ret = ret + (Vfull, Pfull, Tfull)
    return ret
# ==============================================================================
def _direct_solve(A, rhs):
    '''Solve a (dense) equation system directly.'''
    if type(A) == np.float64:
        return rhs / A
    else:
        return np.linalg.solve(A, rhs)
# ==============================================================================
def get_projection(W, AW, b, x0, inner_product = _ipstd):
    """Get projection and appropriate initial guess for use in deflated methods.

    Arguments:
        W:  the basis vectors used for deflation (Nxk array).
        AW: A*W, where A is the operator of the linear algebraic system to be
            deflated. A has to be self-adjoint w.r.t. inner_product. Do not
            include the positive-definite preconditioner (argument M in MINRES)
            here. Let N be the dimension of the vector space the operator is
            defined on.
        b:  the right hand side of the linear system (array of length N).
        x0: the initial guess (array of length N).
        inner_product: the inner product also used for the deflated iterative
            method.

    Returns:
        P:  the projection to be used as _right_ preconditioner (e.g. Mr=P in
            MINRES). The preconditioned operator A*P is self-adjoint w.r.t. 
            inner_product.
            P(x)=x + W*inner_product(W, A*W)^{-1}*inner_product(A*W, x)
        x0new: an adapted initial guess s.t. the deflated iterative solver 
            does not break down (in exact arithmetics).
        AW: AW=A*W. This is returned in order to reduce the total number of
            matrix-vector multiplications with A.
    """
    # --------------------------------------------------------------------------
    def Pfun(x):
        '''Computes x - W * E\<AW,x>.'''
        return x - np.dot(W, _direct_solve(E, inner_product(AW, x)))
    # --------------------------------------------------------------------------
    E = inner_product(W, AW)
    EWb = _direct_solve(E, inner_product(W, b))

    # Define projection operator.
    N = len(b)
    dtype = upcast(W.dtype, AW.dtype, b.dtype, x0.dtype)
    P = scipy.sparse.linalg.LinearOperator( [N,N], Pfun, matmat=Pfun,
                                            dtype=dtype)
    # Get updated x0.
    x0new = P*x0 +  np.dot(W, EWb)

    return P, x0new
# ==============================================================================
def get_ritz(W, AW, Vfull, Tfull, M=None, inner_product = _ipstd):
    """Compute Ritz pairs from a (possibly deflated) Lanczos procedure. 
    
    Arguments
        W:  a Nxk array. W's columns must be orthonormal w.r.t. the
            M-inner-product (inner_product(M^{-1} W, W) = I_k).
        AW: contains the result of A applied to W (passed in order to reduce #
            of matrix-vector multiplications with A).
        Vfull: a Nxn array. Vfull's columns must be orthonormal w.r.t. the
            M-inner-product. Vfull and Tfull must be created with a (possibly
            deflated) Lanczos procedure (e.g. CG/MINRES). For example, Vfull
            and Tfull can be obtained from MINRES applied to a linear system
            with the operator A, the inner product inner_product, the HPD
            preconditioner M and the right preconditioner Mr set to the
            projection obtained with get_projection(W, AW, ...).
        Tfull: see Vfull.
        M:  The preconditioner used in the Lanczos procedure.

        The arguments thus have to fulfill the following equations:
            AW = A*W.
            M*A*Mr*Vfull[:,0:-1] = Vfull*Tfull,
                 where Mr=get_projection(W, AW,...,inner_product).
            inner_product( M^{-1} [W,Vfull], [W,Vfull] ) = I_{k+n}.

    Returns:
        ritz_vals: an array with n+k Ritz values.
        ritz_vecs: a Nx(n+k) array where the ritz_vecs[:,i] is the 
            Ritz vector for the Ritz value ritz_vals[i]. The Ritz vectors
            also are orthonormal w.r.t. the M-inner-product, that is
                inner_product( M^{-1}*ritz_vecs, ritz_vecs ) = I_{k+n}.
        norm_ritz_res: an array with n+k residual norms. norm_ritz_res[i] is 
            the M^{-1}-norm of the residual
                M*A*ritz_vecs[:,i] - ritz_vals[i]*ritz_vecs[:,i].
            ritz_vals, ritz_vecs and norm_ritz_res are sorted s.t. the 
            residual norms are ascending.
    
    Under the above assumptions, [W, Vfull] is orthonormal w.r.t. the
    M-inner-product. Then the Ritz pairs w.r.t. the operator M*A, the basis [W,
    Vfull[:,0:-1]] and the M-inner-product are computed. Also the M-norm of the
    Ritz pair residual is computed. The computation of the residual norms do
    not need the application of the operator A, but the preconditioner has to
    be applied to the basis W. The computation of the residual norm may be
    unstable (it seems as if residual norms below 1e-8 cannot be achieved...
    note that the actual residual may be lower!).
    """
    nW = W.shape[1]
    nVfull = Vfull.shape[1]
    E = inner_product(W, AW)        # ~
    B1 = inner_product(AW, Vfull)   # can (and should) be obtained from MINRES
    B = B1[:, 0:-1]

    # Stack matrices appropriately: [E, B; B', Tfull(1:end-1,:)].
    ritzmat = np.r_[    np.c_[E,B],
                        np.c_[B.T.conj(), Tfull[0:-1,:].todense()] 
                   ]

    # Compute Ritz values / vectors.
    from scipy.linalg import eigh
    lam, U = eigh(ritzmat)

    # Compute residual norms.
    norm_ritz_res = np.zeros(lam.shape[0])
    if nW>0:
        Einv = np.linalg.inv(E) # ~
    else:
        Einv = np.zeros( (0,0) )
    AWE = np.dot(AW, Einv)
    # Apply preconditioner to AWE (I don't see a way to get rid of this! -- André).
    MAWE = _apply(M, AWE)
    D = inner_product(AWE, MAWE)
    D1 = np.eye(nW)
    D2 = np.dot(Einv, B1)
    CC = np.r_[ np.c_[D, D1, D2],
                np.c_[D1.T.conj(), np.eye(nW), np.zeros( (nW,nVfull))],
                np.c_[D2.T.conj(), np.zeros((nVfull,nW)), np.eye(nVfull)]
              ]
    for i in range(0,ritzmat.shape[0]):
        w = U[0:W.shape[1],i]
        v = U[W.shape[1]:,i]
        mu = lam[i]

        z = np.r_[mu*w, -mu*w, -np.dot(B.T.conj(), w), Tfull[-1,-1]*v[-1]]
        z = np.reshape(z, (z.shape[0],1))
        CCz = np.dot(CC, z)
        res_ip = _ipstd(z, CCz)[0,0]
        assert(res_ip.imag < 1e-13)
        assert(res_ip.real > -1e-10)
        norm_ritz_res[i] = np.sqrt(abs(res_ip))

        # Explicit computation of residual (this part only works for M=I)
        #X = np.c_[W, Vfull[:,0:-1]]
        #V = np.dot(X, U)
        #AV = _apply(A, V)
        #res_explicit = AV[:,i] - lam[i]*V[:,i]
        #print np.linalg.norm(res_explicit)

    # Sort Ritz values/vectors and residuals s.t. residual is ascending.
    sorti = np.argsort(norm_ritz_res)
    ritz_vals = lam[sorti]
    ritz_vecs = np.dot(W, U[0:nW,sorti]) \
              + np.dot(Vfull[:,0:-1], U[nW:,sorti])
    norm_ritz_res  = norm_ritz_res[sorti]

    return ritz_vals, ritz_vecs, norm_ritz_res

# ==============================================================================
def gmres_wrap( linear_operator,
                b,
                x0,
                tol = 1.0e-5,
                maxiter = None,
                Mleft = None,
                Mright = None,
                inner_product = _ipstd,
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
           inner_product = _ipstd,
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
        u  = _apply(Mright, np.dot(V[:, :k+1], yy))
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
        norm_r = _norm(r, inner_product=inner_product)
    else:
        x0 = np.zeros( (N,1) )
        r    = MleftB
        norm_r = norm_MleftB

    V[:, [0]] = r / norm_r
    norm_rel_residual  = norm_r / norm_MleftB
    # Right hand side of projected system:
    y = np.zeros( (maxiter+1,1), dtype=xtype )
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
        H[k+1, k] = _norm(V[:, [k+1]], inner_product=inner_product)
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
                    print 'No convergence! expl. res = %e >= tol =%e in last it. %d (upd. res = %e)' \
                        % (norm_rel_residual, tol, k, norm_ur)
                    info = 1
                else:
                    print 'Expl. res = %e >= tol = %e > upd. res = %e in it. %d' \
                        % (norm_rel_residual, tol, norm_ur, k)

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
        u = np.sqrt(1 + t.real**2 + t.imag**2)
        c = t / u
        s = (b.conjugate()/absb) / u
        r = absb * u
    else:
        absa = abs(a)
        t = b.conjugate()/absa
        u = np.sqrt(1 + t.real**2 + t.imag**2)
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
            linear_solver = minres,
            forcing_term = 'constant',
            eta0 = 1.0e-1,
            eta_min = 1.0e-6,
            eta_max = 1.0e-2,
            alpha = 1.5, # only used by forcing_term='type 2'
            gamma = 0.9, # only used by forcing_term='type 2'
            use_preconditioner = False,
            deflate_ix = False
          ):
    '''Newton's method with different forcing terms.
    '''
    from scipy.constants import golden

    # some initializations
    error_code = 0
    k = 0

    x = x0
    Fx = model_evaluator.compute_f( x )
    Fx_norms = [ _norm( Fx, inner_product=model_evaluator.inner_product ) ]
    eta_previous = None
    linear_relresvecs = []
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

        # Setup linear problem.
        jacobian = model_evaluator.get_jacobian( x )
        initial_guess = np.zeros( (len(x),1) )
        rhs = -Fx

        if use_preconditioner:
            M = model_evaluator.get_preconditioner()
            Minv = model_evaluator.get_preconditioner_inverse()
        else:
            M = None
            Minv = None

        # Conditionally deflate the nearly-null vector i*x.
        if deflate_ix:
            W = 1j * x
            # normalize W in the M-norm
            MW = _apply(M, W)
            nrm_W = _norm(W, MW, inner_product = model_evaluator.inner_product)
            W = W / nrm_W
            AW = jacobian * W
            P, x0new = get_projection( W, AW, rhs, initial_guess,
                                       inner_product = model_evaluator.inner_product
                                     )
        else:
            x0new = initial_guess
            P = None
            W = np.zeros( (len(x),0) )
            AW = np.zeros( (len(x),0) )

        # Solve the linear system.
        out = linear_solver( jacobian,
                             rhs,
                             x0new,
                             Mr = P,
                             M = Minv,
                             tol = eta,
                             inner_product = model_evaluator.inner_product,
                             return_lanczos = True
                           )

        #ritz_vals, ritz_vecs, norm_ritz_res = get_ritz(W, AW, out[3], out[5],
                                                       #M = Minv,
                                                       #inner_product = model_evaluator.inner_product)

        #print ritz_vals

        # make sure the solution is alright
        assert( out[1] == 0 )

        # save the convergence history
        linear_relresvecs.append( out[2] )

        # perform the Newton update
        x += out[0]

        # do the household
        k += 1
        Fx = model_evaluator.compute_f( x )
        Fx_norms.append(_norm(Fx, inner_product=model_evaluator.inner_product))

    if k == maxiter:
        error_code = 1

    return x, error_code, Fx_norms, linear_relresvecs
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
