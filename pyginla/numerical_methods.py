#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Collection of numerical algorithms.
'''
# ==============================================================================
from scipy.sparse.linalg import LinearOperator, arpack
import numpy as np
from scipy.linalg import norm
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
def minres_wrap( linear_operator,
                 rhs,
                 x0,
                 tol = 1.0e-5,
                 maxiter = 1000,
                 xtype = None,
                 M = None
               ):
    '''
    Wrapper around the MINRES method to get a vector with the relative residuals
    as return argument.
    '''
    # --------------------------------------------------------------------------
    def _callback( x ):
        relresvec.append( norm(linear_operator * x - rhs) )
    # --------------------------------------------------------------------------

    relresvec = []

    rhs = np.ones( len(rhs) )
    sol, info = minres( linear_operator,
                        rhs,
                        x0,
                        shift = 0.0,
                        tol = tol,
                        maxiter = maxiter,
                        xtype = xtype,
                        M = M,
                        callback = _callback,
                        show = False,
                        check = False
                      )

    return sol, info, relresvec
# ==============================================================================
def minres( A,
            b,
            x0 = None,
            shift = 0.0,
            tol = 1e-5,
            maxiter = None,
            xtype = None,
            M = None,
            callback = None,
            show = False,
            check = False
          ):
    # --------------------------------------------------------------------------
    def SymOrtho( a, b ):
        aa = abs(a)
        ab = abs(b)
        if b == 0.:
            s = 0.
            r = aa
            if aa == 0.:
                c = 1.
            else:
                c = a/aa
        elif a == 0.:
            c = 0.
            s = b/ab
            r = ab
        elif ab >= aa:
            sb = 1
            if b < 0:
                sb = -1
            tau = a/b
            s = sb*(1+tau**2)**-0.5
            c = s*tau
            r = b/s
        elif aa > ab:
            print 'my regime'
            sa = 1
            if a < 0:
                sa=-1
            tau = b/a
            c = sa*(1+tau**2)**-0.5
            s = c*tau
            r = a/c
        return c,s,r
    # --------------------------------------------------------------------------

    show = False
    check = False
    eps = 2.2e-16

    n = len(b)
    if maxiter is None: maxiter = 5*n

    precon = True
    if M is None:
        precon = False

    if show:
        print '\n minres.m   SOL, Stanford University   Version of 10 May 2009'
        print '\n Solution of symmetric Ax = b or (A-shift*I)x = b'
        print '\n\n n      =%8g    shift =%22.14e' % (n,shift)
        print '\n maxiter =%8g    tol  =%10.2e\n'  % (maxiter,tol)

    istop = 0;   itn   = 0;   Anorm = 0;    Acond = 0;
    rnorm = 0;   ynorm = 0;   done  = False;
    x     = np.zeros( n )

    """
    %------------------------------------------------------------------
    % Set up y and v for the first Lanczos vector v1.
    % y  =  beta1 P' v1,  where  P = C**(-1).
    % v is really P' v1.
    %------------------------------------------------------------------
    """
    y     = +b;
    r1    = +b;
    if precon:
        M(y) # y = minresxxxM( M,b ); end
    beta1 = np.vdot(b, y)  # beta1 = b'*y;
    print "beta1: ", beta1

    """
    %  Test for an indefinite preconditioner.
    %  If b = 0 exactly, stop with x = 0.
    """
    if beta1 < 0:
        istop = 8
        show = True
        done = True
    if beta1 == 0:
        show = True
        done = True

    if beta1 > 0:
        beta1  = sqrt( beta1 );       # Normalize y to get v1 later.

    """
    % See if M is symmetric.
    """
    r2 = np.zeros( n )
    if check and precon:
        copy(r2, y)                     # r2     = minresxxxM( M,y );
        M( r2 )
        s = nrm2(y)**2                  # s      = y' *y;
        t = dotu(r1, r2)                # t      = r1'*r2;
        z = abs(s-t)                    # z      = abs(s-t);
        epsa = (s+eps)*eps**(1./3.)     # epsa   = (s+eps)*eps^(1/3);
        if z > epsa: istop = 7;  show = True;  done = True;
    # end if
    """
    % See if A is symmetric.
    """
    w = np.zeros( n )
    if check:
        A(y, w)                         # w    = minresxxxA( A,y );
        A(w, r2)                        # r2   = minresxxxA( A,w );
        s = nrm2(w)**2                  # s    = w'*w;
        t= dotu(y, r2)                  # t    = y'*r2;
        z = abs(s-t)                    # z    = abs(s-t);
        epsa = (s+eps)*eps**(1./3.)     # epsa = (s+eps)*eps^(1/3);
        if z > epsa: istop = 6;  done  = True;  show = True # end if
    # end if

    """
    %------------------------------------------------------------------
    % Initialize other quantities.
    % ------------------------------------------------------------------
    """
    oldb   = 0;       beta   = beta1;   dbar   = 0;       epsln  = 0;
    qrnorm = beta1;   phibar = beta1;   rhs1   = beta1;
    rhs2   = 0;       tnorm2 = 0;       ynorm2 = 0;
    cs     = -1;      sn     = 0;
    Arnorm = 0;

    w  = np.zeros( n )
    w2 = np.zeros( n )
    r2 = r1.copy() # r2     = r1;
    v  = np.zeros( n ) 
    w1 = np.zeros( n )

    if show:
        print ' '
        print ' '
        head1 = '   Itn     x[0]     Compatible    LS';
        head2 = '         norm(A)  cond(A)';
        head2 +=' gbar/|A|';  # %%%%%% Check gbar
        print head1 + head2

    print "v", v
    print "y", y
    """
    %---------------------------------------------------------------------
    % Main iteration loop.
    % --------------------------------------------------------------------
    """
    if not done:                     #  k = itn = 1 first time through
        while itn < maxiter:
            itn    = itn+1;
            """
            %-----------------------------------------------------------------
            % Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
            % The general iteration is similar to the case k = 1 with v0 = 0:
            %
            %   p1      = Operator * v1  -  beta1 * v0,
            %   alpha1  = v1'p1,
            %   q2      = p2  -  alpha1 * v1,
            %   beta2^2 = q2'q2,
            %   v2      = (1/beta2) q2.
            %
            % Again, y = betak P vk,  where  P = C**(-1).
            % .... more description needed.
            %-----------------------------------------------------------------
            """
            print "beta: ", beta
            s = 1 / beta;                 # Normalize previous vector (in y).
            print "s: ", s
            """
            v = s*y;                    # v = vk if P = I
            y = minresxxxA( A,v ) - shift*v;
            if itn >= 2, y = y - (beta/oldb)*r1; end
            """
            v = s * y.copy()
            print "v", v
            y = A * v
            if abs(shift) > 0:
                y -= shift*v
            if itn >= 2:
                y -= beta/oldb * r1

            print "v", v
            print "y", y
            alfa = np.vdot( v, y ) # alphak
            print "r2", r2
            y   -= alfa/beta * r2  # y    = (- alfa/beta)*r2 + y;
            print "yy ", y

            # r1     = r2;
            # r2     = y;
            r1 = y.copy()
            _y = r1
            r1 = r2
            r2 = y
            y  = _y
            print "r22 ", r2

            if precon:
                 y = M * r2     # y = minresxxxM( M,r2 ); # end if
            oldb   = beta;              # oldb = betak
            beta   = np.vdot( r2, y )    # beta = betak+1^2
            if beta < 0:
                istop = 6
                break
            beta   = sqrt(beta)
            tnorm2 = tnorm2 + alfa**2 + oldb**2 + beta**2

            if itn == 1:                # Initialize a few things.
                if beta/beta1 < 10*eps: # beta2 = 0 or ~ 0.
                    istop = -1         # Terminate later.
                # %tnorm2 = alfa**2  ??
                gmax   = abs(alfa)      # alpha1
                gmin   = gmax           # alpha1
            """
            % Apply previous rotation Qk-1 to get
            %   [deltak epslnk+1] = [cs  sn][dbark    0   ]
            %   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
            """
            oldeps = epsln
            delta  = cs*dbar + sn*alfa  # delta1 = 0         deltak
            print "sn, dbar, cs, alfa ", sn, dbar, cs, alfa
            gbar   = sn*dbar - cs*alfa  # gbar 1 = alfa1     gbar k
            epsln  =           sn*beta  # epsln2 = 0         epslnk+1
            dbar   =         - cs*beta  # dbar 2 = beta2     dbar k+1
            root   = sqrt(gbar**2 + dbar**2)
            Arnorm = phibar*root;       # ||Ar{k-1}||
            """
            % Compute the next plane rotation Qk
            gamma  = norm([gbar beta]); % gammak
            gamma  = max([gamma eps]);
            cs     = gbar/gamma;        % ck
            sn     = beta/gamma;        % sk
            """
            print "gbar, beta" , gbar, beta
            cs,sn,gamma = SymOrtho( gbar, beta )
            print "sn: ", sn
            phi    = cs * phibar ;      # phik
            phibar = sn * phibar ;      # phibark+1

            print 'phibar', phibar

            """
            % Update  x.
            """
            denom = 1 / gamma;
            """
            w1    = w2;
            w2    = w;
            w     = (v - oldeps*w1 - delta*w2)*denom;
            x     = x + phi*w;
            """
            w1 = w.copy()
            _w = w1
            w1 = w2
            w2 = w
            w  = _w

            w = denom * ( v - oldeps*w1 - delta*w2 )
            x += phi*w

            if callback is not None:
                callback( x )
            """
            % Go round again.
            """
            gmax   = max( gmax, gamma );
            gmin   = min( gmin, gamma );
            z      = rhs1 / gamma;
            # ynorm2 = z**2  + ynorm2;
            ynorm2 = np.vdot( x, x )
            #rhs1   = rhs2 - delta*z;
            #rhs2   =      - epsln*z;
            """
            % Estimate various norms.
            """
            Anorm  = sqrt( tnorm2 )
            ynorm  = sqrt( ynorm2 )
            epsa   = Anorm*eps
            epsx   = Anorm*ynorm*eps
            epsr   = Anorm*ynorm*tol
            diag   = gbar
            if diag==0:
                diag = epsa

            qrnorm = phibar
            rnorm  = qrnorm
            test1  = rnorm / (Anorm*ynorm) #  ||r|| / (||A|| ||x||)
            test2  = root  / Anorm # ||Ar{k-1}|| / (||A|| ||r_{k-1}||)
            """
            % Estimate  cond(A).
            % In this version we look at the diagonals of  R  in the
            % factorization of the lower Hessenberg matrix,  Q * H = R,
            % where H is the tridiagonal matrix from Lanczos with one
            % extra row, beta(k+1) e_k^T.
            """
            Acond  = gmax / gmin;
            """
            % See if any of the stopping criteria are satisfied.
            % In rare cases, istop is already -1 from above (Abar = const*I).
            """
            if istop==0:
                t1 = 1 + test1;       # These tests work if tol < eps
                t2 = 1 + test2;
                if t2    <= 1      :istop = 2; # end if
                if t1    <= 1      :istop = 1; # end if
                if itn   >= maxiter :istop = 5; # end if
                if Acond >= 0.1/eps:istop = 4; # end if
                if epsx  >= beta1  :istop = 3; # end if
                if test2 <= tol   :istop = 2; # end if
                if test1 <= tol   :istop = 1; # end if
            # end if
            """
            % See if it is time to print something.
            """
            prnt   = False;
            if n      <= 40       : prnt = True; # end if
            if itn    <= 10       : prnt = True; # end if
            if itn    >= maxiter-10: prnt = True; # end if
            if itn%10 == 0        : prnt = True  # end if
            if qrnorm <= 10*epsx  : prnt = True; # end if
            if qrnorm <= 10*epsr  : prnt = True; # end if
            if Acond  <= 1e-2/eps : prnt = True; # end if
            if istop  !=  0       : prnt = True; # end if

            if show and prnt:
                str1 = '%6g %12.5e %10.3e' % ( itn, x[0], test1 );
                str2 = ' %10.3e'           % ( test2 );
                str3 = ' %8.1e %8.1e'      % ( Anorm, Acond );
                str3 +=' %8.1e'            % ( gbar/Anorm);
                print str1, str2, str3
            # end if
            if abs(istop) > 0: break;        # end if
        # end while % main loop
    # end % if ~done early
    """
    % Display final status.
    """
    if show:
        print " "
        print ' istop   =  %3g               itn   =%5g'% (istop,itn)
        print ' Anorm   =  %12.4e      Acond =  %12.4e' % (Anorm,Acond)
        print ' rnorm   =  %12.4e      ynorm =  %12.4e' % (rnorm,ynorm)
        print ' Arnorm  =  %12.4e' % Arnorm
        print msg[istop+2]
    #return x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm
    return x, 0
# ==============================================================================
def cg_wrap( linear_operator,
             rhs,
             x0,
             tol = 1.0e-5,
             maxiter = 1000,
             xtype = None,
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
            errorvec.append( sqrt(inner_product(error,error)) )
    # --------------------------------------------------------------------------

    relresvec = []
    errorvec = []

    sol, info = cg( linear_operator,
                    rhs,
                    x0,
                    tol = tol,
                    maxiter = maxiter,
                    xtype = xtype,
                    M = M,
                    callback = _callback,
                    explicit_residual = explicit_residual,
                    inner_product = inner_product
                  )

    return sol, info, relresvec, errorvec
# ==============================================================================
def cg( linear_operator,
        rhs,
        x0,
        tol = 1.0e-5,
        maxiter = 1000,
        xtype = None,
        M = None,
        callback = None,
        explicit_residual = False,
        inner_product = np.vdot
      ):
    '''Conjugate gradient method with different inner product.
    '''
    # --------------------------------------------------------------------------
    def _compute_rho( r ):
        '''Update the value for rho.
        '''
        info = 0

        if M is not None:
            z = M * r
            rho = inner_product( z, r )
            #if abs(rho.imag) > abs(rho) * 1.0e-10:
                #return rho, maxiter, z
            assert rho.imag == 0.0
            rho = rho.real
            # M needs to be positive definite such that rho > 0.
            # It may happen by roundoff error, though, that
            # -10^{-8} < rho < 0.
            #if rho < -1.0e-8:
                #return rho, maxiter, z
            #elif rho < 0.0:
                #rho = abs( rho )
        else:
            z = None
            rho = inner_product( r, r )
            assert rho.imag == 0.0
            rho = rho.real

        return rho, info, z
    # --------------------------------------------------------------------------
    x = x0.copy()
    r = rhs - linear_operator * x

    rho_old, info, z = _compute_rho( r )
    if info != 0:
        return x, info

    if M is not None:
        p = z.copy()
    else:
        p = r.copy()

    info = maxiter

    # store the initial residual
    rho0 = rho_old

    if callback is not None:
        callback( x, sqrt(rho_old / rho0) )

    for k in xrange( maxiter ):
        Ap = linear_operator * p

        # update current guess and residial
        alpha = rho_old / inner_product( p, Ap )
        x += alpha * p
        if explicit_residual:
            r = rhs - linear_operator * x
        else:
            r -= alpha * Ap

        rho_new, info, z = _compute_rho( r )
        if info != 0:
            return x, info

        relative_rho = sqrt(rho_new / rho0)
        if relative_rho < tol:
            # Compute exact residual
            r = rhs - linear_operator * x
            rho_new, info, z = _compute_rho( r )
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
