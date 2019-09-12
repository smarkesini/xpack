def cg(A,b, x0=0, maxiter=100, tol=1e-4):
    
    bnrm = xp.linalg.norm( b );
    
    if  ( bnrm == 0.0 ):
        bnrm = 1.0
    
    
    flag = 0

    if xp.isscalar(x0):
        r0=b
    else:
        r0=b-A(b)
    p=r0
    x = x0
    for ii in range(maxiter):
        a  = xp.inner(r0,r0)/ xp.inner(p,A(p))
        x += a*p
        r1 = r0 - a*A(p)
        if xp.linalg.norm(r1)/bnrm < tol:
            return  x, flag, ii, xp.linalg.norm(r1)
        b = xp.inner(r1,r1) / xp.inner(r0,r0)
        p = r1 + b*p
        r0 = r1
    print("reached maxit, residual norm=", xp.linalg.norm(r1))
    return x, flag, ii, xp.linalg.norm(r1)

# conjugate gradient squared
#%     Univ. of Tennessee and Oak Ridge National Laboratory
#%     October 1, 1993
#%     Details of this algorithm are described in "Templates for the
#%     Solution of Linear Systems: Building Blocks for Iterative
#%     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
#%     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
#%     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).

def cgs(A, b, x0=0, maxiter=100, tol=1e-4):
    bnrm2 = xp.linalg.norm( b );
    
    if  ( bnrm2 == 0.0 ):
        bnrm2 = 1.0
    
    # r = b - A(x);
    if xp.isscalar(x0):
        # x0==0:
        r=b
    else: 
        r = b - A(x0)
        
    res = xp.linalg.norm( r ) / bnrm2;
    x=x0
    
    r_tld = r;
    for ii in range(1,maxiter+1)  :
        rho = xp.inner(r_tld,r );
        if rho==0:  break
        
        if ii>1:
            beta = rho / rho_1
            u = r + beta*q
            p = u + beta*( q + beta*p )
        else:
            u = r+0;
            p = u+0;
        
        p_hat = p+0
        v_hat = A(p_hat);                     #% adjusting scalars
        alpha = rho / xp.inner(r_tld,v_hat )
        q = u - alpha*v_hat
        
        u_hat = (u+q);

        x = x + alpha*u_hat                 #% update approximation

        r = r - alpha*A(u_hat);
        res = xp.linalg.norm( r ) / bnrm2;           #% check convergence
        
        if ( res <= tol ): break

        rho_1 = rho;
    
    if (res <= tol):                      # converged
        flag =  0;
    elif ( rho == 0.0 ):                  # breakdown
        flag = -1;
    else:                            # no convergence
        flag = 1;
    
    return x, flag, ii, res

###############################################
# ========== TV regularization ===============
#
#  TV-reg by Split Bregman method
#
#  min ½ ‖ R u - data ‖² + µ ‖ ∇ u ‖₁
#
#  Augmented lagrangian:
#
#  L= µ ||p||₁+  ½ ‖ R u - data ‖² + r ⟨ p- ∇ u, Λ ⟩+ ½ r ‖ p-∇ u ‖²
#
#
#  1) p:    µ ||p||_1+ r/2 || p - r (∇ u-Λ) ||^2
#    
#     p      <-- max(|∇ u-Λ |-µ/r,0) sign(∇ u-Λ)
#
#  2) u: ½ ||R u - data||²  + r/2 || ∇ u - (p+Λ) ||²
#
#     u      <-- (Rᵗ R - r ∇ᵗ ∇) u = Rᵗ data - r ∇ᵗ (p+Λ)
#
#  3) Λ <-- Λ + p - ∇ u
#
#  Grad =  ∇, Div = - ∇ᵗ, Lap = ∆= - ∇ᵗ ∇ᵗ
#  τ=µ/r
#------------------------------------------


# define finite difference in 1D, and -transpose 
import numpy as xp
D1   = lambda x,ax: xp.roll(x,-1,axis=ax)-x
Dt1  = lambda x,ax: x-xp.roll(x, 1,axis=ax)

# gradient in 3D
def Grad(x): return xp.stack((D1(x,0),D1(x,1),D1(x,2))) 
# divergence
def Div(x):  return Dt1(x[0,:,:,:],0)+Dt1(x[1,:,:,:],1)+Dt1(x[2,:,:,:],2)
# Laplacian
def Δ(x): return -6*x+xp.roll(x,1,axis=0)+xp.roll(x,-1,axis=0)+xp.roll(x,1,axis=1)+np.roll(x,-1,axis=1)+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)
def Lap(x): return Div(Grad(x))

# soft thersholding ell1 operrator max(|a|-t,0)*sign(x)
def Pell1(x,τ): return xp.clip(xp.abs(x)-τ,0,None)*xp.sign(x)


def solveTV(radon,radont, data, r, tau, x0=0, tol=1e-2, maxiter=5, verbose=0):
    
    # verbose=1: text output, 2: graphic output
    
    num_slices=data.shape[0]
    num_rays = data.shape[2]
    shapetomo=(num_slices,num_rays  ,num_rays)
    
    v2t = lambda x: xp.reshape(x,(shapetomo))
    t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
    
    # (Rᵗ R + r ∇ᵗ ∇) ∆
    RTRpLapt= lambda x: radont(radon(x))- r*Lap(x)
    RTRpLap = lambda x: RTRpLapt(v2t(x)).ravel() #i/o vectors


    RTdata=radont(data).ravel()
    # scale the data
    Rsf=1./xp.mean(t2i(v2t(RTdata)))
    RTdata*=Rsf

    Lambda=0
    

    cgsmaxit=4 # internal cg solver 
 
    if xp.isscalar(x0):
        RTR = lambda x: xp.reshape(radont(radon(v2t(x))),(-1))
        #x0=radont(data)
        u,info, imax, resnrm = cgs(RTR, RTdata, maxiter=cgsmaxit)
    else: u=x0.ravel()
    
    for ii in range(1,maxiter+1):
        
        # soft thresholding p
        p=Pell1(Grad(v2t(u))-Lambda,tau)
        
        # update tomogram
        u,info, imax, resnrm = cgs(RTRpLap, RTdata-r*Div(Lambda+p).ravel(),x0=u,tol=tol,maxiter=cgsmaxit)
        
        # update multiplier
        Lambda = Lambda + (p-Grad(v2t(u)))
        
        if verbose>0:   
            stitle = "TV iter=%d, cgs(convergence=%g,ii=%g,rnrm=%g)" %(ii,info,imax,resnrm)
            print(stitle)
            
#            if verbose ==2:
#                plt.imshow(v2t(u)[num_slices//2,:,:])    
#                plt.title(stitle)
#                plt.show()

    # rescale
    u    *= 1./Rsf
    return v2t(u)
