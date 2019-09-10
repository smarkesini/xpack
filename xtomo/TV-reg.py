#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:48:37 2019

@author: smarchesini
"""
# ========== TV regularization ===============
#  TV-reg by Split Bregman method
#  min ||R x - data||^2+ mu ||Grad x||_1
#
# 

# define finite difference in 1D, and -transpose 
D1   = lambda x,ax: np.roll(x,-1,axis=ax)-x
Dt1  = lambda x,ax: x-np.roll(x, 1,axis=ax)

# gradient in 3D
def Grad(x): return np.stack((D1(x,0),D1(x,1),D1(x,2))) 
# divergence
def Div(x):  return Dt1(x[0,:,:,:],0)+Dt1(x[1,:,:,:],1)+Dt1(x[2,:,:,:],2)
# Laplacian
#def Lap(x): return -6*x+np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)
def Lap(x): return Div(Grad(x))

# soft thersholding ell1 operrator max(|a|-t,0)*sign(x)
def Pell1(x,reg): return xp.clip(np.abs(x)-tau,0,None)*np.sign(x)


plt.imshow(tomocg[num_slices//2,:,:])
plt.title("CG init")
plt.show()


# we need to solve RT R(x)+r*Lap(x)= RT(data)+r*Div(Lambda+p)
# we can precompute RT(data) for the r.h.s term

# let's scale the Radon trasform so that RT (data)~1 in the central slice
Rsf=1./np.mean(t2i(radont(data)))

# scale the RT(data) accordingly and make it into a vector
RTdata=Rsf*radont(data).ravel()

# we need RTR(x)+r*Lap(x) as an operator acting on a vector
RTRpLapt= lambda x: Rsf*radont(radon(x))- r*Lap(x)
RTRpLap = lambda x: RTRpLapt(v2t(x)).ravel() #i/o vectors

p=Grad(tomocg)
tau=0.06*np.max(np.abs(p)) # soft thresholding 
print("tau",tau)

r = .8     # regularization weight 
# initial
u=tomocg.ravel()
Lambda=0

cgsmaxit=4 # internal cg solver 
maxit=5    # TV iterations
verbose=True # pring and plot figures

start=timer()
for ii in range(1,maxit+1):
    
    # soft thresholding p
    p=Pell1(Grad(v2t(u))-Lambda,tau)
    
    # update tomogram
    u,info, imax, resnrm = cgs(RTRpLap, RTdata-r*Div(Lambda+p).ravel(),x0=u,tol=tolerance,maxiter=cgsmaxit)
    
    # update multiplier
    Lambda = Lambda + (p-Grad(v2t(u)))
    
    if verbose:
#        stitle = "TV iter=%d" %(ii)    
        title = "TV iter=%d, cgs(inf=%g,ii=%g,rnrm=%g)" %(ii,info,imax,resnrm)
        print(stitle)
        plt.imshow(v2t(u)[num_slices//2,:,:])    
        plt.title(stitle)
        plt.show()

    
end = timer()
TV_time=(end - start)



tomotv=v2t(u)

#plt.imshow(t2i(tomotv))
plt.imshow(np.concatenate((t2i(tomotv),t2i(tomocg)),axis=1))
plt.title("TV vs CG")
plt.show()

fig, axs = plt.subplots(1,3)


axs[0].imshow(t2i(tomocg))
axs[0].set_title('cg-ls')
axs[1].imshow(t2i(v2t(u)))
axs[1].set_title('tv')
axs[2].imshow(t2i(true_obj))
axs[2].set_title('truth')
plt.show()
    
    
tomotvc=t2i(tomotv)

#scalingtv=(np.sum(tomo_stack0c * tomotvc))/np.sum(tomotvc *tomotvc)
scalingtv=scale(tomotvc,tomo_stack0c)

#snr_TV  =np.sum(tomo_stack0c**2)/np.sum(abs(tomotvc*scalingtv-tomo_stack0c)**2)
snr_TV  = ssnr2(tomotvc,tomo_stack0c)#np.sum(tomo_stack0c**2)/np.sum(abs(tomotvc-tomo_stack0c)**2)
snr_cgls  = ssnr2(tomocgc,tomo_stack0c)



print("tomopy rec time =%3.3g, \t snr=%3.3g " %( tomopy_time, snr_tomopy))
#print("tomopy rec time  = ",tomopy_time, "sim time", time_tomopy_forward, "srn", snr_tomopy)
print("spmv   rec time =%3.3g, \t snr=%3.3g"% (  spmv_time, snr_spmv))

print("cgls   rec time =%3.4g, \t snr=%3.3g"% ( cgls_time, snr_cgls))

print("TV     rec time =%3.4g, \t snr=%3.3g"% ( TV_time, snr_TV))
