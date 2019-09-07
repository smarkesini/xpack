#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:48:37 2019

@author: smarchesini
"""
# ========== TV regularization ===============
# let's do TV-reg by Split Bregman method


# define finite difference in 1D, -transpose 
D1   = lambda x,ax: np.roll(x,-1,axis=ax)-x
Dt1  = lambda x,ax: x-np.roll(x, 1,axis=ax)

# gradient in 3D
def Grad(x): return np.stack((D1(x,0),D1(x,1),D1(x,2))) 
# divergence
def Div(x):  return Dt1(x[0,:,:,:],0)+Dt1(x[1,:,:,:],1)+Dt1(x[2,:,:,:],2)
# Laplacian
#def Lap(x): return -6*x+np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)
def Lap(x): return Div(Grad(x))

# also need l1 operrator max(|a|-t,0)*sign(x)
def Pell1(x,reg): return xp.clip(np.abs(x)-reg,0,None)*np.sign(x)


# vector to tomo
#def v2t(x): np.reshape(x,(num_slices,num_rays,num_rays))
shape_tomo=(num_slices,num_rays  ,num_rays)
v2t = lambda x: np.reshape(x,(num_slices,num_rays,num_rays))

# we need RTR(x)+r*Lap as an operator acting on a vector
#def RtRpDtD (x):


#np.max(radont(data)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3])


# let's scale the Radon trasform so that RT (data)~1
Rsf=1./np.mean(radont(data)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3])

# scale the RT(data) accordingly and make it into a vector
RTdata=Rsf*radont(data).ravel()

# setup linear operator for CGS
def RTRpLap_setup(radon,radont,Lap, num_slices,num_rays,r):

    
    # add the two functions    
    RTRpLapf = lambda x: (Rsf*radont(radon(x))-r*Lap(x)).ravel()
    #
    RTRpLapfv= lambda x: RTRpLapf(np.reshape(x,shape_tomo))
    
    
    # now let's setup the operator
    from scipy.sparse.linalg import LinearOperator
    RRshape = (num_slices*num_rays*num_rays,num_slices*num_rays*num_rays)
    RTRpLap = LinearOperator(RRshape, dtype='float32', matvec=RTRpLapfv, rmatvec=RTRpLapfv)
    return RTRpLap




p=Grad(true_obj)
ii=np.where(np.abs(p)>0)
reg=0.7*np.min(np.abs(p[ii]))
print("reg",reg)

r = .8
#reg = 10e-3 
#mu = reg*r

# Setup R_T R x+ r* Laplacian(x)
RTRpLap = RTRpLap_setup(radon,radont,Lap, num_slices,num_rays,r)



# initial

#u=iradon(data).ravel()
u=tomocg.ravel()
Lambda=0

plt.imshow(tomocg[num_slices//2,:,:])
plt.title("CG init")
plt.show()

start=timer()
cgsmaxit=4

maxit=5
# tomo to image
t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real

start=timer()
for ii in range(1,maxit+1):
    
    p=Pell1(Grad(v2t(u))-Lambda,reg)
    
    u,info = cgs(RTRpLap, RTdata-r*Div(Lambda+p).ravel(),x0=u,tol=tolerance*10,maxiter=cgsmaxit) 
    
    Lambda = Lambda + (p-Grad(v2t(u)))
    
    #plt.imshow(v2t(u)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3])
    stitle = "TV iter=%d" %(ii)
    
    plt.imshow(v2t(u)[num_slices//2,:,:])    
    plt.title(stitle)
    plt.show()
    print(stitle)
    
end = timer()
TV_time=(end - start)

#D1   = lambda x,ax: np.roll(x,-1,axis=ax)-x

tomotv=v2t(u)

plt.imshow(t2i(tomotv))
plt.show()

fig, axs = plt.subplots(1,3)


axs[0].imshow(t2i(tomocg))
axs[0].set_title('cg-ls')
axs[1].imshow(t2i(v2t(u)))
axs[1].set_title('tv')
axs[2].imshow(t2i(true_obj))
axs[2].set_title('truth')
plt.show()
    
    
tomotvc=tomocg[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]

scalingtv=(np.sum(tomo_stack0c * tomotvc))/np.sum(tomotvc *tomotvc)
#snr_TV  =np.sum(tomo_stack0c**2)/np.sum(abs(tomotvc*scalingtv-tomo_stack0c)**2)
snr_TV  =np.sum(tomo_stack0c**2)/np.sum(abs(tomotvc-tomo_stack0c)**2)


print("tomopy rec time =%3.3g, \t snr=%3.3g " %( tomopy_time, snr_tomopy))
#print("tomopy rec time  = ",tomopy_time, "sim time", time_tomopy_forward, "srn", snr_tomopy)
print("spmv   rec time =%3.3g, \t snr=%3.3g"% (  spmv_time, snr_spmv))

print("cgls   rec time =%3.4g, \t snr=%3.3g"% ( cgls_time, snr_cgls))

print("TV     rec time =%3.4g, \t snr=%3.3g"% ( TV_time, snr_TV))
