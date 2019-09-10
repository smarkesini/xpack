#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:42:33 2019

@author: smarchesini
"""

# make ring artifacts: dead pixels


sino=simulation1+0.

tomo0=iradon(simulation1)
tmax=np.max(t2i(tomo0))

# random missing pixel
ndeadpix=2
deadpix=np.ones((num_slices*num_rays))
ii=numpy.random.randint(-num_rays//4/1.5,high=num_rays//4/1.5-1,size=(num_slices,ndeadpix))+num_rays//2
slicei=np.arange(num_slices)
slicei.shape=(64,1)
iii=slicei*num_rays+ii


deadpix[iii]=np.random.rand(ndeadpix)
deadpix[iii]=0
deadpix.shape=(num_slices,1,num_rays)

# fix central slice
deadpix[32,:,:]=1
deadpix[32,:,(110,170)]=0


sino*=deadpix

print("Ring removeal by TV")
plt.imshow(sino[num_slices//2,:,:])
plt.title("bad pixels")


#plt.imshow(tomo[num_slices//2,:,:])
#plt.title("bad pixels")


tomo=iradon(sino)


plt.imshow(sino[num_slices//2,:,:])
plt.show()
tomo=iradon(sino)
tmax
plt.imshow(np.concatenate((t2i(tomo0),t2i(tomo)),axis=1))
plt.clim(0,tmax)
plt.title('no ring vs ring')

# set up the tv
fradon=lambda x: deadpix*radon(x)
fradont=lambda x: radont(x*deadpix)

fdata = sino

# let's scale the Radon trasform so that RT (data)~1
Rsf=1./np.mean(radont(fdata)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3])

# scale the RT(data) accordingly and make it into a vector
RTdata=Rsf*fradont(fdata).ravel()

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


p=Grad(tomo)

reg=5e-3*np.max(np.abs(Grad(tomo)))

#p=Grad(true_obj)
#ii=np.where(np.abs(p)>0)
#reg=0.7*np.min(np.abs(p[ii]))
print("reg",reg)

r = 0.8
#reg = 10e-3 
#mu = reg*r

# Setup R_T R x+ r* Laplacian(x)
RTRpLap = RTRpLap_setup(fradon,fradont,Lap, num_slices,num_rays,r)



# initial

#u=iradon(data).ravel()
u=tomo.ravel()
Lambda=0

plt.imshow(tomo[num_slices//2,:,:])
plt.title(" init")
plt.show()

start=timer()
cgsmaxit=3

maxit=30
# tomo to image
t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real

start=timer()
for ii in range(1,maxit+1):
    
    p=Pell1(Grad(v2t(u))-Lambda,reg)
    
    u,info = cgs(RTRpLap, RTdata-r*Div(Lambda+p).ravel(),x0=u,tol=tolerance*10,maxiter=cgsmaxit) 
    
    Lambda = Lambda + (p-Grad(v2t(u)))
    
    #plt.imshow(v2t(u)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3])
    stitle = "TV iter=%d" %(ii)
    
    plt.imshow(t2i(v2t(u)))    
    plt.clim(0,tmax)
    plt.title(stitle)
    plt.show()
    print(stitle)
    
end = timer()
TV_time=(end - start)

#D1   = lambda x,ax: np.roll(x,-1,axis=ax)-x

tomotv=v2t(u)

tmax=np.max(t2i(tomotv))

#plt.imshow(t2i(tomotv))
plt.imshow(np.concatenate((t2i(tomotv),t2i(tomo)),axis=1))
plt.title("TV vs iradon")
plt.clim(0,tmax)
plt.show()


"""
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


"""


