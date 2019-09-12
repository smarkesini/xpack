import gridrec
import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os
import gridrec
from timeit import default_timer as timer

def masktomo(num_rays,xp,width=.65):
    
    xx=xp.array([range(-num_rays//2, num_rays//2)])
    msk_sino=np.float32(np.abs(xx)<(num_rays//2*width))
    msk_sino.shape=(1,1,num_rays)
    
    xx=xx**2
    rr2=xx+xx.T
    msk_tomo=rr2<(num_rays//2*width*1.02)**2
    msk_tomo.shape=(1,num_rays,num_rays)
    return msk_tomo, msk_sino

def sirtMcalc(radon,radont,shape,xp):
    # compute the tomogram and sinogram of all ones
    # if R (radon) is a matrix sum_rows R = R*1= radon(1) 
    #                  sum_cols R = RT*1 = radont(1) 
    # we need to clip outside a mask of interest
    
    num_rays=shape[2]
    num_angles=shape[1]
    
    msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
    
    eps=xp.finfo(xp.float32).eps
    #t1=tomo0*0+1.
    S1=radon(np.ones((1,num_rays,num_rays),dtype='float32'))
    S1=1./np.clip(S1,eps*10,None)*msk_sino
    T1=radont(np.ones((1,num_angles,1),dtype='float32')*msk_sino)
    T1=1./np.clip(T1,eps,None)*msk_tomo
    return T1,S1


def sirtBB(radon, radont, sino_data, max_iter=30, alpha=1, verbose=0, useRC=False):

    #image = np.zeros((sino_data.shape[0], sino_data.shape[1], sino_data.shape[2]))
    xp=np
      
    nrm0 = xp.linalg.norm(sino_data)

    if useRC:
        C,R=sirtMcalc(radon,radont,xp.shape(data),xp)
        iradon= lambda x: C*radont(R*x)
    else: 
        iradon=radont

    tomo = iradon(sino_data)
    
    

    for i in range(max_iter):
        

        
        residual =  radon(tomo) - sino_data 
        rnrm=np.linalg.norm(residual)/nrm0
        
        if verbose >0:
            title = "SIRT-BB iter=%d, rnrm=%g" %(i, rnrm)
            print(title )
            if verbose >1:
                plt.imshow(t2i(tomo))
                plt.title(title)
                plt.show()

       
        grad = iradon(residual)

        # BB step  (alternating)       
        if i>0:
            if np.mod(i,6)<3:
                alpha=xp.linalg.norm(tomo-tomo_old)**2/xp.inner((tomo-tomo_old).ravel(),(grad-grad_old).ravel())
            else:
                alpha=xp.inner((tomo-tomo_old).ravel(),(grad-grad_old).ravel())/xp.linalg.norm(grad-grad_old)**2
#
        tomo_old=tomo+0
        
        
        tomo -=  grad*alpha
        
        grad_old=grad+0
        
    return tomo,rnrm


xp=np
scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

from testing_setup import setup_tomo

size = 64*2
num_slices = size//2
num_angles = 27
num_rays   = size
# tomo to image
t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
# vector to tomo
v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))


radon, iradon, radont, true_obj, data, theta=setup_tomo(num_slices, num_angles, num_rays, xp)

# check iradon as initial
tomo0=iradon(data)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
factor=scale(tomo0,true_obj)

#msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
#T1,S1=sirtMcalc(radon,radont,xp.shape(data),xp)

#iradon1= lambda x: T1*radont(S1*x)
 
#image = sirt(data, theta, num_rays, k_r, kernel_type, gpu_accelerated, 100, factor=factor)
start = timer()
tomo_sirtBB,rnrm = sirtBB(radon, radont, data, max_iter=30, alpha=1.,verbose=0,useRC=True)
end = timer()
time_sirtBB=end-start

start = timer()
tomo_psirtBB,rnrmp = sirtBB(radon, iradon, data, max_iter=30, alpha=1.,verbose=0)
end = timer()
time_psirtBB=end-start

print("sirtBB  time=", time_sirtBB, "rnrm=", rnrm)
print("psirtBB time=", time_psirtBB, "rnrm=", rnrmp)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
scalem   = lambda x,y: xp.max(y)/xp.max(x *x)

ssirt=scalem(t2i(tomo_sirtBB),t2i(tomo0))
ssirtp=scalem(t2i(tomo_psirtBB),t2i(tomo0))

plt.imshow(np.concatenate((t2i(tomo0),ssirt*t2i(tomo_sirtBB),ssirtp*t2i(tomo_psirtBB)),axis=1))
plt.title('iradon vs sirt')
plt.show()


scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

print("sirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo_sirtBB))
print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))