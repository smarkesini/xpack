import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
#import tomopy
#import imageio
#import os
GPU=True
GPU=False
import fubini


if GPU:
    import cupy as xp
else:
    xp=np



scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

from testing_setup import setup_tomo
from fubini import radon_setup as radon_setup

size = 64*2
num_slices = size//2
num_angles = 27
num_rays   = size
# tomo to image
t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
# vector to tomo
v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))


true_obj, theta=setup_tomo(num_slices, num_angles, num_rays, xp)
radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, kernel_type = 'gaussian', k_r =1)


data = radon(true_obj)

# check iradon as initial
tomo0=iradon(data)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
factor=scale(tomo0,true_obj)

from solve_sirt import sirtBB
#msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
#T1,S1=sirtMcalc(radon,radont,xp.shape(data),xp)

#iradon1= lambda x: T1*radont(S1*x)
 
#image = sirt(data, theta, num_rays, k_r, kernel_type, gpu_accelerated, 100, factor=factor)
start = timer()
tomo_sirtBB,rnrm = sirtBB(radon, radont, data, xp, max_iter=20, alpha=1.,verbose=0,useRC=True)
end = timer()
time_sirtBB=end-start

start = timer()
tomo_psirtBB,rnrmp = sirtBB(radon, iradon, data, xp, max_iter=20, alpha=1.,verbose=0)
end = timer()
time_psirtBB=end-start

print("sirtBB  time=", time_sirtBB, "rnrm=", rnrm)
print("psirtBB time=", time_psirtBB, "rnrm=", rnrmp)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
scalem   = lambda x,y: xp.max(y)/xp.max(x *x)

ssirt=scalem(t2i(tomo_sirtBB),t2i(tomo0))
ssirtp=scalem(t2i(tomo_psirtBB),t2i(tomo0))

img=xp.concatenate((t2i(tomo0),ssirt*t2i(tomo_sirtBB),ssirtp*t2i(tomo_psirtBB)),axis=1)
if GPU:
     img=xp.asnumpy(img)

plt.imshow(img)
plt.title('iradon vs sirt')
plt.show()


scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

print("sirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo_sirtBB))
print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))