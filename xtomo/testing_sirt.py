# import gridrec
import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os

from timeit import default_timer as timer


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



tomo0=iradon(data)

from solve_sirt import sirtBB

#image = sirt(data, theta, num_rays, k_r, kernel_type, gpu_accelerated, 100, factor=factor)
start = timer()
tomo_sirt,rnrm = sirtBB(radon, radont, data, max_iter=20,verbose=1, useRC=True, BBstep=False)
end = timer()
time_sirt=end-start
print("sirt data time=", time_sirt, "rnrm",rnrm)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
scalem   = lambda x,y: xp.max(y)/xp.max(x *x)

ssirt=scalem(t2i(tomo_sirt),t2i(tomo0))
plt.imshow(np.concatenate((t2i(tomo0),ssirt*t2i(tomo_sirt)),axis=1))
plt.title('iradon vs sirt')
plt.show()

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

print("sirt    time=", time_sirt, "snr=", ssnr(true_obj,tomo_sirt))




#plt.imshow(np.concatenate((t2i(tomo0),t2i(tomo_sirt)),axis=1))
#plt.title('iradon vs sirt')

