import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os
GPU=True
#GPU=False
import fubini

from timeit import default_timer as timer


if GPU:
    import cupy as xp
    print("GPU code")
else:
    xp=np
    print("CPU code")

#xp=np


scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

from testing_setup import setup_tomo
from fubini import radon_setup as radon_setup


size = 1024*2
num_slices = 32# size//2
num_angles =    size//2
num_rays   = size

# tomo to cropped image
t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
# vector to tomo
v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))

print("setting up the phantom, ", end = '')
start=timer()
true_obj, theta=setup_tomo(num_slices, num_angles, num_rays, xp)
end = timer()
time_phantom=(end - start)
print("phantom setup time=", time_phantom)

# push to GPU (or do nothing)
true_obj=xp.array(true_obj)
theta=xp.array(theta)

#true_obj=true_obj[num_slices//2:num_slices//2+1,:,:]
#num_slices=1

print("setting up radon. ", end = '')
start=timer()
radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, kernel_type = 'gaussian', k_r =1)
end = timer()
time_radonsetup=(end - start)
print("time=", time_radonsetup)


print("warm up radon. ", end = '')
start=timer()
data = radon(true_obj[num_slices//2:num_slices//2+1,:,:])
end = timer()
time_radon=(end - start)
print("time=", time_radon)
print("doing radon. ", end = '')
start=timer()
data = radon(true_obj)
end = timer()
time_radon=(end - start)
print("time=", time_radon)



print("warmup iradon. ", end = '')
start=timer()
tomo0=iradon(data[num_slices//2:num_slices//2+1,:,:])
end = timer()
time_iradon=(end - start)
print(" time=", time_iradon)

print("doing iradon. ", end = '')
start=timer()
tomo0=iradon(data)
end = timer()
time_iradon=(end - start)
print("time=", time_iradon)


# compute snr
tomo0c=t2i(tomo0)
truth=t2i(true_obj)

scaling_iradon=scale(tomo0c,truth)
#(np.sum(truth * tomowcgc))/np.sum(tomowcgc *tomowcgc)
snr_iradon  = ssnr(tomo0c,truth) 

print("radon time=", time_radon, "iradon  time=", time_iradon, "snr=", snr_iradon)
if GPU:
    data=xp.asnumpy(data)
    tomo0c=xp.asnumpy(tomo0c)

"""
plt.imshow(data[num_slices//2,:,:])
plt.show()
plt.imshow(tomo0c)
plt.show()


plt.imshow(t2i(tomo0))
plt.title('iradon')
plt.show()

########################################
# plotting
#if GPU:
#     img=xp.asnumpy(data)
#else:
#     img=data
#     
#plt.imshow(img[num_slices//2,:,:])
#plt.show()
#print("=========iradon with ramlak filter=========")
#
#nn=xp.ones((num_rays//2,1))*xp.nan
#img=xp.concatenate((truth,nn,tomo0c*scaling_iradon),axis=1)
#
#if GPU:
#     img=xp.asnumpy(img)
#     #img=fubini.memcopy_to_host(img)
#plt.imshow(img)
#plt.title("truth vs iradon")
#plt.show()
#


"""
