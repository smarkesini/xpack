import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os
GPU=True
GPU=False
import fubini

from timeit import default_timer as timer


if GPU:
    import cupy as xp
    print("GPU code")
else:
    xp=np
    print("CPU code")
   
    

def read_h5(file_name,dirname="data"):   
    import h5py
    with h5py.File(file_name, "r") as f:
        print("reading from:", dirname)

        data = f[dirname][:]

    print(data.shape)
    return data
 
#xp=np


#scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
#rescale = lambda x,y: scale(x,y)*x
#ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
#ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
rescale = lambda x,y: scale(x,y)*x
ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
ssnr2    = lambda x,y: ssnr(x,y)**2



from testing_setup import setup_tomo
from fubini import radon_setup as radon_setup


#obj_size = 1024*2//8
#num_slices = 32# size//2
#num_angles =    32*2 #obj_size*2
#num_rays   = obj_size
#obj_width=0.95


#sim_400_1024_2048_95
#/sim_40_1024_2048_95 

obj_size = 1024*2
num_slices = 40# size//2
num_angles =    obj_size//2
num_rays   = obj_size
obj_width=0.95


file_name="/data/tomosim/shepp_logan.h5"
grp="sim_{}_{}_{}_{}".format(num_slices,num_angles,num_rays,int(obj_width*100))
dname_tomo="{}/tomo".format(grp)
dname_sino="{}/sino".format(grp)
dname_theta="{}/theta".format(grp)


# tomo to cropped image
#t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
t2i = lambda x: x[num_slices//2,:,:].real
# vector to tomo
v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))

#print("setting up the phantom, ", end = '')
print("setting up the phantom,...")
start=timer()
#true_obj, theta=setup_tomo(num_slices, num_angles, num_rays, xp, width=obj_width)

true_obj=read_h5(file_name,dirname=dname_tomo)
theta=read_h5(file_name,dirname=dname_theta)


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
radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, filter_type = 'hamming',kernel_type = 'gaussian', k_r =1,width=obj_width)
#radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, filter_type = 'ram-lak',kernel_type = 'gaussian', k_r =1,width=obj_width)
#radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, kernel_type = 'kb', k_r =1, width=obj_width)
#end = timer()
time_radonsetup=(end - start)
print("time=", time_radonsetup)


#print("warm up radon. ", end = '')
#start=timer()
#data = radon(true_obj[num_slices//2:num_slices//2+1,:,:])
#end = timer()
#time_radon=(end - start)
#print("time=", time_radon)

print("doing radon. ", end = '')
start=timer()
#data = radon(true_obj)
data= read_h5(file_name,dirname=dname_sino)
end = timer()
time_radon=(end - start)
print("time=", time_radon)


#
#print("warmup iradon. ", end = '')
#start=timer()
#tomo0=iradon(data[num_slices//2:num_slices//2+1,:,:])
#end = timer()
#time_iradon=(end - start)
#print(" time=", time_iradon)

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
#snr_iradon  = ssnr(tomo0c,truth) 
snr_iradon  = ssnr(tomo0,true_obj) 

print("radon time=", time_radon, "iradon  time=", time_iradon, "snr=", snr_iradon)
if GPU:
    data=xp.asnumpy(data)
    tomo0c=xp.asnumpy(tomo0c)

plt.imshow(data[num_slices//2,:,:])
plt.show()
#plt.imshow(tomo0c)
s="snr : %g" %(snr_iradon)
print(s)
#plt.imshow(np.abs(tomo0c)**.1)
plt.imshow(np.abs(tomo0c))

plt.title(s)
plt.show()
print(s)
#plt.imshow(data[num_slices//2,:,:])

"""
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
