#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:57:52 2019

@author: anu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:14:36 2019

@author: anu
"""

import gridrec
import numpy as np
import matplotlib.pyplot as plt
import tomopy
import h5py
import dxchange
import sys
from scipy import special
import time
from sklearn.metrics import mean_squared_error
import subprocess
import operator
import os

#from mpi4py import *
gpu_accelerated = False

def mpi_gather(send_buff):

    if MPI.COMM_WORLD.Get_size() > 1:
        recv_buff = MPI.COMM_WORLD.gather(send_buff)
    else:
        recv_buff = [send_buff]

    return np.array(recv_buff)

def get_chunk_slice(n_slices):
    try:
        mpi
        rank = int(MPI.COMM_WORLD.Get_rank())
        chunk_size = n_slices/MPI.COMM_WORLD.Get_size()
            
        extra_work = 0
        if int(MPI.COMM_WORLD.Get_size()) == rank + 1:
            extra_work = n_slices % MPI.COMM_WORLD.Get_size()
         
    except:         
        rank = 0
        chunk_size = n_slices
        extra_work = 0
        return slice(int(0), int(n_slices))

            
    return slice(int(rank * chunk_size), int(((rank + 1) * chunk_size) + extra_work))

#Setup local GPU device based on a gpu_priority integer. The lower the priority, the more available gpu this assigns.
def set_visible_device(gpu_priority):
    if gpu_accelerated == False:        
        return 0, 0, 0

    cmd = "nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used,pstate,utilization.gpu --format=csv"
 
    result = str(subprocess.check_output(cmd.split(" ")))

    if sys.version_info <= (3, 0):
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split()[1:] # use this for python2
    else:
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split("\\n")[1:] # use this for python3


    result = [res.split(",") for res in result]
  
    if sys.version_info >= (3, 0):
        result = result[0:-1]

    result = [[int(r[0]), int(r[3])] for r in result]
    result = sorted(result, key=operator.itemgetter(1))

    n_devices = len(result)

    nvidia_device_order = [res[0] for res in result]

    visible_devices = str(nvidia_device_order[(gpu_priority % len(nvidia_device_order))])
      
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    return nvidia_device_order, visible_devices, n_devices

np.set_printoptions(threshold=sys.maxsize)

k_r = 2
try:
    mpi
    if(MPI.COMM_WORLD.Get_rank() == 0): base_folder = gridrec.create_unique_folder("shepp_logan")
except:
    base_folder = gridrec.create_unique_folder("shepp_logan")
    

size = 64

num_slices = size
num_angles = int(np.ceil(size//2*np.pi*2))
#num_angles = 512
num_rays   = size

true_obj_shape = (num_slices, num_rays, num_rays)

true_obj = gridrec.generate_Shepp_Logan(true_obj_shape)

pad_1D = num_rays//2
padding_array = ((0,0), (pad_1D, pad_1D), (pad_1D, pad_1D))

num_rays = num_rays + pad_1D*2

print("True obj shape", true_obj.shape)

#if(MPI.COMM_WORLD.Get_rank() == 0): gridrec.save_cube(true_obj, base_folder + "true")

#padding...
true_obj = np.lib.pad(true_obj, padding_array, 'constant', constant_values=0)

#angles in radians
#theta = np.arange(-90, 90., 180. / num_angles)*np.pi/180.
theta = np.arange(0, 180, 180. / num_angles)*np.pi/180.

simulation = gridrec.forward_project(true_obj, theta)


#The simulation generates a stack of projections, meaning, it changes the dimensions order
#if(MPI.COMM_WORLD.Get_rank() == 0): gridrec.save_cube(simulation, base_folder + "sim_project")

simulation = np.swapaxes(simulation,0,1)
plt.imshow(simulation[num_slices//2])
plt.show()

import numpy as xp
kernel_type = 'gaussian'
mode = "python"
simulation1 = gridrec.gridrec_transpose(true_obj[num_slices//2:num_slices//2+1], theta, num_rays, k_r, kernel_type, xp, mode)
#simulation1 = gridrec.gridrec_transpose(true_obj, theta, num_rays, k_r, kernel_type, xp, mode)
print("ratio gridrec_transpose i/r=", np.max(np.abs(np.imag(simulation1[0])))/np.max(np.real(simulation1[0])))
simulation1=simulation1.real
scaling_1_0=(np.sum(simulation * simulation1))/np.sum(simulation1 *simulation1)

#simulation = np.swapaxes(simulation,0,1)
plt.imshow(np.real(simulation1[0]))
plt.show()


#tomo_stack = tomopy.recon(simulation, theta, center=None, sinogram_order=True, algorithm="gridrec")
#sim1=simulation[num_slices//2:num_slices//2+1,:,num_rays//4:num_rays//4*3]
#tomo_stack = tomopy.recon(sim1, theta, center=None, sinogram_order=True, algorithm="gridrec")

tomo_stack = tomopy.recon(simulation[num_slices//2:num_slices//2+1], theta, center=None, sinogram_order=True, algorithm="gridrec", filter_name='ramlak')
plt.imshow(tomo_stack[0])
plt.show()


#gridrec(sinogram_stack, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj 
#tomo_stack1 = gridrec.gridrec(sim1, theta, num_rays//2, k_r,"gaussian", xp, "gridrec")
xp = np
sim1=simulation[num_slices//2:num_slices//2+1,:]
tomo_stack1 = gridrec.gridrec(sim1, theta, num_rays, k_r,"gaussian", xp, "gridrec")

#tomo_stack1 = gridrec.gridrec(sim1, theta, num_rays, k_r,kernel_type="gaussian", xp,  algorithm="gridrec")
plt.imshow((tomo_stack1[0]).real)

#plt.imshow((tomo_stack1[0,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]).real)
plt.show()

print("gridrec r/i=", np.max(abs(tomo_stack1.imag))/np.max(abs(tomo_stack1.real)))

tomo_stackc=tomo_stack[0,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
tomo_stack1c=(tomo_stack1[0,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]).real
tomo_stack0c=true_obj[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]

plt.imshow(tomo_stackc)
plt.show()
plt.imshow(tomo_stack1c.real)
plt.show()


#
scalingc=(np.sum(tomo_stack0c * tomo_stackc))/np.sum(tomo_stackc *tomo_stackc)
##scaling1c=(np.sum(tomo_stackc * tomo_stack1c))/np.sum(tomo_stack1c *tomo_stack1c)
scaling1c=(np.sum(tomo_stack0c * tomo_stack1c))/np.sum(tomo_stack1c *tomo_stack1c)
#tomo_stack1c*=scaling1c
#tomo_stackc*=scalingc
print("forward tomopy/nufft scaling=", scaling_1_0)

print("scaling tomopy/nufft",scaling1c)

snr0=np.sum(abs(tomo_stackc*scalingc-tomo_stack0c)**2)/np.sum(tomo_stack0c**2)
snr1=np.sum(abs(tomo_stack1c*scaling1c-tomo_stack0c)**2)/np.sum(tomo_stack0c**2)
print("snrs",1./snr0,1./snr1)


#
###scaling1c=(np.sum(tomo_stackc * tomo_stack1c))/np.sum(tomo_stack1c *tomo_stack1c)
#plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(tomo_stack1c)))**.5)
#
#
#tomo_stackc=(tomo_stack[0]).real
#tomo_stack1c=(tomo_stack1[0]).real
#tomo_stack0=true_obj[num_slices//2]
#
#scaling1c=(np.sum(tomo_stackc * tomo_stack1c))/np.sum(tomo_stack1c *tomo_stack1c)
#tomo_stack1c*=scaling1c



 
#import numpy as xp
#kernel_type = 'gaussian'
#mode = "python"
#simulation1 = gridrec.gridrec_transpose(true_obj[num_slices//4:num_slices//4+1], theta, num_rays, k_r, kernel_type, xp, mode)
##simulation1 = gridrec.gridrec_transpose(true_obj, theta, num_rays, k_r, kernel_type, xp, mode)
#
##simulation = np.swapaxes(simulation,0,1)
#plt.imshow(np.real(simulation1[0]))
#plt.show()
#print("ratio", np.max(np.abs(np.imag(simulation1[0])))/np.max(np.real(simulation1[0])))



#if(MPI.COMM_WORLD.Get_rank() == 0): gridrec.save_cube(simulation[64], base_folder + "sim_slice")
#
##Here I take only a subset of slices to not reconstruct the whole thing...
#sub_slice = 15
#
##Use this to test CPU multithreading:
## export MKL_NUM_THREADS=N_threads
#
##Calls backward projection
##tomo_stack = gridrec.gridrec(simulation[32:33], theta, num_rays, k_r, 'gaussian')
##tomo_stack = gridrec.gridrec(simulation[32:33], theta, num_rays, k_r, 'gaussian')
#
#start_time = time.time()
#
#
#chunk_slice = get_chunk_slice(num_slices)
#
#print(chunk_slice)
#
#try:
#    mpi
#    set_visible_device(MPI.COMM_WORLD.Get_rank())
#except:
#    set_visible_device(1)

    
#
#tomo_stack = gridrec.tomo_reconstruct(simulation[chunk_slice], theta, num_rays, k_r, 'gaussian', 'gridrec',gpu_accelerated)
##tomo_stack = tomopy.recon(simulation, theta, center=None, sinogram_order=True, algorithm="gridrec")
#
#print(tomo_stack.shape)
#
#if(MPI.COMM_WORLD.Get_size() > 0): print("Communicating results...")
#
#tomo_stack = mpi_gather(tomo_stack)
#
#if(MPI.COMM_WORLD.Get_rank() == 0):
#
#    tomo_stack = tomo_stack.reshape((num_slices, tomo_stack.shape[-2], tomo_stack.shape[-1]))
#
#    print(tomo_stack.shape)
#
#    print("A simulation of size", simulation.shape, "with kernel radius", k_r, "took", time.time() - start_time, "seconds to run")
#
#    if pad_1D > 0: tomo_stack = tomo_stack[:, pad_1D: -pad_1D, pad_1D: -pad_1D]
#
##Calls forward projection
##sino_stack = gridrec.gridrec_transpose(true_obj[64:65], theta, num_rays, k_r, 'gaussian')
##if pad_1D > 0: sino_stack = sino_stack[:, :, pad_1D: -pad_1D]
##plt.imshow(abs(sino_stack[0]),cmap='gray')
#
#    #Save images
#    gridrec.save_cube(abs(tomo_stack), base_folder + "rec")
#    #gridrec.save_cube(abs(sino_stack), base_folder + "sino")
#
#
#
#    plt.imshow(abs(tomo_stack[tomo_stack.shape[0]//2]),cmap='Greys')
#    plt.show()


#Calculates and plots mean squared error
"""
#image1 = true_obj[0, 64: -64, 64: -64]
#image2 = tomo_stack[0][0:128,0:128]

image1 = true_obj[64][200:328,200:328]
image2 = tomo_stack[0][:128,:128]

#image3 = simulation[64][0:128,0:128]
#image4 = sino_stack[0][0:128,0:128]

#np.save('image1.npy',image1)
#np.save('image2.npy',image2)
#np.save('image3.npy',image3)
#np.save('image4.npy',image4)

image2 = abs(image2)
#image4 = abs(image4)

image1 = (image1-image1.min())/(image1.max()-image1.min())
image2 = (image2-image2.min())/(image2.max()-image2.min())
#image3 = (image3-image3.min())/(image3.max()-image3.min())
#image4 = (image4-image4.min())/(image4.max()-image4.min())

diff1 = abs(image1**2 - image2**2)
#diff2 = image3**2 - image3**2

plt.imshow(diff1,cmap='Greys')
plt.colorbar()
plt.show()

#plt.imshow(diff2,cmap='Greys')
#plt.show()

mse_tomo = mean_squared_error(image1,image2)
print(mse_tomo)

#mse_sino = mean_squared_error(image3,image4)
#print(mse_sino)
"""



