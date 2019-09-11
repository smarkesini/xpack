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

import numpy
import gridrec

from timeit import default_timer as timer

#from timeit import timer as timer
#from time import process_time as timer
#from time import perf_counter as timer



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

k_r = 1
try:
    mpi
    if(MPI.COMM_WORLD.Get_rank() == 0): base_folder = gridrec.create_unique_folder("shepp_logan")
except:
    base_folder = gridrec.create_unique_folder("shepp_logan")
    

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))



size = 64*2

num_slices = size//2
#num_angles = int(180)

#num_angles = int(np.ceil(size//2*np.pi*2))

#num_angles = int(90)
#num_angles = int(45)

num_angles = int(24)
#num_angles = int(20)

#num_angles = 180
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
num_angles+=1
theta = np.linspace(0,180, num=num_angles)*np.pi/180

#theta = np.arange(0, 180, 180. / num_angles,dtype='float128')*np.pi/180.


#radon,iradon,radont=gridrec.radon_setup(num_rays, theta, center=None, xp=np, kernel_type = 'gaussian', k_r =1)
radon,iradon,radont=gridrec.radon_setup(num_rays, theta, center=num_rays//2, xp=np, kernel_type = 'gaussian', k_r =1)






start = timer()
simulation = gridrec.forward_project(true_obj, theta)
end = timer()
time_tomopy_forward=end-start
print("tomopy simulation time=", time_tomopy_forward)


#The simulation generates a stack of projections, meaning, it changes the dimensions order
#if(MPI.COMM_WORLD.Get_rank() == 0): gridrec.save_cube(simulation, base_folder + "sim_project")

#simulation = np.swapaxes(simulation,0,1)

plt.imshow(simulation[num_slices//2])
plt.show()

#simulations=np.roll(simulation,10,axis=2)
#np.roll(simulation[32],10)

import numpy as xp
kernel_type = 'gaussian'
mode = "python"
#simulation1 = gridrec.gridrec_transpose(true_obj[num_slices//2:num_slices//2+1], theta, num_rays, k_r, kernel_type, xp, mode)
#simulation1=radon(true_obj[num_slices//2:num_slices//2+1])

start = timer()
simulation1=radon(true_obj)
#msk_sino=np.ones(num_rays)
#msk_sino[0:num_rays//4]=0
#msk_sino[num_rays//4*3+4:]=0
#msk_sino.shape=(1,1,num_rays)
#simulation1*=msk_sino
#xx=xp.array([range(-num_rays//2, num_rays//2),])
#xx=xx**2
#rr2=xx+xx.T
#msk_tomo=rr2<(num_rays//4)**2
#msk_tomo.shape=(1,num_rays,num_rays)
#simulation1*=msk_sino

end = timer()
time_radon=end-start
#
#simulation1s=simulation1[num_slices//2:num_slices//2+1]
print("tomopy simulation time=", time_tomopy_forward)

simulation1s=radon(true_obj[num_slices//2:num_slices//2+1])[0]

#simulation1 = gridrec.gridrec_transpose(true_obj, theta, num_rays, k_r, kernel_type, xp, mode)
simulation1s=simulation1s.real
#scaling_1_0=(np.sum(simulation * simulation1s))/np.sum(simulation1s *simulation1s)
scaling_1_0=scale(simulation1s,simulation)

#simulation = np.swapaxes(simulation,0,1)
plt.imshow(np.real(simulation1s))
plt.show()


#tomo_stack = tomopy.recon(simulation, theta, center=None, sinogram_order=True, algorithm="gridrec")
#sim1=simulation[num_slices//2:num_slices//2+1,:,num_rays//4:num_rays//4*3]
#tomo_stack = tomopy.recon(sim1, theta, center=None, sinogram_order=True, algorithm="gridrec")

start = timer()
#tomo_stack = tomopy.recon(simulation[num_slices//2:num_slices//2+1], theta, center=None, sinogram_order=True, algorithm="gridrec", filter_name='ramlak')
tomo_stack = tomopy.recon(simulation, theta, center=None, sinogram_order=True, algorithm="gridrec", filter_name='ramlak')
end = timer()
tomopy_time=(end - start)
print("tomopy recon time=",tomopy_time)

simulation1=simulation1.real

start = timer()
tomo_stack_g = iradon(simulation1)
#tomo_stack_g = iradon(simulation[num_slices//2:num_slices//2+1])
end = timer()
spmv_time=(end - start)

#print("tomopy recon time=",tomopy_time)
print("spmv recon time  =",spmv_time)

tomo_stack = tomopy.recon(simulation[num_slices//2:num_slices//2+1], theta, center=None, sinogram_order=True, algorithm="gridrec", filter_name='ramlak')



#gridrec(sinogram_stack, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj 
#tomo_stack1 = gridrec.gridrec(sim1, theta, num_rays//2, k_r,"gaussian", xp, "gridrec")
xp = np
sim=simulation[num_slices//2:num_slices//2+1,:]
#sim1=simulation1[num_slices//2:num_slices//2+1,:]
#tomo_stack1 = gridrec.gridrec(sim1, theta, num_rays, k_r,"gaussian", xp, "gridrec")
#simulation1=simulation1

#tomo_stack1 = iradon(simulation1)
tomo_stack1 = tomo_stack_g
#tomo_stack1*=msk_tomo


#vector to tomo
v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))

# tomo to cropped image
t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
    
    

#tomo_stack1 = gridrec.gridrec(sim1, theta, num_rays, k_r,kernel_type="gaussian", xp,  algorithm="gridrec")
plt.imshow((tomo_stack1[num_slices//2]).real)

#plt.imshow((tomo_stack1[0,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]).real)
plt.show()

print("gridrec i/r=", np.max(abs(tomo_stack1.imag))/np.max(abs(tomo_stack1.real)))

tomo_stackc=tomo_stack[0,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
tomo_stack1c=(tomo_stack1[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]).real
tomo_stack0c=true_obj[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]

plt.imshow(tomo_stackc)
plt.show()
plt.imshow(tomo_stack1c.real)
plt.show()


#
#scalingc=(np.sum(tomo_stack0c * tomo_stackc))/np.sum(tomo_stackc *tomo_stackc)
scalingc=scale(tomo_stackc,tomo_stack0c)


##scaling1c=(np.sum(tomo_stackc * tomo_stack1c))/np.sum(tomo_stack1c *tomo_stack1c)
#scaling1c=(np.sum(tomo_stack0c * tomo_stack1c))/np.sum(tomo_stack1c *tomo_stack1c)
scaling1c=scale(tomo_stack1c,tomo_stack0c)

#tomo_stack1c*=scaling1c
#tomo_stackc*=scalingc
print("forward tomopy/nufft scaling=", scaling_1_0)

print("backward truth/nufft scaling=",scaling1c)

#snr_tomopy=np.sum(tomo_stack0c**2)/np.sum(abs(tomo_stackc*scalingc-tomo_stack0c)**2)
snr_tomopy=ssnr2(tomo_stack0c,tomo_stackc)

#snr_spmv  =np.sum(tomo_stack0c**2)/np.sum(abs(tomo_stack1c*scaling1c-tomo_stack0c)**2)
snr_spmv  = ssnr2(tomo_stack0c,tomo_stack1c)

#print("snr tomopy",snr_tomopy,snr_spmv)

#print("tomopy simulation time=", time_tomopy_forward)
#print("new simulation time=",time_radon)
print("tomopy sim time= %3.3g,\t rec time =%3.3g,\t snr=%3.3g " %(time_tomopy_forward, tomopy_time, snr_tomopy))

#print("tomopy rec time  = ",tomopy_time, "sim time", time_tomopy_forward, "srn", snr_tomopy)
print("spmv   sim time= %3.3g,\t rec time =%3.3g,\t snr=%3.3g"% ( time_radon, spmv_time, snr_spmv))

##======= CG-Least Squares
# solve by conjugate gradient  
# min_tomo || R tomo - data ||**2
# which is eqiova;ent to
#         R.T R tomo = R.T dat

data=simulation1

runfile('solvers.py')
runfile('cgls.py')


#plt.imshow(t2i(true_obj))
#plt.title('truth')
#plt.show()


runfile('TV-reg.py')

#runfile('rings.py')
factor=1

jk=sirt(sino_data, theta_array, num_rays, k_r, kernel_type, gpu_accelerated, max_iter, factor)

"""
print("setting up the CG-LS") 

# we are solving min_x ||R x-data||^2
# the gradient w.r.t x is:  R.T R x -R.T data
# setting the gradient to 0:  R.T R x = R.T data    

# let's setup R.T R as an operator
# setting radont(radon(x)) as a linear operator

def RTR_setup(radon,radont,num_slices, num_rays):
    
    # reshape vector to tomogram 
    v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))
    mradon2 = lambda x: xp.reshape(radont(radon(v2t(x))),(-1))
    
    # now let's setup the operator
    from scipy.sparse.linalg import LinearOperator
    RRshape = (num_slices*num_rays*num_rays,num_slices*num_rays*num_rays)
    RTR = LinearOperator(RRshape, dtype='float32', matvec=mradon2, rmatvec=mradon2)
    return RTR


RTR=RTR_setup(radon,radont,num_slices,num_rays)
# R.T data  (as a long vector)
RTdata =np.reshape(radont(data),(-1))

# initial guess (as a vector)
tomo0=np.reshape(iradon(data),(-1))
tolerance=np.linalg.norm(tomo0)*1e-4

from scipy.sparse.linalg import cgs as cgs
print("solving CG-LS")

start = timer()
#tomocg,info = cgs(RTR,RTdata,x0=tomo0,tol=tolerance) 
tomocg,info = cgs(RTR,RTdata,tol=tolerance) 
tomocg.shape=(num_slices,num_rays,num_rays)

end = timer()
cgls_time=(end - start)
print("cg time=",cgls_time)
# cropped
tomocgc=tomocg[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
scalingcgls=(np.sum(tomo_stack0c * tomocgc))/np.sum(tomocgc *tomocgc)
snr_cgls  =np.sum(tomo_stack0c**2)/np.sum(abs(tomocgc*scalingcgls-tomo_stack0c)**2)

plt.imshow(tomocgc)
plt.show()
"""

"""
print("tomopy rec time =%3.3g, \t snr=%3.3g " %( tomopy_time, snr_tomopy))
#print("tomopy rec time  = ",tomopy_time, "sim time", time_tomopy_forward, "srn", snr_tomopy)
print("spmv   rec time =%3.3g, \t snr=%3.3g"% (  spmv_time, snr_spmv))

print("cgls   rec time =%3.4g, \t snr=%3.3g"% ( cgls_time, snr_cgls))
"""

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
def Lap(x): return -6*x+np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)

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
    
    mradon2 = lambda x: Rsf*radont(radon(x))
    #Lapf = lambda x: Lap(x)

    # add the two functions    
    RTRpLapf = lambda x: (mradon2(x)-r*Lap(x)).ravel()
    #
    RTRpLapfv= lambda x: RTRpLapf(np.reshape(x,shape_tomo))
    
    
    # now let's setup the operator
    from scipy.sparse.linalg import LinearOperator
    RRshape = (num_slices*num_rays*num_rays,num_slices*num_rays*num_rays)
    RTRpLap = LinearOperator(RRshape, dtype='float32', matvec=RTRpLapfv, rmatvec=RTRpLapfv)
    return RTRpLap



r = 10e-1
reg = 10e-1 
#mu = reg*r

# Setup R_T R x+ r* Laplacian(x)
RTRpLap = RTRpLap_setup(radon,radont,Lap, num_slices,num_rays,r)

# also need max(|a|-t,0)*sign(x)
def Pell1(x,reg): return xp.clip(np.abs(x)+reg,0,None)*np.sign(x)


# initial

#u=iradon(data).ravel()
u=tomocg.ravel()

Lambda=0

start=timer()

maxit=10
for ii in range(maxit):
    
    p=Pell1(Grad(v2t(u))-Lambda,reg)
    
    u,info = cgs(RTRpLap, RTdata-r*Div(Lambda+p).ravel(),x0=u,tol=tolerance*10) 
    
    Lambda = Lambda +(p-Grad(v2t(u)))
    #plt.imshow(v2t(u)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3])
    plt.imshow(v2t(u)[num_slices//2,:,:])
    stitle = "TV iter=%d" %(ii)
    plt.title(stitle)
    plt.show()
    print(stitle)
    
end = timer()
TV_time=(end - start)

plt.imshow(v2t(u)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3])
plt.show()
    
    
tomotv=v2t(u)
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
## fix the dimension`None`s
#mD3    = lambda x:  np.reshape(  D3(np.reshape(x,(num_slices,num_rays,num_rays))),(-1))
#mDt3   = lambda x:  np.reshape( Dt3(np.reshape(x,(num_slices,num_rays,num_rays))),(-1))
#mDtD3  = lambda x:  np.reshape(DtD3(np.reshape(x,(num_slices,num_rays,num_rays))),(-1))


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

#plt.imshow(diff1,cmap='Greys')
#plt.colorbar()
#plt.show()

#plt.imshow(diff2,cmap='Greys')
#plt.show()

mse_tomo = mean_squared_error(image1,image2)
print(mse_tomo)

#mse_sino = mean_squared_error(image3,image4)
#print(mse_sino)
"""



