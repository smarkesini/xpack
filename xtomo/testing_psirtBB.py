import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os
GPU=True
#GPU=False

from timeit import default_timer as timer

from xcale.communicator import mpi_Gather, rank, size
from xcale.devmanager import set_visible_device

def get_chunk_slices(n_slices):

    chunk_size =np.int( np.ceil(n_slices/size)) # ceil for better load balance
    nreduce=(chunk_size*(size)-n_slices)  # how much we overshoot the size
    start = np.concatenate((np.arange(size-nreduce)*chunk_size,
                            (size-nreduce)*chunk_size+np.arange(nreduce)*(chunk_size-1)))
    stop = np.append(start[1:],n_slices)

    start=start.reshape((size,1))
    stop=stop.reshape((size,1))
    slices=np.longlong(np.concatenate((start,stop),axis=1))
    return slices 

from mpi4py import MPI
comm = MPI.COMM_WORLD

def scatterv(data,chunk_slices,slice_shape):
    dspl=chunk_slices[:,0]
    cnt=np.diff(chunk_slices)
    sdim=np.prod(slice_shape)
    chunk_shape=(np.append(int(cnt[rank]),slice_shape))
    data_local=np.empty(chunk_shape,dtype='float32')
    comm.Scatterv([data,tuple(cnt*sdim),tuple(dspl*sdim),MPI.FLOAT],data_local)
    return data_local

def gatherv(data_local,chunk_slices,slice_shape): 

    cnt=np.diff(chunk_slices)
    sdim=np.prod(slice_shape)
    if rank==0:
        data = np.empty((num_slices,num_rays,num_rays),dtype='float32')
    else:
        data=None
        
    comm.Gatherv(sendbuf=[data_local, MPI.FLOAT],recvbuf=[data,(cnt*sdim,None),MPI.FLOAT],root=0)
    return data


    
if GPU:
    import cupy as xp
    set_visible_device(rank)
    mode = 'cuda'
else:
    xp=np
    mode= 'CPU'

if rank==0: print("mode = ", mode)
#import fubini


scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

from testing_setup import setup_tomo
from fubini import radon_setup as radon_setup


size_obj = 1024*2//32
num_slices = 8*16*4
num_angles =    size_obj//2
num_rays   = size_obj

#chunk_slice = get_chunk_slice(num_slices)

#chunk_slices,cnt,dspl = get_chunk_slices(num_slices)
chunk_slices = get_chunk_slices(num_slices)


if rank==0: print('setting up the phantom, ', end = '')
    

if rank==0: 
    start=timer()
    true_obj, theta=setup_tomo(num_slices, num_angles, num_rays, xp)
    end = timer()
    
    time_phantom=(end - start)
    if rank==0: print("phantom setup time=", time_phantom)

    #print("theta type",theta.dtype)
else:
    true_obj=None
    theta = np.empty(num_angles,dtype='float32')



start=timer()
# bcast theta
comm.Bcast(theta)
slice_shape=(num_rays,num_rays)
# scatter
truth=scatterv(true_obj,chunk_slices,slice_shape)


comm.Barrier()
end = timer()
time_mpi0=(end - start)
if rank==0: print("mpi-scatterv time=", time_mpi0)
#

truth=xp.array(truth)
theta=xp.array(theta)

if rank==0: print("shape tomo=", true_obj.shape, "n angles",num_angles, "cunk slice", chunk_slices[0] )

if rank==0:  print("setting up radon. ", end = '')
start=timer()
radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, kernel_type = 'gaussian', k_r =1)
end = timer()
time_radonsetup=(end - start)
if rank==0: print("time=", time_radonsetup)


#print("warm up radon. ", end = '')
#start=timer()
#data = radon(true_obj[0,:,:])
#end = timer()
#time_radon=(end - start)
#print("time=", time_radon)
#print("warmup iradon. ", end = '')
#start=timer()
#tomo0=iradon(data[num_slices//2:num_slices//2+1,:,:])
#end = timer()
#time_iradon=(end - start)
#print(" time=", time_iradon)


if rank==0: print("generating data with radon. ", end = '')
start=timer()
data = radon(truth)
end = timer()
time_radon=(end - start)
if rank==0: print("time=", time_radon)



#
#if rank==0: print("doing iradon. ", end = '')
#start=timer()
#tomo0=iradon(data)
#end = timer()
#time_iradon=(end - start)
#if rank==0: print("time=", time_iradon)



scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
#factor=scale(tomo0,true_obj)

from solve_sirt import sirtBB
#msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
#T1,S1=sirtMcalc(radon,radont,xp.shape(data),xp)

#iradon1= lambda x: T1*radont(S1*x)
 
#image = sirt(data, theta, num_rays, k_r, kernel_type, gpu_accelerated, 100, factor=factor)
start = timer()
verbose=np.int(rank==0)

#tomo_sirtBB,rnrm = sirtBB(radon, radont, data, xp, max_iter=20, alpha=1.,verbose=verbose,useRC=True)
tomo_sirtBB,rnrmp = sirtBB(radon, iradon, data, xp, max_iter=20, alpha=1.,verbose=verbose)

end = timer()

time_sirtBB=end-start
if rank==0: print("sirtBB  time=", time_sirtBB)


comm.Barrier()
start1 = timer()

#tomo_sirtBB = mpi_Gather(tomo_sirtBB,tomo_sirtBB,mode='cpu')

# gather
if GPU:
    tomo_sirtBB=xp.asnumpy(tomo_sirtBB)

end1 = timer()

tomo = gatherv(tomo_sirtBB,chunk_slices,slice_shape)


end = timer()
#time_sirtBB=end-start
time_mpi=end-start1
time_g2c=end1-start1

#print("rank:",rank)

if rank>0:
    quit()

scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
rescale = lambda x,y: scale(x,y)*x
ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
ssnr2    = lambda x,y: ssnr(x,y)**2

print("mpi+g2c gather time=", time_mpi, "GPU->CPU time:", time_g2c)

print("sirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo))

## tomo to cropped image
#t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
## vector to tomo
#v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))

#print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))