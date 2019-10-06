import numpy as np
import matplotlib.pyplot as plt
import h5py
    
#import tomopy
#import imageio
#import os
GPU=True
#GPU=False

algo='iradon'
#algo='sirt'
max_chunk_slice=128



#import tomopy
#GPU=False
#algo='tomopy-gridrec'
##max_chunk_slice=np.inf
##max_chunk_slice=512
##max_chunk_slice=50

 

from timeit import default_timer as timer

from xcale.devmanager import set_visible_device

from communicator import rank, gatherv, get_loop_chunk_slices, get_chunk_slices
from communicator import size as mpi_size


from mpi4py import MPI
comm = MPI.COMM_WORLD


    
if GPU:
    import cupy as xp
    do,vd,nd=set_visible_device(rank)
    #device_gbsize=xp.cuda.Device(vd).mem_info[1]/((2**10)**3)
    device_gbsize=xp.cuda.Device(0).mem_info[1]/((2**10)**3)
    #print("rank:",rank,"device:",vd, "gb memory:", device_gbsize)
    
    mode = 'cuda'
else:
    xp=np
    mode= 'cpu'
    device_gbsize=(128-32)/mpi_size # GBs used per rank, leave 32 GB for the rest


if rank==0: print("GPU: ", GPU,", algorithm",algo)

"""
obj_size = 1024*2
num_slices = 16# size//2
#num_angles =    obj_size//2
num_angles =  1501
#num_angles =    11

num_rays   = obj_size
rot_center = None


from simulate_data import get_data as gdata
def get_data(x,chunks=None):    
    return gdata(num_slices,num_rays,num_angles,obj_width,x,chunks=chunks) 
"""
from read_tomobank import get_data
# (1792, 1501, 2048)
data_shape = get_data('dims')
#print("data shape",data_shape)
num_slices = data_shape[0]
num_angles = data_shape[1]
num_rays   = data_shape[2]
rot_center = 1024


obj_width=0.95
max_iter = 20


#float_size=32/8; alg_tsize=4; alg_ssize=3
#slice_gbsize=num_rays*(num_rays*alg_tsize+num_angles*alg_ssize)*(float_size)/((2**10)**3)
#
## leave .5 gb and another 9*3 (kernel*3, 3 is data(complex+col+row) for sparse
#max_chunk_slice=np.int(np.floor((device_gbsize-.5)/slice_gbsize)-9*3*num_angles/num_rays)
#print("max chunk size", max_chunk_slice, 'device_gbsize',device_gbsize,"slice size",slice_gbsize,
#      "(device-2)/slice size",(device_gbsize-1)/slice_gbsize)


if rank==0: print("max chunk size", max_chunk_slice)

    


if rank==0: 
    print('reading up the angles, num angles', num_angles)

    start=timer()
    theta = get_data('theta')

    
    print("tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)

 
else:
    true_obj=None
    theta = np.empty(num_angles,dtype='float32')


# allocate result
tomo = None
# bcast theta
comm.Barrier()
comm.Bcast([theta,num_angles,MPI.FLOAT])
comm.Barrier()

#print("rank",rank,"theta dtype",theta.dtype,"theta",theta)

theta=xp.array(theta)

# set up radon
if rank==0:  print("setting up the solver. ", end = '')
start=timer()


if algo=='tomopy-gridrec':
    import tomopy
    reconstruct  = lambda data,verbose: (tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec"),None)
else:
    from fubini import radon_setup as radon_setup
    if algo=='iradon':
        iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width,iradon_only=True)

        reconstruct = lambda data,verbose:  (iradon(data),None)
    elif algo == 'sirtBB':
        radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
        del radont
        reconstruct = lambda data,verbose:  sirtBB(radon, iradon, data, xp, max_iter=max_iter, alpha=1.,verbose=verbose_iter)



end = timer()
time_radonsetup=(end - start)
if rank==0: print("time=", time_radonsetup)

# solver
from solve_sirt import sirtBB



slice_shape=(num_rays,num_rays)


loop_chunks=get_loop_chunk_slices(num_slices, mpi_size, max_chunk_slice )
if rank==0: print("nslices",num_slices,"mpi_size", mpi_size,"max_chunk",max_chunk_slice)
if rank==0:print("loop_chunks", loop_chunks)

#times={'scatt':0, 'c2g':0, 'radon':0 ,'solver':0, 'g2c':0, 'gather':0 }
times={'h5read':0, 'c2g':0,'solver':0, 'g2c':0, 'gather':0 }
times_loop=times.copy()


if rank == 0: tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
#    if rank == 0: tomo = np.empty((nslices,num_rays,num_rays),dtype = 'float32')

verboseall = True
verbose_iter= (1/5) * (rank == 0) # print every 5 iterations
verbose_iter= (1) * (rank == 0) # print every iterations

#print("verbose_iter",verbose_iter)
verbose= (rank ==0) and verboseall
###############################
for ii in range(loop_chunks.size-1):
    #if rank == 0: print("doing slices", loop_chunks[ii],loop_chunks[ii+1])
    nslices = loop_chunks[ii+1]-loop_chunks[ii]
    chunk_slices = get_chunk_slices(nslices) 


    #if verbose: print()
    if verbose: print( 'loop_chunk',ii,':', loop_chunks[ii:ii+2], "chunks",loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel(),)
    # if rank==0: print("chunk slices",chunk_slices)
    

    start = timer()
    # data = read_h5(file_name,dirname=dname_sino,chunks=chunk_slices[rank,:]+loop_chunks[ii])

    if verbose: print("reading slices:", loop_chunks[ii],"-",loop_chunks[ii+1]) 

    #if rank ==0: print("data")
    data = get_data('sino',chunks=chunk_slices[rank,:]+loop_chunks[ii])
    #if rank ==0: print("data type",type(data),"dtype",data.dtype)

    #data = get_data('sino',chunks=chunk_slices[rank,:]+loop_chunks[ii])
    end = timer()
    times['h5read']+=(end - start)

    start = timer()
    data=xp.array(data)
    end = timer()
    times['c2g']=(end - start)
    
    # scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
    # factor=scale(tomo0,true_obj)
    
     
    if verbose: print("reconstructing slices:", loop_chunks[ii],"-",loop_chunks[ii+1]) 
    start = timer()
#    verbose_iter=(verbose) 

    #print("rank no", rank, "chunks",chunk_slices[rank]+loop_chunks[ii])
    #tomo_sirtBB,rnrm = sirtBB(radon, radont, data, xp, max_iter=100, alpha=1.,verbose=verbose,useRC=True, BBstep=False)
    #tomo_sirtBB,rnrmp = sirtBB(radon, iradon, data, xp, max_iter=max_iter, alpha=1.,verbose=verbose_iter)
    #tomo_sirtBB =  iradon(data)
    tomo_chunk, rnrm =  reconstruct(data,verbose_iter)
    
    #tomo_chunk = tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec")
    times['solver']+=timer()-start
    #print("rank no", rank, "chunks",chunk_slices[rank]+loop_chunks[ii], "rnrm",rnrmp,"data shapes", data.shape)
    
    
    # gather from GPU to CPU
    start1 = timer()
    if GPU: tomo_chunk=xp.asnumpy(tomo_chunk)
    end1 = timer()
    times['g2c']+=end1-start1
    
    start = timer()
    if rank ==0: 
        ptomo = tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]
        
        #print("tomo_chunk shape",ptomo.shape)
    else:
        ptomo=None

   
 #   comm.Barrier()    
    gatherv(tomo_chunk,chunk_slices,data=ptomo)

    comm.Barrier()
    end = timer()

    times['gather']=end-start
    
    for ii in times:
        times_loop[ii]+=times[ii]
        


if rank>0:     quit()

#print("times last loop_chunk",times)

print("times full tomo", times_loop)



import fubini
msk_tomo,msk_sino=fubini.masktomo(num_rays, np, width=.95)
msk_tomo=msk_tomo[0,...]

t2i = lambda x: x[num_slices//2,:,:].real
tomo0c=t2i(tomo)*msk_tomo
plt.imshow(np.abs(tomo0c))
plt.show()



try:
    true_obj = get_data('tomo')
except:
    true_obj = None

if type(true_obj) == type(None): 
    print("no tomogram to compare")
    #quit()

print("phantom shape",true_obj.shape, "n_angles",num_angles, 'algorithm:', algo,"GPU:",GPU,"max_iter:",max_iter)
print("reading tomo, shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)


scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
rescale = lambda x,y: scale(x,y)*x
ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
ssnr2    = lambda x,y: ssnr(x,y)**2


print("solver time=", times_loop['solver'], "snr=", ssnr(true_obj,tomo))

#tomo0=tomo_chunk


#print("psirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo1))

## tomo to cropped image
#t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
## vector to tomo
#v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))

#print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))
