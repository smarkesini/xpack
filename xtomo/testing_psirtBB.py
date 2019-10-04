import numpy as np
import matplotlib.pyplot as plt
import h5py
    
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

def get_loop_chunk_slices(ns, ms, mc ):
    # ns: num_slices, ms=mpi_size, mc=max_chunk
    # loop_chunks size
    ls=np.int(np.ceil(ns/(ms*mc)))    
    # nreduce: how many points we overshoot if we use max_chunk everywhere
    nr=ls*mc*ms-ns
    #print(nr,ls,mc)
    # number of reduced loop_chunks:
    
    cr=np.ceil(nr/ms/ls)
    if nr==0:
        rl=0
    else:
        rl=np.int(np.floor((nr/ms)/cr))
    
    loop_chunks=np.concatenate((np.arange(ls-rl)*ms*mc,(ls-rl)*ms*mc+np.arange(rl)*ms*(mc-cr),[ns]))
    """
    print("rl={}, cr={}, lc={}".format(rl,cr,loop_chunks),flush=True)
    cr=0 # chunk reduction
    rl=ls+1 # number of reduced loop_chunks - start big
    while rl>ls:
        cr+=1 # reduce each chunk by one more
        rl= np.floor((nr/ms)/cr)
    print("rl={}, cr={}, lc={}".format(rl,cr,loop_chunks))
    #print("rl",rl,"cr",cr)
    """ 
    #loop_chunks=np.concatenate((np.arange(ls-rl)*ms*mc,(ls-rl)*ms*mc+np.arange(rl)*ms*(mc-cr),[ns]))
    return np.int64(loop_chunks)

from mpi4py import MPI
comm = MPI.COMM_WORLD

def scatterv(data,chunk_slices,slice_shape):
    if size==1: return data[chunk_slices[0,0]:chunk_slices[0,1],...]
    dspl=chunk_slices[:,0]
    cnt=np.diff(chunk_slices)
    sdim=np.prod(slice_shape)
    chunk_shape=(np.append(int(cnt[rank]),slice_shape))
    data_local=np.empty(chunk_shape,dtype='float32')

    #comm.Scatterv([data,tuple(cnt*sdim),tuple(dspl*sdim),MPI.FLOAT],data_local)
    
    # sending large messages
    mpichunktype = MPI.FLOAT.Create_contiguous(4).Commit()
    sdim=sdim* MPI.FLOAT.Get_size()//mpichunktype.Get_size()
    comm.Scatterv([data,tuple(cnt*sdim),tuple(dspl*sdim),mpichunktype],data_local)
    mpichunktype.Free()


    return data_local

def gatherv(data_local,chunk_slices, data = None): 
    if size==1: 
        if type(data) == type(None):
            data=data_local+0
        else:
            data[...] = data_local[...]
        return data

    cnt=np.diff(chunk_slices)
    slice_shape=data_local.shape[1:]
    sdim=np.prod(slice_shape)
    
    if rank==0 and type(data) == type(None) :
        tshape=(np.append(chunk_slices[-1,-1]-chunk_slices[0,0],slice_shape))
        data = np.empty(tuple(tshape),dtype=data_local.dtype)


    #comm.Gatherv(sendbuf=[data_local, MPI.FLOAT],recvbuf=[data,(cnt*sdim,None),MPI.FLOAT],root=0)
    
    # for large messages
    mpichunktype = MPI.FLOAT.Create_contiguous(4).Commit()
    #mpics=mpichunktype.Get_size()
    sdim=sdim* MPI.FLOAT.Get_size()//mpichunktype.Get_size()
    comm.Gatherv(sendbuf=[data_local, mpichunktype],recvbuf=[data,(cnt*sdim,None),mpichunktype])
    mpichunktype.Free()
    
    return data


def read_h5(file_name,dirname="data",chunks=None):   

    with h5py.File(file_name, "r") as f:
        #print("reading from:", dirname)
        if type(chunks)==type(None):
            data = f[dirname][:]
        else:
            data = f[dirname][chunks[0]:chunks[1],...]

    #print(data.shape)
    return data
 
    
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
    device_gbsize=(128-32)/size # GBs used per rank, leave 32 GB for the rest


if rank==0: print("GPU: ", GPU)
if rank==0: print("mode = ", mode)
#import fubini


scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

from testing_setup import setup_tomo
from fubini import radon_setup as radon_setup


obj_size = 1024*2//2
num_slices = 1600# size//2
num_angles =    obj_size//2
#num_angles =    11

num_rays   = obj_size
obj_width=0.95
max_iter = 20


#file_name="/data/tomosim/shepp_logan.h5"
#grp="sim_{}_{}_{}_{}".format(num_slices,num_angles,num_rays,int(obj_width*100))
#dname_tomo="{}/tomo".format(grp)
#dname_sino="{}/sino".format(grp)
#dname_theta="{}/theta".format(grp)


#float_size=32/8; alg_tsize=4; alg_ssize=3
#slice_gbsize=num_rays*(num_rays*alg_tsize+num_angles*alg_ssize)*(float_size)/((2**10)**3)
#
## leave .5 gb and another 9*3 (kernel*3, 3 is data(complex+col+row) for sparse
#max_chunk_slice=np.int(np.floor((device_gbsize-.5)/slice_gbsize)-9*3*num_angles/num_rays)
#print("max chunk size", max_chunk_slice, 'device_gbsize',device_gbsize,"slice size",slice_gbsize,
#      "(device-2)/slice size",(device_gbsize-1)/slice_gbsize)
max_chunk_slice=64


#max_chunk_slice=100

# 9 slices for the sparse matrix with 3x3 kernel
#print("max chunk size", max_chunk_slice)

#max_chunk_slice=110

if rank==0: print("max chunk size", max_chunk_slice)

    
from simulate_data import get_data

if rank==0: 
    #print('setting up the phantom, ', end = '')
    #print('reading up the phantom.',(num_slices,num_rays,num_rays),"num angles", num_angles)
    print('reading up the angles, num angles', num_angles)

    start=timer()
    #true_obj, theta=setup_tomo(num_slices, num_angles, num_rays, xp, width=obj_width)
#    print("theta1 type",theta1.dtype,type(theta1),"shape",theta1.shape)
#    print("theta1",theta1*180/np.pi)    
    
    try:             
        theta = get_data(num_slices,num_rays,num_angles,obj_width,'theta')
        
    except:
        print("simulating first",flush=True)
        from simulate_data import simulate
        simulate(num_slices,num_rays,num_angles,obj_width)
        theta = get_data(num_slices,num_rays,num_angles,obj_width,'theta')
    
    
    print("tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)

    """
    #theta=read_h5(file_name,dirname=dname_theta)
    #theta    = xp.arange(0., 180., 180. / num_angles,dtype='float32')*xp.pi/180.
    
    #print("theta type",theta.dtype,type(theta),"shape",theta.shape)
    #print("theta",theta*180/np.pi)
    #print("theta difference",theta-theta1)
    end = timer()
    time_phantom=(end - start)

    print("phantom setup time=", time_phantom)
    print("phantom shape",true_obj.shape, "n_angles",num_angles, "max_iter",max_iter)
        

    #print("theta type",theta.dtype)
    """

else:
    true_obj=None
    theta = np.empty(num_angles,dtype='float32')


# allocate result
tomo = None
# bcast theta
#comm.Barrier()
comm.Bcast([theta,MPI.FLOAT])
theta=xp.array(theta)

# set up radon

if rank==0:  print("setting up radon. ", end = '')
start=timer()
#radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, kernel_type = 'kb', k_r =1, width=obj_width)
radon,iradon,radont = radon_setup(num_rays, theta, xp=xp, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
end = timer()
time_radonsetup=(end - start)
if rank==0: print("time=", time_radonsetup)

# solver
from solve_sirt import sirtBB



slice_shape=(num_rays,num_rays)


loop_chunks=get_loop_chunk_slices(num_slices, size, max_chunk_slice )
if rank==0: print("nslices",num_slices,"mpi size",size,"max_chunk",max_chunk_slice)
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
    
    """
    # scatter the initial object
    
    #########################
    start=timer()
    comm.Barrier()
    truth=scatterv(true_obj,chunk_slices+loop_chunks[ii],slice_shape)
    #print("rank", rank, "chunks shapes", truth.shape)
    comm.Barrier()
    end = timer()
    times['scatt']+=(end - start)
    
    
    start=timer()
    truth=xp.array(truth)
    end = timer()
    times['c2g']=(end - start)
    
    #if rank==0: print("shape tomo=", true_obj.shape, "n angles",num_angles, "cunk slice", chunk_slices[0] )
    
    #if rank==0: print("generating data with radon. ", end = '')
    if verbose: print("generating data with radon.  slices:", loop_chunks[ii],"-",loop_chunks[ii+1])
    start=timer()
    data = radon(truth)
    end = timer()
    times['radon']=(end - start)
    del truth
    #########################
    #print("rank",rank,"chunks",chunk_slices[rank,:]+loop_chunks[ii])
    """
    start = timer()
    # data = read_h5(file_name,dirname=dname_sino,chunks=chunk_slices[rank,:]+loop_chunks[ii])
    data = get_data(num_slices,num_rays,num_angles,obj_width,'sino',chunks=chunk_slices[rank,:]+loop_chunks[ii])
    end = timer()
    times['h5read']+=(end - start)

    start = timer()
    data=xp.array(data)
    end = timer()
    times['c2g']=(end - start)
    
    # scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
    # factor=scale(tomo0,true_obj)
    
     
    start = timer()
#    verbose_iter=(verbose) 
    if verbose: print("reconstructing. slices:", loop_chunks[ii],"-",loop_chunks[ii+1]) 

    #print("rank no", rank, "chunks",chunk_slices[rank]+loop_chunks[ii])
    #tomo_sirtBB,rnrm = sirtBB(radon, radont, data, xp, max_iter=100, alpha=1.,verbose=verbose,useRC=True, BBstep=False)
    tomo_sirtBB,rnrmp = sirtBB(radon, iradon, data, xp, max_iter=max_iter, alpha=1.,verbose=verbose_iter)
    #tomo_sirtBB =  iradon(data)

    end = timer()
    times['solver']+=end-start
    #print("rank no", rank, "chunks",chunk_slices[rank]+loop_chunks[ii], "rnrm",rnrmp,"data shapes", data.shape)
    
    
    # gather from GPU to CPU
    start1 = timer()
    if GPU: tomo_sirtBB=xp.asnumpy(tomo_sirtBB)
    end1 = timer()
    times['g2c']+=end1-start1
    
#    print("getting the tomo")
 #   comm.Barrier()
    start = timer()
    if rank ==0:  #ptomo=tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]
#        slice_shape=tomo_sirtBB.shape[1:]
#        tshape=(np.append(chunk_slices[-1,-1]-chunk_slices[0,0],slice_shape))
#        ptomo = np.empty(tuple(tshape),dtype=tomo_sirtBB.dtype)
        ptomo = tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]
        
        #print("tomo_sirtBB shape",ptomo.shape)
    else:
        ptomo=None

   
    
    gatherv(tomo_sirtBB,chunk_slices,data=ptomo)
#    if rank ==0: tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]=ptomo
    comm.Barrier()
    end = timer()
#    print("got the tomo")
#    print("copied the tomo")
    #time_sirtBB=end-start
    times['gather']=end-start
    
    for ii in times:
        times_loop[ii]+=times[ii]
        #if rank==0: print(ii,times[ii],times_loop[ii])
        
    #if rank == 0: print("times",times,"\ntotal times",times_loop)

#print("finished loop")
 
##################################
#print("rank:",rank)

if rank>0:     quit()

print("reading tomo, shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)

true_obj = get_data(num_slices,num_rays,num_angles,obj_width,'tomo')

#print("finished mpi")
print("times last loop_chunk",times)
print("times full tomo", times_loop)
#print("mpi gather time=", time_mpi, "GPU->CPU time:", time_g2c)


scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
rescale = lambda x,y: scale(x,y)*x
ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
ssnr2    = lambda x,y: ssnr(x,y)**2


print("sirtBB  time=", times_loop['solver'], "snr=", ssnr(true_obj,tomo))

#print("psirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo1))

## tomo to cropped image
#t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
## vector to tomo
#v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))

#print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))
