import numpy as np
import matplotlib.pyplot as plt
#import h5py

#print("hello",flush=True)
#import tomopy
#import imageio
#import os
GPU=True
GPU=False

shmem=True
#shmem = False

algo='iradon'
#algo='sirt'
#algo = 'tv'
#algo='tomopy-gridrec'




obj_size = 1024//4
num_slices = 16*32# size//2
#num_angles =    obj_size//2
num_angles =  1501
#num_angles =    11

obj_width=0.95

num_rays   = obj_size
rot_center = None


simulate=True
if simulate:
    from simulate_data import get_data as gdata   
    def get_data(x,chunks=None):    
        return gdata(num_slices,num_rays,num_angles,obj_width,x,chunks=chunks) 
else:
    from read_tomobank import get_data
    
    #rot_center = 1403
    #rot_center = 1024
    #from read_sigray import get_data
    #rot_center=512
    # (1792, 1501, 2048)
#    data_shape = get_data('dims')
#    #print("data shape",data_shape)
#    num_slices = data_shape[0]
#    num_angles = data_shape[1]
#    num_rays   = data_shape[2]
    

theta = get_data('theta')
sino = get_data('sino')


max_iter = 10

# ================== reconstruct ============== #

num_slices = sino.shape[0]
num_angles = sino.shape[1]
num_rays   = sino.shape[2]

if type(rot_center)==type(None):
    rot_center = num_rays//2

max_chunk_slice=16*3

if algo=='tomopy-gridrec':
    GPU=False
    algo='tomopy-gridrec'
    #import psutil
    #psutil.cpu_count()
    max_chunk_slice=64





#float_size=32/8; alg_tsize=4; alg_ssize=3
#slice_gbsize=num_rays*(num_rays*alg_tsize+num_angles*alg_ssize)*(float_size)/((2**10)**3)
#
## leave .5 gb and another 9*3 (kernel*3, 3 is data(complex+col+row) for sparse
#max_chunk_slice=np.int(np.floor((device_gbsize-.5)/slice_gbsize)-9*3*num_angles/num_rays)
#print("max chunk size", max_chunk_slice, 'device_gbsize',device_gbsize,"slice size",slice_gbsize,
#      "(device-2)/slice size",(device_gbsize-1)/slice_gbsize)



from timeit import default_timer as timer
import time

from communicator import rank, mpi_size, get_loop_chunk_slices, get_chunk_slices, mpi_barrier

if rank==0: print("GPU: ", GPU,", algorithm",algo,flush=True)
if rank==0: print("tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)
if rank==0: print("max chunk size", max_chunk_slice)


if GPU:
    from devmanager import set_visible_device

    import cupy as xp
    do,vd,nd=set_visible_device(rank)
    #device_gbsize=xp.cuda.Device(vd).mem_info[1]/((2**10)**3)
    device_gbsize=xp.cuda.Device(0).mem_info[1]/((2**10)**3)
    #xp.cuda.profiler.initialize()
    xp.cuda.profiler.start()
    #print("rank:",rank,"device:",vd, "gb memory:", device_gbsize)
    #xp.cuda.profile()
    
    mode = 'cuda'
    if GPU:
        cupy = xp
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        
else:
    xp=np
    mode= 'cpu'
    #device_gbsize=(128-32)/mpi_size # GBs used per rank, leave 32 GB for the rest



    

theta=xp.array(theta)


# set up radon
if rank==0:  print("setting up the solver. ", end = '')

#times={'scatt':0, 'c2g':0, 'radon':0 ,'solver':0, 'g2c':0, 'gather':0 }
times={'loop':0, 'setup':0, 'h5read':0, 'solver':0, 'c2g':0, 'g2c':0, 'barrier':0, 'gather':0 }
times_loop=times.copy()


start=timer()

#print("[0] rank",rank, "used bytes", mempool.used_bytes())


if algo=='tomopy-gridrec':
    import tomopy
    reconstruct  = lambda data,verbose: (tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec"),None)
    num_slices 
else:
    from fubini import radon_setup as radon_setup
    if algo=='iradon':
        
        iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='None', kernel_type = 'gaussian', k_r =1, width=obj_width,iradon_only=True)
        rnrm=0
        def reconstruct(data,verbose):
            tomo_t=iradon(data)
            if GPU:
                start1 = timer()
                tomo= xp.asnumpy(tomo_t)
                t=timer()-start1
                return tomo,rnrm,t
            else: return tomo_t,None,0.
            
            
    elif algo == 'sirt':

        radon,iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)

        import solve_sirt 
        solve_sirt.init(xp)
        sirtBB=solve_sirt.sirtBB
        t=0.
        def reconstruct(data,verbose):
            tomo_t,rnrm=sirtBB(radon, iradon, data, xp, max_iter=max_iter, alpha=1.,verbose=verbose)
            if GPU:
                start1 = timer()
                tomo= xp.asnumpy(tomo_t)
                t=timer()-start1
                return tomo,rnrm,t
            else: return tomo_t,rnrm,t
    elif algo == 'tv':
        
        radon,iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
        tau=0.05
        print("τ=",tau)
        r = .8   
        #from solvers import Grad
        from solvers import solveTV
        #def init(data):
        #    tomo0=iradon(data)
        t = 0.
        def reconstruct(data,verbose):
            tomo_t,rnrm = solveTV(radon, iradon, data, r, tau,  tol=1e-2, maxiter=10, verbose=verbose)
            if GPU:
                start1 = timer()
                tomo= xp.asnumpy(tomo_t)
                t=timer()-start1
                return tomo,rnrm,t
            else: return tomo_t,None,0.


end = timer()
time_radonsetup=(end - start)
if rank==0: print("time=", time_radonsetup,flush=True)

# solver
#from solve_sirt import sirtBB


loop_chunks=get_loop_chunk_slices(num_slices, mpi_size, max_chunk_slice )
if rank==0: print("nslices",num_slices,"mpi_size", mpi_size,"max_chunk",max_chunk_slice)
if rank==0:print("loop_chunks", loop_chunks)

times_loop['setup']=time_radonsetup

start_loop_time =time.time()

# if rank == 0: tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')


if algo!='tomopy-gridrec':
    if shmem:
        from communicator import allocate_shared_tomo
        #print("using shared memory to communicate")
        tomo = allocate_shared_tomo(num_slices,num_rays,rank,mpi_size)
    else:
        from communicator import gatherv
        # gatherv - allocate 
        tomo=None
        if rank == 0: tomo = np.empty((num_slices,num_rays,num_rays),dtype = 'float32')

else:
    tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
    
#    if rank == 0: tomo = np.empty((nslices,num_rays,num_rays),dtype = 'float32')

verboseall = True
verbose_iter= (1/20) * (rank == 0) # print every 20 iterations
#verbose_iter= (1) * (rank == 0) # print every iterations

#print("verbose_iter",verbose_iter)
verbose= (rank ==0) and verboseall
#print("dir",dir())

######### 
for ii in range(loop_chunks.size-1):
#for ii in [loop_chunks.size//2-1]: #range(loop_chunks.size-1):
    nslices = loop_chunks[ii+1]-loop_chunks[ii]
    chunk_slices = get_chunk_slices(nslices) 

    if verbose: print( 'loop_chunk {}/{}'.format(ii+1,loop_chunks.size-1),':', loop_chunks[ii:ii+2], "mpi chunks",loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel(),)
    # if rank==0: print("chunk slices",chunk_slices)
    

    start_read = time.time()
    if verbose: print("reading slices:", end = '')  

    chunks=chunk_slices[rank,:]+loop_chunks[ii]

    #data = get_data('sino',chunks=chunks)
    data = sino[chunks[0]:chunks[1],...]
    #data = get_data('sino')
    end_read=time.time()
    if rank ==0: times['h5read']=(end_read - start_read)
    if verbose: print("read time ={:3g}".format(times['h5read']),flush=True)

    start = timer()
    #data=xp.array(data[chunks[0]:chunks[1],...])
    data=xp.array(data)
    #if GPU: data=xp.array(data[chunks[0]:chunks[1],...])
    #if GPU: data=xp.array(data)
    end = timer()
    times['c2g']=end - start
    if verbose: print("cpu2gpu time ={:3g}".format(times['c2g']),flush=True)
    #print("[2]",ii ,"rank",rank, "used bytes", mempool.used_bytes())

   
    if verbose: print("reconstructing slices:", end = '') 
    start = timer()
#
    #tomo_chunk, rnrm =  reconstruct(data,verbose_iter)
    mpi_time=0.
    if algo == 'tomopy-gridrec':
        #tomo, rnrm =  reconstruct(data,verbose_iter)
        tomo[chunks[0]:chunks[1],...]=tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec")
    else:
        if shmem:
            tomo[chunks[0]:chunks[1],...], rnrm, g2ctime =  reconstruct(data,verbose_iter)
        else:
            tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter)
            start_gather = timer()
            if rank ==0:
                ptomo = tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]
            else:
                ptomo=None
            gatherv(tomo_chunk,chunk_slices,data=ptomo)
            mpi_time= timer()-start_gather
            

    #tomo_chunk = tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec")
    times['solver'] = timer()-start-g2ctime
    times['g2c']    = g2ctime
    times['gather'] = mpi_time
    if verbose: print("solver time ={:3g}, g2c time={:3g}".format(times['solver'],times['g2c']),flush=True)

    #start = timer()
    # gather from GPU to CPU
    '''
    start1 = timer()
    if GPU: tomo_chunk=xp.asnumpy(tomo_chunk)
    end1 = timer()
    times['g2c']=end1-start1
    #end = timer()
    # times['gather']=end-start
    '''
    for ii in times: times_loop[ii]+=times[ii]
        
start = timer()
mpi_barrier()
times['barrier']+=timer()-start

if GPU:
    print("stopping profiler")
    xp.cuda.profiler.stop()

if rank>0:     
    quit()

#print("times last loop_chunk",times)
end_loop=time.time()
times_loop['loop']=end_loop-start_loop_time 


print("times full tomo", times_loop,flush=True)




import fubini
msk_tomo,msk_sino=fubini.masktomo(num_rays, np, width=.95)
msk_tomo=msk_tomo[0,...]

t2i = lambda x: x[num_slices//2,:,:].real
tomo0c=t2i(tomo)*msk_tomo
plt.imshow(np.abs(tomo0c))
plt.show()

#
#plt.imshow((data[0]))
#img = None
#for f in range(num_slices):
#    im=tomo[f,:,:]*msk_tomo
#    if img is None:
#        img = plt.imshow(im)
#    else:
#        img.set_data(im)
#    plt.pause(.01)
#    plt.draw()
#

try:
    true_obj = get_data('tomo')[...]
    print("comparing with truth, summary coming...\n\n")
except:
    true_obj = None
    #quit()

if type(true_obj) == type(None): 
    print("no tomogram to compare")
    #quit()

else:
    
    print("phantom shape",true_obj.shape, "n_angles",num_angles, 'algorithm:', algo,"GPU:",GPU,"max_iter:",max_iter)
    print("reading tomo, shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)
    
    
    scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
    rescale = lambda x,y: scale(x,y)*x
    ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
    ssnr2    = lambda x,y: ssnr(x,y)**2
    
    
    print("times full tomo", times_loop)
    print("solver time=", times_loop['solver'], "snr=", ssnr(true_obj,tomo))
    
    #tomo0=tomo_chunk
    
    
    #print("psirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo1))
    
    ## tomo to cropped image
    #t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
    ## vector to tomo
    #v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))
    
    #print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))
    
    #if GPU:
    #    cupy = xp
    #    mempool = cupy.get_default_memory_pool()
    #    pinned_mempool = cupy.get_default_pinned_memory_pool()
    #    print(mempool.used_bytes())
    #    del data,iradon,theta
    #    try: del radon
    #    except: None
    #        
    #    mempool.free_all_blocks()
    #    pinned_mempool.free_all_blocks()
    #    print(mempool.used_bytes())
    #
