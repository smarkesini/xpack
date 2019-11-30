import numpy as np
from timeit import default_timer as timer
import time
import multiprocessing

from communicator import rank, mpi_size, get_loop_chunk_slices, get_chunk_slices, mpi_barrier

verboseall = True and (rank == 0)
verbose_iter= (1/20) * int(verboseall) # print every 20 iterations
#verbose_iter= int(verboseall) # print every 20 iterations
#verbose_iter= int(False) # print every 20 iterations
 
def printv0(*args,**kwargs):
    if not verboseall: return
    #if 'verbose' in kwargs:
    #    if kwargs['verbose']==0: return
        
    if len(kwargs)==0:
        print(' '.join(map(str,args)))
        return
    elif 'flush' in kwargs:
        if kwargs['flush']:
            #print(args)
            print(' '.join(map(str,args)),flush=True)
        else:
            print(' '.join(map(str,args)))
        return
    elif 'end' in kwargs:
        print(' '.join(map(str,args)), end = '')
# ================== reconstruct ============== #

def dnames_get():
    dnames={'sino':"exchange/data", 'theta':"exchange/theta", 'tomo':"exchange/tomo", 'rot_center':"exchange/rot_center"}
    return dnames #dname_sino,dname_theta,dname_tomo

def recon_file(fname,dnames=None, algo = 'iradon' ,rot_center = None, max_iter = None, tol=None, GPU = True, shmem = False, max_chunk_slice=16,  reg = None, tau = None, verbose = verboseall,ncore=None, chunks=None):
    #print("recon_file max_iter",max_iter)
    if verbose>0: printv0('recon file',fname)
    csize = 0
    import h5py
    fid= h5py.File(fname, "r",rdcc_nbytes=csize)
    if type(dnames) == type(None):
        dnames=dnames_get()
    sino  = fid[dnames['sino']]
    theta = fid[dnames['theta']]
    if tol==None: tol=5e-3
    tomo, times_loop = recon(sino, theta, algo = algo ,rot_center = rot_center, max_iter = max_iter, tol=tol, GPU=GPU, shmem=shmem, max_chunk_slice=max_chunk_slice,  reg = reg, tau = tau, verbose = verbose,ncore=ncore, crop=chunks)
    return tomo, times_loop, sino.shape
    
######################################### 
# MP ring buffer setup
mpring=True


#########################################

def recon(sino, theta, algo = 'iradon' ,rot_center = None, max_iter = None, tol=5e-3, GPU = True, shmem = False, max_chunk_slice=16,  reg = None, tau = None, verbose = verboseall,ncore=None, crop=None):

    def printv(*args,**kwargs): 
        if verbose>0:  printv0(*args,**kwargs)
        
    #printv("tolerance:",tol)
    #shmem = True
    #shmem = False
    
    #GPU   = True
    
    num_slices = sino.shape[0]
    num_angles = sino.shape[1]
    num_rays   = sino.shape[2]
    obj_width  = .95
    
    loop_offset=0
    if crop!=None:
        if len(crop)==1:
            loop_offset=num_slices//2
            num_slices=min([crop[0],num_slices])
            loop_offset-=num_slices//2
            
        else:
            loop_offset=crop[0]
            num_slices=crop[1]-crop[0]
    
    if type(rot_center)==type(None):
        rot_center = num_rays//2
    
    
    if algo=='tomopy-gridrec':
        GPU=False
        algo='tomopy-gridrec'
        #import psutil
        #nproc=psutil.cpu_count()
        #max_chunk_slice=64
        #max_chunk_slice=nproc*4
    
    
    
    if max_iter == None: max_iter = 10
    #print("recon max_iter",max_iter)

    #float_size=32/8; alg_tsize=4; alg_ssize=3
    #slice_gbsize=num_rays*(num_rays*alg_tsize+num_angles*alg_ssize)*(float_size)/((2**10)**3)
    #
    ## leave .5 gb and another 9*3 (kernel*3, 3 is data(complex+col+row) for sparse
    #max_chunk_slice=np.int(np.floor((device_gbsize-.5)/slice_gbsize)-9*3*num_angles/num_rays)
    #print("max chunk size", max_chunk_slice, 'device_gbsize',device_gbsize,"slice size",slice_gbsize,
    #      "(device-2)/slice size",(device_gbsize-1)/slice_gbsize)
    
    
    
    printv("GPU:{} , algorithm:{}".format(GPU,algo))
    #printv("tomo shape (",num_slices,num_rays,num_rays, ") n_angles",num_angles, "max_iter",max_iter)
    printv("tomo shape ({},{},{}) n_angles {} max_iter {}".format(num_slices,num_rays,num_rays ,num_angles, max_iter))
    
    printv("max chunk size ", max_chunk_slice,flush=True)
    
    
    
    if GPU:
        from devmanager import set_visible_device
    
        import cupy as xp
        do,vd,nd=set_visible_device(rank)
        #device_gbsize=xp.cuda.Device(vd).mem_info[1]/((2**10)**3)
        try:
            device_gbsize=xp.cuda.Device().mem_info[1]/((2**10)**3)
            printv("gpu memory:",device_gbsize, 
                   "GB, chunk sino memory:",max_chunk_slice*num_rays*num_angles*4/(2**10)**2,'MB',
                   "GB, chunk tomo memory:",max_chunk_slice*(num_rays**2)*4/(2**10)**2,'MB')
            #xp.cuda.profiler.initialize()
            xp.cuda.profiler.start()
        except:
            pass
        #print("rank:",rank,"device:",vd, "gb memory:", device_gbsize)
        #xp.cuda.profile()
        
#        #mode = 'cuda'
#        if GPU:
#            cupy = xp
#            mempool = cupy.get_default_memory_pool()
#            pinned_mempool = cupy.get_default_pinned_memory_pool()
            
    else:
        xp=np
        #mode= 'cpu'
        #device_gbsize=(128-32)/mpi_size # GBs used per rank, leave 32 GB for the rest
    
    
    
        
    
    theta=xp.array(theta)
    
    
    # set up radon
    printv("setting up the solver. ", end = '')
    
    times={'loop':0, 'setup':0, 'h5read':0, 'solver':0, 'c2g':0, 'g2c':0, 'barrier':0, 'gather':0 }
    times_loop=times.copy()
    
    
    start=timer()
    from wrap_algorithms import wrap
    reconstruct=wrap(sino.shape,theta,rot_center,algo,xp=xp, obj_width=obj_width, max_iter=max_iter, tol=tol, reg=reg, tau=tau, ncore=ncore, verbose=verbose)   
       
    #print("[0] rank",rank, "used bytes", mempool.used_bytes())
    
    end = timer()
    time_radonsetup=(end - start)
    #if rank==0: print("time=", time_radonsetup,flush=True)
    printv("time=", time_radonsetup,flush=True)
    
    # solver
    #from solve_sirt import sirtBB
    
    
    loop_chunks=get_loop_chunk_slices(num_slices, mpi_size, max_chunk_slice )
    
    printv("nslices:",num_slices," mpi_size:", mpi_size," max_chunk:",max_chunk_slice)
    printv("rank",rank,"loop_chunks:", loop_chunks)
    
    times_loop['setup']=time_radonsetup
    
    start_loop_time =time.time()
    
    # if rank == 0: tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
    #if algo=='tomopy-gridrec':
    #    shmem=False
    
    if algo!='tomopy-gridrec':
        if shmem:
            #from communicator import allocate_shared_tomo
            from communicator import allocate_shared
            #print("using shared memory to communicate")
            tomo = allocate_shared((num_slices,num_rays,num_rays),rank,mpi_size)
        else:
            from communicator import gatherv
            # gatherv - allocate 
            tomo=None
            tomo_local=None
            if rank == 0: tomo = np.empty((num_slices,num_rays,num_rays),dtype = 'float32')
    
    else:
        tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
        
    halo=0
    if algo == 'tv': halo = 2
    halo+=0
    
    ######### loop through chunks
    for ii in range(loop_chunks.size-1):
    #for ii in [loop_chunks.size//2-1]: #range(loop_chunks.size-1):
        nslices = loop_chunks[ii+1]-loop_chunks[ii]
        chunk_slices = get_chunk_slices(nslices) 
        
        #printv("rank",rank,"size",mpi_size,"loop_chunks:", loop_chunks)
        
        printv( 'loop_chunk {}/{}:{}, mpi chunks {}'.format(ii+1,loop_chunks.size-1, loop_chunks[ii:ii+2],loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel()))
        
    
        start_read = time.time()
        
        printv("reading slices:", end = '')    
        chunks=chunk_slices[rank,:]+loop_chunks[ii]
        
        #bchunk=np.clip(chunks[0]-halo,0,num_slices)
        #echunk=np.clip(chunks[1]+halo,0,num_slices)

        #bhalo=chunks[0]-np.clip(chunks[0]-halo,0,num_slices-1)
        #ehalo=np.clip(chunks[1]+halo,0,num_slices-1)-chunks[1]
        #data1 = sino[chunks[0]-bhalo:chunks[1]+bhalo,...]
        #chunksi=np.clip(np.arange(chunks[0]-halo,chunks[1]+halo),0,num_slices-1)
        #data = sino[chunksi,...]
        
        #data = sino[chunks[0]:chunks[1],...]
        data = sino[chunks[0]+loop_offset:chunks[1]+loop_offset,...]
        
        #data = sino[bchunk:echunk,...]
        
        end_read=time.time()
        if rank ==0: times['h5read']=(end_read - start_read)

        printv("read time ={:3g}".format(times['h5read']),flush=True)

        # copy to gpu or cpu
        start = timer()
        data=xp.array(data)
        end = timer()
        times['c2g']=end - start

        
        printv("reconstructing slices, ", end = '') 
        start = timer()
 
        mpi_time=0.
        if algo == 'tomopy-gridrec':
            #tomo, rnrm =  reconstruct(data,verbose_iter)
            tomo[chunks[0]:chunks[1],...], rnrm, g2ctime =  reconstruct(data,verbose_iter,ncore)
            #tomo[chunks[0]:chunks[1],...]=tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec")
        else:
            #tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter)

            if shmem:
                #tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter)
                ##tomo_chunk = tomo_chunk[bhalo:-ehalo,...]
                #tomo[chunks[0]:chunks[1],...] = tomo_chunk[bhalo:-ehalo,...]
                
                tomo[chunks[0]:chunks[1],...], rnrm, g2ctime =  reconstruct(data,verbose_iter)
            else:
                tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter)
                start_gather = timer()
                if rank ==0:
                    tomo_local = tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]
#                else:
#                    tomo_local=None
                gatherv(tomo_chunk,chunk_slices,data=tomo_local)
                #gatherv(tomo_chunk[bhalo:-ehalo,...],chunk_slices,data=tomo_local)
                
                mpi_time= timer()-start_gather
                
    
        times['solver'] = timer()-start-g2ctime
        times['g2c']    = g2ctime
        times['gather'] = mpi_time

        printv("solver time ={:3g}, g2c time={:3g}".format(times['solver'],times['g2c']),flush=True)
    
        for jj in times: times_loop[jj]+=times[jj]

            
    start = timer()
    mpi_barrier()
    times['barrier']+=timer()-start
    
    if GPU:
        #printv("stopping profiler")
        try:
            xp.cuda.profiler.stop()
        except:
            None
    
    if rank>0:     
        quit()
    
    #print("times last loop_chunk",times)
    end_loop=time.time()
    times_loop['loop']=end_loop-start_loop_time 
    
    """
    bold='\033[1m'
    endb= '\033[0m'
    print(bold+"tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    print("times full tomo", times_loop)
    print("loop+setup time=", times_loop['loop']+times_loop['setup'],endb)
    #print("times full tomo", times_loop,flush=True)
    """
    return tomo, times_loop

