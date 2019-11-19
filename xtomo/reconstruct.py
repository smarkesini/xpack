import numpy as np
from timeit import default_timer as timer
import time

from communicator import rank, mpi_size, get_loop_chunk_slices, get_chunk_slices, mpi_barrier

verboseall = True and (rank == 0)
verbose_iter= (1/20) * verboseall # print every 20 iterations

def printv(*args,**kwargs):
    if verboseall:
        if len(kwargs)==0:
            print(''.join(map(str,args)))
        elif 'flush' in kwargs:
            if kwargs['flush']:
                #print(args)
                print(''.join(map(str,args)),flush=True)
            else:
                print(''.join(map(str,args)))
        elif 'end' in kwargs:
            print(''.join(map(str,args)), end = '')
# ================== reconstruct ============== #

def recon_file(fname,dnames, algo = 'iradon' ,rot_center = None, max_iter = None):
    csize = 0
    import h5py
    fid= h5py.File(fname, "r",rdcc_nbytes=csize)
    sino  = fid[dnames['sino']]
    theta = fid[dnames['theta']]
    reconstruct(sino, theta, algo = algo ,rot_center = rot_center, max_iter = max_iter)
    
    

def reconstruct(sino, theta, algo = 'iradon' ,rot_center = None, max_iter = None):

    
    shmem = True
    GPU   = False

    num_slices = sino.shape[0]
    num_angles = sino.shape[1]
    num_rays   = sino.shape[2]
    obj_width  = .95
    
    if type(rot_center)==type(None):
        rot_center = num_rays//2
    
    max_chunk_slice=16*3
    
    if algo=='tomopy-gridrec':
        GPU=False
        algo='tomopy-gridrec'
        import psutil
        nproc=psutil.cpu_count()
        #max_chunk_slice=64
        max_chunk_slice=nproc*4
    
    
    
    if max_iter == None: max_iter = 10
    
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
    
    printv("max chunk size", max_chunk_slice,flush=True)
    
    
    
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
    
    #print("[0] rank",rank, "used bytes", mempool.used_bytes())
    
    
    if algo=='tomopy-gridrec':
        import tomopy
        rnrm=None
        def reconstruct(data,verbose):
            tomo_t = tomopy.recon(data, theta, center=rot_center, sinogram_order=True, algorithm="gridrec")
            return tomo_t, rnrm, 0.
    else:
        from fubini import radon_setup as radon_setup
        if algo=='iradon':
            
            iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width,iradon_only=True)
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
            #t=0.
            def reconstruct(data,verbose):
                tomo_t,rnrm=sirtBB(radon, iradon, data, xp, max_iter=max_iter, alpha=1.,verbose=verbose)
                if GPU:
                    start1 = timer()
                    tomo= xp.asnumpy(tomo_t)
                    t=timer()-start1
                    return tomo,rnrm,t
                else: return tomo_t,rnrm, 0.
        elif algo == 'tv':
            
            radon,iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
            tau=0.05
            print("τ=",tau)
            r = .8   
            #from solvers import Grad
            from solvers import solveTV
            #def init(data):
            #    tomo0=iradon(data)
            #t = 0.
            def reconstruct(data,verbose):
                tomo_t,rnrm = solveTV(radon, iradon, data, r, tau,  tol=1e-2, maxiter=10, verbose=verbose)
                if GPU:
                    start1 = timer()
                    tomo= xp.asnumpy(tomo_t)
                    t=timer()-start1
                    return tomo,rnrm,t
                else: return tomo_t,rnrm,0.
    
    
    end = timer()
    time_radonsetup=(end - start)
    #if rank==0: print("time=", time_radonsetup,flush=True)
    printv("time=", time_radonsetup,flush=True)
    
    # solver
    #from solve_sirt import sirtBB
    
    
    loop_chunks=get_loop_chunk_slices(num_slices, mpi_size, max_chunk_slice )
    
    printv("nslices",num_slices,"mpi_size", mpi_size,"max_chunk",max_chunk_slice)
    printv("loop_chunks", loop_chunks)
    
    times_loop['setup']=time_radonsetup
    
    start_loop_time =time.time()
    
    # if rank == 0: tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
    
    
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
            if rank == 0: tomo = np.empty((num_slices,num_rays,num_rays),dtype = 'float32')
    
    else:
        tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
        
    
    ######### loop through chunks
    for ii in range(loop_chunks.size-1):
    #for ii in [loop_chunks.size//2-1]: #range(loop_chunks.size-1):
        nslices = loop_chunks[ii+1]-loop_chunks[ii]
        chunk_slices = get_chunk_slices(nslices) 
    
        printv( 'loop_chunk {}/{}:{}, mpi chunks {}'.format(ii+1,loop_chunks.size-1, loop_chunks[ii:ii+2],loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel()))
        
    
        start_read = time.time()
        
        printv("reading slices:", end = '')    
        chunks=chunk_slices[rank,:]+loop_chunks[ii]
        data = sino[chunks[0]:chunks[1],...]

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
            tomo[chunks[0]:chunks[1],...], rnrm, g2ctime =  reconstruct(data,verbose_iter)
            #tomo[chunks[0]:chunks[1],...]=tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec")
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
                
    
        times['solver'] = timer()-start-g2ctime
        times['g2c']    = g2ctime
        times['gather'] = mpi_time

        printv("solver time ={:3g}, g2c time={:3g}".format(times['solver'],times['g2c']),flush=True)
    
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
    
    return tomo, times_loop