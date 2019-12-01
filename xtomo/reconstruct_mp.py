import numpy as np
from timeit import default_timer as timer
import time


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

def recon_file(fname, tomo_out=None, dnames=None, algo = 'iradon' ,rot_center = None, 
               max_iter = None, tol=None, GPU = True, 
               shmem = False, max_chunk_slice=16,  
               reg = None, tau = None, verbose = verboseall, 
               ncore=None, chunks=None,mpring=True):
    #print("recon_file max_iter",max_iter)
    if verbose>0: 
        printv0('reconstructing',fname)
        try:
            printv0('saving to',tomo_out.filename)
        except:
            pass
    
#    if verbose>0: printv0('saving to',tomo_out.filename)
    csize = 0
    import h5py
    fid= h5py.File(fname, "r",rdcc_nbytes=csize)
    if type(dnames) == type(None):
        dnames=dnames_get()
    sino  = fid[dnames['sino']]
    theta = fid[dnames['theta']]
    if tol==None: tol=5e-3
    tomo, times_loop = recon(sino, theta, algo = algo , tomo_out=tomo_out,
                             rot_center = rot_center, max_iter = max_iter, tol=tol, GPU=GPU, shmem=shmem, max_chunk_slice=max_chunk_slice,  reg = reg, tau = tau, 
                             verbose = verbose,ncore=ncore, crop=chunks, mpring=mpring)
    return tomo, times_loop, sino.shape
    

#########################################

def recon(sino, theta, algo = 'iradon', tomo_out=None, rot_center = None, max_iter = None, tol=5e-3, GPU = True, shmem = False, max_chunk_slice=16,  reg = None, tau = None, verbose = verboseall,ncore=None, crop=None, mpring=False):

    def printv(*args,**kwargs): 
        if verbose>0:  printv0(*args,**kwargs)
    
    #timing
    def printvt(*args,**kwargs): 
        if verbose>1:  printv0(*args,**kwargs)
    
    num_slices = sino.shape[0]
    num_angles = sino.shape[1]
    num_rays   = sino.shape[2]
    obj_width  = .95

    if type(rot_center)==type(None):
        rot_center = num_rays//2

    times={'loop':0, 'setup':0, 'h5read':0, 'solver':0, 'c2g':0, 'g2c':0, 'barrier':0, 'gather':0 }
    times_loop=times.copy()
    
    #cropped output
    loop_offset=0
    if crop!=None:
        if len(crop)==1:
            loop_offset=num_slices//2
            num_slices=min([crop[0],num_slices])
            loop_offset-=num_slices//2
            
        else:
            loop_offset=crop[0]
            num_slices=crop[1]-crop[0]
    
    
    
    if algo=='tomopy-gridrec':
        GPU=False
        #algo='tomopy-gridrec'

    
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
                   ", chunk tomo memory:",max_chunk_slice*(num_rays**2)*4/(2**10)**2,'MB')
            #xp.cuda.profiler.initialize()
            xp.cuda.profiler.start()
        except:
            pass
    else:
        xp=np

    theta=xp.array(theta)
    
    # set up radon
    printv("setting up the solver. ", end = '')
    
    
    
    start=timer()
    from wrap_algorithms import wrap
    reconstruct=wrap(sino.shape,theta,rot_center,algo,xp=xp, obj_width=obj_width, max_iter=max_iter, tol=tol, reg=reg, tau=tau, ncore=ncore, verbose=verbose)   
       
    #print("[0] rank",rank, "used bytes", mempool.used_bytes())
    
    end = timer()
    time_radonsetup=(end - start)
    times_loop['setup']=time_radonsetup
    printv("time=", time_radonsetup,flush=True)
    

    
    #divide up loop chunks evenly across mpi ranks    
    loop_chunks=get_loop_chunk_slices(num_slices, mpi_size, max_chunk_slice )
    
    printv("nslices:",num_slices," mpi_size:", mpi_size," max_chunk:",max_chunk_slice)
    printv("rank",rank,"loop_chunks:", loop_chunks)
    
    
    start_loop_time =time.time()
    
    # if rank == 0: tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
    #if algo=='tomopy-gridrec':
    #    shmem=False
    
    
    ######################################### 
    # MP ring buffer setup
    #mpring=True
    
    if mpring>0:
        printv("multiprocessing ring buffer",mpring,flush=True)
        import multiprocessing as mp
        import ctypes
        from multiprocessing import sharedctypes
        
        def shared_array(shape=(1,), dtype=np.float32):  
            np_type_to_ctype = {np.float32: ctypes.c_float,
                                np.float64: ctypes.c_double,
                                np.bool: ctypes.c_bool,
                                np.uint8: ctypes.c_ubyte,
                                np.uint64: ctypes.c_ulonglong}

            numel = np.int(np.prod(shape))
            arr_ctypes = sharedctypes.RawArray(np_type_to_ctype[dtype], numel)
            np_arr = np.frombuffer(arr_ctypes, dtype=dtype, count=numel)
            np_arr.shape = shape

            return np_arr 
        
    
    if np.mod(mpring,2)==1: # reading ring buffer (1 or 3)
        pr=[0,0] #process even or odd
        data_ring  = shared_array(shape=(2,max_chunk_slice, num_angles, num_rays),dtype=np.float32)
        
#        def read_data(sino, loop_chunks, ii):
        def read_data(sino, chunks, ii):
            """no synchronization."""
            #info("start %s" % (i,))
            even = np.mod(ii+1,2)
            #nslices = loop_chunks[ii+1]-loop_chunks[ii]
            #chunk_slices = get_chunk_slices(nslices)
            #chunks=chunk_slices[rank,:]+loop_chunks[ii]
            data_ring[even,0:chunks[1]-chunks[0],...]= sino[chunks[0]:chunks[1],...]
    #print('here')
    if mpring>1: # writing ring buffer (2 or 3)
        printv('ring buffer writing to file')
        pw=[0,0] #process even or odd        
        tomo_ring  = shared_array(shape=(2,max_chunk_slice, num_rays, num_rays),dtype=np.float32)
        #global tomo_out 
        #def write_tomo(tomo_out,chunks,ii):
        #import h5py
        def write_tomo(tomo_out,chunks,ii):
            #fid = h5py.File(tomo_out.file_name, 'a')
            #t_out=fid['/exchange/tomo']
            #t_out[chunks[0]:chunks[1],...]=tomo_ring[even,0:chunks[1]-chunks[0],...]
            #fid.close()
            #global tomo_out 
            #even = np.mod(ii+1,2)
            #if rank==0: print('\n writing out', tomo_ring[even,(chunks[1]-chunks[0])//2,num_rays//2,num_rays//2])
            #####t
            tomo_out[chunks[0]:chunks[1],...]=tomo_ring[even,0:chunks[1]-chunks[0],...]
            #print('\n written h5', tomo_out[chunks[0]+(chunks[1]-chunks[0])//2,num_rays//2,num_rays//2],'\n')
            #print('\n mp recon norm',np.linalg.norm(tomo_out))
            #if rank==0: tomo_out.flush()
   
    # MP ring buffer setup
    ######################################### 
    
    ##########################################
    # gather the results: write to memmap, shared memory, gatherv

    if algo[0:min(len(algo),6)]!='tomopy': # not for tomopy-...
        #print('there1')
#        if tomo_out!=-1:
#            print('there2')
#            pass
#            #tomo=tomo_out
#            #shmem=1 # trick it to avoid mpi gather
        if shmem and (mpring<2):
            #print('there shared')
#        if shmem:
            #from communicator import allocate_shared_tomo
            from communicator import allocate_shared
            try:
                #print("using shared memory to communicate")
                tomo = allocate_shared((num_slices,num_rays,num_rays),rank,mpi_size)
            except:
                bold='\033[1m'
                endb= '\033[0m'
                printv(bold+"shared memory is not working, set the flag -S to 0")
                printv("reverting to mpi gatherv that but it may not work\n",'='*60,endb)
                shmem=False
        #print('hi there shmem,type(tomo_out)',type(tomo_out),flush=True)
                
        if (not shmem) and (mpring<2):
            printv('using gatherv distributed mpi', flush=True)
            
            from communicator import gatherv
            # gatherv - allocate 
            tomo=None
            tomo_local=None
            if rank == 0: tomo = np.empty((num_slices,num_rays,num_rays),dtype = 'float32')
        #print('neither?')
    else: # tomopy with no mpi no ring buffer
        print('tomopy')
        tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')
    #print('finally')
    halo=0
    if algo == 'tv': halo = 2 #not used at the moment
    halo+=0

    ######### loop through chunks
    printv('starting loop')
    for ii in range(loop_chunks.size-1):

        nslices = loop_chunks[ii+1]-loop_chunks[ii]
        chunk_slices = get_chunk_slices(nslices) 
        
        #printv("rank",rank,"size",mpi_size,"loop_chunks:", loop_chunks)
        
        printv( 'loop_chunk {}/{}:{}, mpi chunks {}'.format(ii+1,loop_chunks.size-1, loop_chunks[ii:ii+2],loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel()))
        
    
        start_read = time.time()
        
        printv("reading slices:", end = '')    
        chunks=chunk_slices[rank,:]+loop_chunks[ii]
        
        # halo for TV - to do ...
        #bchunk=np.clip(chunks[0]-halo,0,num_slices)
        #echunk=np.clip(chunks[1]+halo,0,num_slices)

        #bhalo=chunks[0]-np.clip(chunks[0]-halo,0,num_slices-1)
        #ehalo=np.clip(chunks[1]+halo,0,num_slices-1)-chunks[1]
        #data1 = sino[chunks[0]-bhalo:chunks[1]+bhalo,...]
        #chunksi=np.clip(np.arange(chunks[0]-halo,chunks[1]+halo),0,num_slices-1)
        #data xrxrm= sino[chunksi,...]
        
        #data = sino[chunks[0]:chunks[1],...]

        # ring buffer read
        even = np.mod(ii+1,2)
        # read directly the first round, or if mpring is odd (1 or 3) 
        if (not np.mod(mpring,2)) or ii==0:
            data = sino[chunks[0]+loop_offset:chunks[1]+loop_offset,...]
        # read ring buffer
        else: 
            pr[even].join()
            data=data_ring[even,0:chunks[1]-chunks[0],...]
            pr[even].terminate()
        # launch next read 
        if np.mod(mpring,2) and ii<loop_chunks.size-2:
            #pr[1-even] = mp.Process(target=read_data, args=(sino, loop_chunks, ii+1))
            pr[1-even] = mp.Process(target=read_data, args=(sino, chunks, ii+1))
            pr[1-even].start()
        
        end_read=time.time()
        if rank ==0: times['h5read']=(end_read - start_read)

        printv("\rread time ={:3g}".format(times['h5read']),end=' ')

        # copy to gpu or cpu
        start = timer()
        data=xp.array(data)
        end = timer()
        times['c2g']=end - start

        
        printv("reconstructing slices, ", end = '') 
        start_solver = timer()
 
        mpi_time=0.
        if algo == 'tomopy-gridrec':
            #tomo, rnrm =  reconstruct(data,verbose_iter)
            tomo[chunks[0]:chunks[1],...], rnrm, g2ctime =  reconstruct(data,verbose_iter,ncore)
            #tomo[chunks[0]:chunks[1],...]=tomopy.recon(data, theta, center=None, sinogram_order=True, algorithm="gridrec")
        else:
            #tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter,)
            tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter)
            start_gather = timer()

            if mpring>1: # 2 or 3                

                if ii>1: 
                    #printv('\n iter',ii,'joining pw[',even,']',pw[even])
                    pw[even].join() #make sure this is done
                tomo_ring[even,0:chunks[1]-chunks[0],...]=tomo_chunk
                # kill the previous job
                if ii>1: pw[even].terminate()
                # start writing
                #printv('\n writin iter',ii,'starting pw[]',even)
                pw[even] = mp.Process(target=write_tomo, args=(tomo_out,chunks,ii))
                pw[even].start()
                
                #print('\n recon norm loop',np.linalg.norm(tomo_out))
                
            elif shmem:
#                tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter)
                ##tomo_chunk = tomo_chunk[bhalo:-ehalo,...]
                #tomo[chunks[0]:chunks[1],...] = tomo_chunk[bhalo:-ehalo,...]
                #tomo_chunk
                start_gather = timer() # this is in shared memory
                tomo[chunks[0]:chunks[1],...] = tomo_chunk
                #tomo[chunks[0]:chunks[1],...], rnrm, g2ctime =  reconstruct(data,verbose_iter)
            else:
#                tomo_chunk, rnrm, g2ctime =  reconstruct(data,verbose_iter)
#                start_gather = timer()                
                if rank ==0:
                    tomo_local = tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]
#                else:
#                    tomo_local=None
                    
                gatherv(tomo_chunk[0:chunks[1]-chunks[0]],chunk_slices,data=tomo_local)
                #gatherv(tomo_chunk[bhalo:-ehalo,...],chunk_slices,data=tomo_local)
                
        mpi_time= timer()-start_gather
        times['gather'] = mpi_time

        #printv("mpi time ={:3g}".format(times['gather']),flush=True)

    
        times['solver'] = timer()-start_solver-mpi_time
        times['g2c']    = g2ctime

        printv("\r read time={:3g}, solver ={:3g}, gather={:3g}, g2c ={:3g}".format(times['h5read'],times['solver'], times['gather'], times['g2c']),flush=True)
    
        for jj in times: times_loop[jj]+=times[jj]

    #if rank==0:
    #    print('\n recon norm',np.linalg.norm(tomo_out))

    start = timer()
    mpi_barrier()
    times['barrier']+=timer()-start
    
    if GPU:
        #printv("stopping profiler")
        try:
            xp.cuda.profiler.stop()
        except:
            None
    
    
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
    # make sure all writing is done
    if mpring>1:
        pw[1-even].join()
        pw[1-even].terminate()
        pw[even].join()
        pw[even].terminate()
        tomo=tomo_out

        
    if rank>0: quit()
    print('done')    

    return tomo, times_loop

