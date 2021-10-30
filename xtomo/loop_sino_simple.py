import numpy as np
from timeit import default_timer as timer
import time
#import warnings

from .communicator import rank, mpi_size, get_loop_chunk_slices, get_chunk_slices, mpi_barrier
bold='\033[1m'
endb= '\033[0m'
verboseall = True and (rank == 0)
verbose_iter= (1/20) * int(verboseall) # print every 20 iterations
 
def printv0(*args,**kwargs):
    if not verboseall: return
    #if 'verbose' in kwargs:
    #    if kwargs['verbose']==0: return
    #print('rank',rank, end=' ')    
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


def chunktomo(num_slices, chunks):
    if type(chunks)!=type(None):
        chunks = np.array(chunks)
        if chunks.size==1: 
            chunks=np.int64(chunks)
            chunks=np.array([0,chunks[0]])+(num_slices-chunks[0])//2
            
        chunks=np.clip(np.array(chunks),0,num_slices)
        num_slices_cropped=np.int(chunks[1]-chunks[0])
        if rank ==0:
            print('='*50)
            print(num_slices_cropped, type(num_slices_cropped), num_slices, type(num_slices))
            print('='*50)
        
    else:
        num_slices_cropped=num_slices
        
    return num_slices_cropped, chunks




def dnames_get():
    dnames={'sino':"exchange/data", 'theta':"exchange/theta", 'tomo':"exchange/tomo", 'rot_center':"exchange/rot_center"}
    return dnames #dname_sino,dname_theta,dname_tomo

# defaults
DDopts={'algo':'iradon', 'GPU': 0, 'shmem':1, 'max_chunk_slice': 16, 'verbose':1, 'max_iter':10, 'tol': 1e-3, 'file_out':'*', 'reg':.5, 'tau':.05, 'ringbuffer':0, 'ncore':None, 'chunks': None, 'Positivity': False }


def reconstruct(sino, theta, rot_center, Dopts, order='sino'):
    if order != 'sino':
        sino=np.swapaxes(sino,0,1)
    #GPU = True
    for keys in DDopts: 
        if keys not in Dopts: Dopts[keys]=DDopts[keys]
    max_chunk=Dopts['max_chunk_slice']
    reg=Dopts['reg']
    tau=Dopts['tau']
    verboseall=Dopts['verbose']
    #max_chunk=Dopts['max_chunk_slice'] 
    ringbuffer=0 # Dopts['ringbuffer']
    max_iter=Dopts['max_iter']
    tol=Dopts['tol']
    ncore=Dopts['ncore']
    algo=Dopts['algo']
    GPU = Dopts['GPU']
    Positivity = Dopts['Positivity']
    
    chunks = Dopts['chunks']
    
    tomo_out, times_loop = recon(sino, theta, algo = algo, tomo_out=None,
              rot_center = rot_center, max_iter = max_iter, tol=tol, 
              GPU = GPU, shmem = False, max_chunk_slice=max_chunk,  
              reg = reg, tau = tau, verbose = verboseall, 
              ncore=ncore, crop=chunks, mpring=ringbuffer, Positivity = Positivity)

    return tomo_out        
    
  

def recon(sino, theta, algo = 'iradon', tomo_out=None, rot_center = None, max_iter = 10, tol=5e-3, GPU = True, shmem = False, max_chunk_slice=16,  reg = None, tau = None, verbose = verboseall,ncore=None, crop=None, mpring=False, Positivity = False):

    def printv(*args,**kwargs): 
        if verbose>0:  printv0(*args,**kwargs)
    
    #timing
    def printvt(*args,**kwargs): 
        if verbose>1:  printv0(*args,**kwargs)
    

    num_slices = sino.shape[0]
    num_angles = sino.shape[1]
    num_rays   = sino.shape[2]
    obj_width  = .98
    tomo = tomo_out

    if rank == 0: 
        if type(tomo) ==type(None):
            tomo = np.empty((num_slices,num_rays,num_rays),dtype = 'float32')      
            tomo_out = tomo


    if type(rot_center)==type(None):
        rot_center = num_rays//2
    
    times={'loop':0, 'setup':0, 'h5read':0, 'solver':0, 'c2g':0, 'g2c':0, 'barrier':0, 'gather':0 }
    times_loop=times.copy()
    
    #cropped output
    loop_offset=0
    if type(crop)!=type(None):
        if len(crop)==1:
            crop=[(num_slices-crop)//2,(num_slices-crop)//2+crop]
        loop_offset=crop[0]
        num_slices=crop[1]-crop[0]
        printv('offset',loop_offset,'chunks',crop)
    

    printv("GPU:{} , algorithm:{}".format(GPU,algo), end=' ')
    if algo in ('cgls','sirt'):
        printv(", maxit:{}".format(max_iter))
    elif algo in ('tv', 'TV'):
        printv(", maxit:{}, reg:{}, tau:{}".format(max_iter,reg,tau))
    else:
        printv('')
        
    
    #printv("tomo shape (",num_slices,num_rays,num_rays, ") n_angles",num_angles, "max_iter",max_iter)
    printv("tomo shape ({},{},{}) n_angles {} max_iter {}".format(num_slices,num_rays,num_rays ,num_angles, max_iter))
    
    printv("max chunk size ", max_chunk_slice,flush=True)
    
    if algo[0:min(len(algo),6)]=='tomopy':
        GPU=False

    from .devmanager import backend
    xp, GPU = backend (GPU, rank)
    
    theta=xp.array(theta)
    

    ##################################################

        

    # set up radon
    printv("setting up the solver. ", end = '')
    
    
    
    start=timer()
    from .wrap_algorithms import wrap
    reconstruct=wrap(sino.shape,theta,rot_center,algo,xp=xp, obj_width=obj_width, max_iter=max_iter, tol=tol, reg=reg, tau=tau, ncore=ncore, verbose=verbose, Positivity = Positivity)   
           
    end = timer()
    time_radonsetup=(end - start)
    times_loop['setup']=time_radonsetup
    printv("time=", time_radonsetup,flush=True)
    
    
    #divide up loop chunks evenly across mpi ranks    
    loop_chunks=get_loop_chunk_slices(num_slices, mpi_size, max_chunk_slice )
    loop_chunks+=loop_offset

    
    printv("nslices:",num_slices," mpi_size:", mpi_size," max_chunk:",max_chunk_slice)
    printv("rank",rank,"loop_chunks:", loop_chunks+loop_offset)
    
    
    
    
    ######################################### 
    # IO setup
            
    ##############################################
    halo=0
    if algo == 'tv': halo = 2 #not used at the moment
    halo+=0
    #print('number of loops',)
    ######### loop through chunks
    #loop_chunks+=loop_offset
    start_loop_time =time.time()

    printv('starting loop')
    for ii in range(loop_chunks.size-1):

        nslices = loop_chunks[ii+1]-loop_chunks[ii]
        chunk_slices = get_chunk_slices(nslices) 
        
        #printv("rank",rank,"size",mpi_size,"loop_chunks:", loop_chunks)
        
        chunks=chunk_slices[rank,:]+loop_chunks[ii]
        printv( 'loop_chunk {}/{}:{}, mpi chunks {}'.format(ii+1,loop_chunks.size-1, loop_chunks[ii:ii+2],loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel()))
        
    
        start_read = time.time()
        
        printv("reading slices:", end = '')    
        
        """
        # halo for TV - to do ...
        #bchunk=np.clip(chunks[0]-halo,0,num_slices)
        #echunk=np.clip(chunks[1]+halo,0,num_slices)

        #bhalo=chunks[0]-np.clip(chunks[0]-halo,0,num_slices-1)
        #ehalo=np.clip(chunks[1]+halo,0,num_slices-1)-chunks[1]
        #data1 = sino[chunks[0]-bhalo:chunks[1]+bhalo,...]
        #chunksi=np.clip(np.arange(chunks[0]-halo,chunks[1]+halo),0,num_slices-1)
        #data xrxrm= sino[chunksi,...]
        
        #data = sino[chunks[0]:chunks[1],...]
"""
        data = sino[chunks[0]:chunks[1],...]
        
        end_read=time.time()

        if rank ==0: times['h5read']=(end_read - start_read)

        printv("\rread time ={:3g}".format(times['h5read']),end=' ')

        # copy to gpu or cpu
        start = timer()
        if GPU:
            data=xp.array(data)
        end = timer()
        times['c2g']=end - start

        
        printv("reconstructing slices, ", end = '') 
        mpi_time=0.
        start_solver = timer()
        tomo[chunks[0]-loop_offset:chunks[1]-loop_offset,...], rnrm, g2ctime =  reconstruct(data,verbose_iter)
        printv('tomo norm', np.sum(np.abs(tomo)),np.sum(data))
        times['solver'] = timer()-start_solver
        
        start_gather = timer()

        mpi_time += timer()-start_gather
        times['gather'] = mpi_time

        
        times['g2c']    = g2ctime

        printv("\r read time={:3g}, solver ={:3g}, gather={:3g}, g2c ={:3g}".format(times['h5read'],times['solver'], times['gather'], times['g2c']),flush=True)
    
        for jj in times: times_loop[jj]+=times[jj]

    ##### end of loop
    

    start = timer()
    mpi_barrier()
    times['barrier']+=timer()-start
    
    if GPU:
        #printv("stopping profiler")
        try:
            xp.cuda.profiler.stop()
        except:
            None
    

    end_loop=time.time()
    times_loop['loop']=end_loop-start_loop_time 


    '''    
    # make sure all buffers are done
    if mpiring:
        time_gather=time.time()    

        if pgather[even]!=0:
            pgather[even].Wait()
        if pgather[1-even]!=0:
            pgather[even].Wait()
        time_last_gather = time.time()-time_gather
        times_loop['gather']+=time_last_gather
    elif mpigather:
        time_gather=time.time()    
        if pgather!=0: pgather.Wait()
        time_last_gather = time.time()-time_gather
        times_loop['gather']+=time_last_gather
    '''
    
    time_write=time.time()
    '''
    # make sure all writing is done
    if mpring>1:
        printv('finish flushing results to disk',flush=True)        
        pw[1-even].join()
        pw[1-even].terminate()
        pw[even].join()
        pw[even].terminate()
        tomo=tomo_out

    #print('rank',rank, '*'*20)        
    
    if rank>0: return None, None
    if mpring>1:
        tomo_out.flush()
    '''
    
    time_write=time.time()-time_write
    times_loop['write']=time_write
    '''
    if not mpigather:
        if type(tomo_out)!=type(None):#  or type(tomo_out)==np.ndarray:
        #print('------------------',mpigather,type(tomo))
            if algo[0:min(len(algo),6)]!='tomopy':
                tomo = tomo_out

    # else:
    #     print('**********######********',mpigather,type(tomo), np.size(tomo))

    
    #
    #    tomo = tomo_out
    '''
    
    return tomo, times_loop

'''
#########################################
# ================== reconstruct ============== #
def double_buffers(mpring,verbose):
    def printv(*args,**kwargs): 
        if verbose>0:  printv0(*args,**kwargs)

    ######################################### 
    # IO setup
    # MP ring buffer setup

    
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
        
         # reading ring buffer (1 or 3)
        if np.mod(mpring,2)==1:
            pr=[0,0] #process even or odd
            data_ring  = shared_array(shape=(2,max_chunk_slice, num_angles, num_rays),dtype=np.float32)
            
            # def read_data(sino, loop_chunks, ii):
            def read_data(ii):
                """no synchronization."""
                even = np.mod(ii+1,2)
                nslices = loop_chunks[ii+1]-loop_chunks[ii]
                chunk_slices = get_chunk_slices(nslices)
                chunks=chunk_slices[rank,:]+loop_chunks[ii]
                data_ring[even,0:chunks[1]-chunks[0],...]= sino[chunks[0]:chunks[1],...]
    
        # writing ring buffer (2 or 3)
        if mpring>1: 
            printv('ring buffer writing to file')
            pw=[0,0] #process even or odd        
            pflush=None # flushing
            tomo_ring  = shared_array(shape=(2,max_chunk_slice, num_rays, num_rays),dtype=np.float32)
    
            def write_tomo(tomo_out,chunks,ii):
                even = np.mod(ii+1,2)
                tomo_out[chunks[0]-loop_offset:chunks[1]-loop_offset,...]=tomo_ring[even,0:chunks[1]-chunks[0],...]
    
            def flush():
                #print('\nflushing',ii)
                tomo_out.flush()
                #print('flushed',ii,'\n')
    # mpi ring buffer
    if mpigather:
        if mpiring:
            printv('ring buffer gatherv')
            #pw=[0,0] #process even or odd   pr=[0,0]     
            pgather=[0,0];#None # process control
            
            tomo_ring  = np.empty((2,max_chunk_slice, num_rays, num_rays),dtype='float32')
            
            tomo=None
            tomo_local=None
            if rank == 0: 
                if type(tomo_out)!=type(None):
                    tomo = tomo_out
                else:
                    tomo = np.empty((num_slices,num_rays,num_rays),dtype = 'float32')
                    
                    
            from .communicator import igatherv #, gatherv
           
        # no mpi ring buffer
        else:
            
            printv('using gatherv distributed mpi- mpring',mpring, flush=True)
                
            #from .communicator import gatherv
            from .communicator import igatherv 
            mpigather=True
            pgather=0
            # gatherv - allocate 
            tomo=None
            tomo_local=None
            if rank == 0: 
                if type(tomo_out)!=type(None):
                    tomo = tomo_out
                else:
                    tomo = np.empty((num_slices,num_rays,num_rays),dtype = 'float32')
                


   
    # MP ring buffer setup
    ######################################### 
    
    ##########################################
    # gather the results: write to memmap, shared memory, gatherv

    if algo[0:min(len(algo),6)]!='tomopy': # not for tomopy-...
        if shmem and (mpring<2):
            if type(tomo_out)!=type(None): #np.memmap:
                tomo=tomo_out
            else:
                from .communicator import allocate_shared
                try:
                    #tomo = allocate_shared((num_slices,num_rays,num_rays),rank,mpi_size)
                    tomo = allocate_shared((num_slices,num_rays,num_rays))
                    
                except:
                    printv(bold+"shared memory is not working, set the flag -S to 0")
                    printv("reverting to mpi gatherv that but it may not work\n",'='*60,endb)
                    shmem=False
   
    else: #  no mpi no ring buffer
        
        tomo=None
        if rank == 0:
            tomo=np.empty((num_slices, num_rays,num_rays),dtype='float32')


'''
