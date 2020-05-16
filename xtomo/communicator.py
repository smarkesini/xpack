#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys


size = 1
rank = 0
comm = None

#print("importing mpi",flush=True)

try: 
    from mpi4py import MPI
except ImportError: pass

try: 
    import cupy as cp
except ImportError: pass

mpi_enabled = "mpi4py" in sys.modules


if mpi_enabled:
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
else:
    print("mpi not enabled")

#print("imported mpi", flush=True)


mpi_size=size


def printv(string):
    if rank == 0:
        print(string)
def mpi_barrier():
    if type(comm)==type(None): return
    if mpi_enabled:  comm.Barrier()
        
    
#from xcale.communicator import  rank, size

#from mpi4py import MPI


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
    if np.isinf(mc): 
#        print("ms",ms)
        return np.array([0,ns],dtype='int64')
#    print("glc ms",ms)

    ls=np.int(np.ceil(ns/(ms*mc)))    
    # nreduce: how many points we overshoot if we use max_chunk everywhere
    nr=ls*mc*ms-ns
    #print(nr,ls,mc)
    # number of reduced loop_chunks:
    
    #cr=np.ceil(nr/ms/ls)
    # make it a multiple of 2 since we do 2 slices at once
    cr=np.ceil(nr/ms/ls/2)*2

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

def igatherv(data_local,chunk_slices, data = None): 
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
    req=comm.Igatherv(sendbuf=[data_local, mpichunktype],recvbuf=[data,(cnt*sdim,None),mpichunktype])
    mpichunktype.Free()
    
    return req

    


def allocate_shared(shape_obj, comm=comm):

    #global size, rank
    #mpi_size = size
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    
    print('allocating shared mem',mpi_size,mpi_rank)
    
    if mpi_size == 1:
       return np.empty(shape_obj,dtype='float32') 
    
    slice_size =np.prod(shape_obj)
    
    itemsize = MPI.FLOAT.Get_size() 
    if mpi_rank == 0: 
        nbytes = slice_size* itemsize 
    else: 
        nbytes = 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm) 
    print('allocated shared mem',mpi_size,mpi_rank)

    # int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
    #                        void *baseptr, MPI_Win * win)
    
    # create a numpy array whose data points to the shared mem
    buf, itemsize = win.Shared_query(0) 
    #print("buf",type(buf),'itemsize',itemsize)
    assert itemsize == MPI.FLOAT.Get_size() 
    obj = np.ndarray(buffer=buf, dtype='f', shape=shape_obj) 
    return obj

#def attach_shared(obj, rank, mpi_size):
#    if mpi_size == 1:
#        return obj
#    itemsize = MPI.FLOAT.Get_size() 

    ## win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm) 
    # win = MPI.Win.Create_dynamic( comm=comm) 
    # MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size)
    # MPI.Win.Attach(mem)
    # MPI_Win_attach(win, base, size)
# MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win * win)
# attached using the function 
# MPI_Win_attach:  MPI.Win.Attach(self, memory)
 
def allocate_shared_tomo(num_slices,num_rays):
    tomo = allocate_shared((num_slices, num_rays,num_rays))
    return tomo


    """
    #########################
    # scatter the initial object    
    slice_shape=(num_rays,num_rays)

    truth=scatterv(true_obj,chunk_slices+loop_chunks[ii],slice_shape)    
    truth=xp.array(truth)
    # generate data
    data = radon(truth)
    del truth
    #########################
    # gatherv - allocate 
    tomo=None
    if rank == 0: tomo = np.empty((nslices,num_rays,num_rays),dtype = 'float32')
    #########################
    # gatherv
    if rank ==0: 
        ptomo = tomo[loop_chunks[ii]:loop_chunks[ii+1],:,:]
    else:
        ptomo=None
    gatherv(tomo_chunk,chunk_slices,data=ptomo)
    #########################
    """
    
def mpi_allGather(send_buff, heterogeneous_comm = True, mode = "cuda"):

    global size, rank, mpi_enabled

    if heterogeneous_comm and mode=="cuda":
        #recv_buff = cp.asnumpy(recv_buff)
        send_buff = cp.asnumpy(send_buff)

    if size > 1 and mpi_enabled:
        recv_buff = MPI.COMM_WORLD.allgather(send_buff)
    else:
        recv_buff = [send_buff]

    if heterogeneous_comm and mode=="cuda":
        recv_buff = cp.asarray(recv_buff)
        send_buff = cp.asarray(send_buff)

    return recv_buff