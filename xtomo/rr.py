#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI

try:
    comm = MPI.Comm.Get_parent()
    
    rank = comm.Get_rank()
except:
    raise ValueError('Could not connect to parent - ')

Dopts=None
comm.bcast(Dopts)
print(Dopts)
status = MPI.Status()

comm.Disconnect()
"""

#comm_intra = comm.Merge(MPI.COMM_WORLD)  
#ranki = comm_intra.Get_rank()

######################3


def rr(Dopts):
    fname=Dopts['fname']
    import h5py
    fid= h5py.File(fname, "r")
    sino  = fid['exchange/data']
    theta = fid['exchange/theta']
    
    
    from xtomo.loop_sino import recon as recon
    GPU=Dopts['GPU']
    shmem=Dopts['shmem']
    max_chunk=Dopts['max_chunk_slice']
    reg=Dopts['reg']
    tau=Dopts['tau']
    verboseall=Dopts['verbose']
    max_chunk=Dopts['max_chunk_slice'] 
    ringbuffer=Dopts['ringbuffer']
    max_iter=Dopts['max_iter']
    tol=Dopts['tol']
    ncore=Dopts['tol']
    file_out = Dopts['file_out']
    rot_center=None
    algo='iradon'
    tomo_out=None
    chunks=None
    
    
    #tomo_out, ring_buffer = tomofile(file_out, file_in=fname, algo=algo, shape_tomo=(num_slices_cropped,num_rays,num_rays), ring_buffer=ringbuffer)
    from .IO import maptomofile

    tomo_out, rb = maptomofile(file_out, cstring=str(Dopts))
    #print(tomo_out)
    from .communicator import rank#, mpi_size, get_loop_chunk_slices, get_chunk_slices, mpi_barrier
    
    print("starting", rank)
    tomo, times_loop = recon(sino, theta, algo = algo, tomo_out=tomo_out, 
              rot_center = rot_center, max_iter = max_iter, tol=tol, 
              GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk,  
              reg = reg, tau = tau, verbose = verboseall, 
              ncore=ncore, crop=chunks, mpring=ringbuffer)
    
    print("rdone", rank)
    
    return tomo, times_loop
    
    
