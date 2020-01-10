
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:49:17 2020

@author: smarchesini
"""

#import loop_sino.recon


def rr(Dopts):
    fname=Dopts['fname']
    import h5py
    fid= h5py.File(fname, "r")
    sino  = fid['exchange/data']
    theta = fid['exchange/theta']

    
#    from .communicator import rank, mpi_size
    
    #from xtomo.loop_sino import recon as recon
    
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
    rot_center=None
    algo='iradon'
    tomo_out=None
    chunks=None
    
    from xtomo.loop_sino import recon 
    
    tomo, times_loop = recon(sino, theta, algo = algo, tomo_out=tomo_out, 
              rot_center = rot_center, max_iter = max_iter, tol=tol, 
              GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk,  
              reg = reg, tau = tau, verbose = verboseall, 
              ncore=ncore, crop=chunks, mpring=ringbuffer)
    return tomo, times_loop
    
 
    

import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI

try:
    comm = MPI.Comm.Get_parent()
    
    rank = comm.Get_rank()
except:
    raise ValueError('Could not connect to parent - ')

comm_intra = comm.Merge(MPI.COMM_WORLD)    

irank = comm_intra.Get_rank()

print('rank',rank, 'intra rank', comm_intra.Get_rank())
#comm.barrier()
comm_intra.barrier()
print("barrier, rank",rank )

Dopts=None
if irank == 1:
    Dopts = { 'hi':0, 'algo':'iradon', 'rank': rank} 
    
#comm.bcast(Dopts)
#Dopts=comm.bcast(Dopts,root=1)

#comm.barrier()

#comm.bcast(Dopts,root=0)
Dopts=comm_intra.bcast(Dopts,root=0)
#comm_intra.bcast(Dopts,root=MPI.ROOT)
comm_intra.barrier()

tomo, times_loop = rr(Dopts)


print('rank',rank,'done')
comm_intra.barrier()
print('rank',rank,'opts',Dopts)

#print(Dopts)
#status = MPI.Status()

#comm_intra.Disconnect()
comm.Disconnect()


