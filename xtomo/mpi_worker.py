
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI
import xtomo
import xtomo.communicator

try:
    comm = MPI.Comm.Get_parent()
    
    rank = comm.Get_rank()
except:
    raise ValueError('Could not connect to parent - ')

# Defaults
DDopts={'algo':'iradon', 'GPU': 0, 'shmem':1, 'max_chunk_slice': 16, 'verbose':1, 'max_iter':10, 'tol': 1e-3, 'file_out':'*', 'reg':.5, 'tau':.05, 'ringbuffer':0, 'ncore':None }

# wrapper for reconstruction from the data
def rrr(sino, theta, rot_center, tomo_out, Dopts):
    
    #num_slices = sino.shape[0]
    #num_rays   = sino.shape[2]
    
    #shape_tomo=(num_slices, num_rays, num_rays)
    
    for keys in DDopts: 
        if keys not in Dopts: Dopts[keys]=DDopts[keys]
        #print('key:',keys)
    
    GPU=Dopts['GPU']
    shmem=Dopts['shmem']
    max_chunk=Dopts['max_chunk_slice']
    reg=Dopts['reg']
    tau=Dopts['tau']
    verboseall=Dopts['verbose']
    #max_chunk=Dopts['max_chunk_slice'] 
    ringbuffer=Dopts['ringbuffer']
    max_iter=Dopts['max_iter']
    tol=Dopts['tol']
    ncore=Dopts['ncore']
    algo=Dopts['algo']

    #file_out=Dopts['file_out']    
    #from xtomo.IO import maptomofile
    #tomo_out, rb = maptomofile(file_out, shape_tomo = shape_tomo, cstring=str(Dopts))
    
    #comm.barrier()
    #print('-'*50,'\nhello',rank,'tomo_out',tomo_out,flush=True)
    #algo='iradon'
    #rot_center=None
    
    chunks=None
    
    #from xtomo.loop_sino import recon 
    from xtomo.loop_sino_simple import recon 
    
    # tomo_out = None 
    # times_loop = None

    tomo_out, times_loop = recon(sino, theta, algo = algo, tomo_out=tomo_out,
              rot_center = rot_center, max_iter = max_iter, tol=tol, 
              GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk,  
              reg = reg, tau = tau, verbose = verboseall, 
              ncore=ncore, crop=chunks, mpring=ringbuffer)
    #if rank==0:
    #    from xtomo.IO import tomosave
    #    tomosave(tomo_out, 0,times_loop)
    #    print('flushed')

    #if file_out != '-1':
            
    return tomo_out, times_loop
    
 
    
comm_intra = comm.Merge(MPI.COMM_WORLD)    

irank = comm_intra.Get_rank()

#print('rank',rank, 'intra -- rank', comm_intra.Get_rank())
#comm.barrier()
#comm_intra.barrier()
#print("barrier, rank",rank )

# initialize variables
Dopts=None

rot_center = None
sino_shape = None
theta_shape = None
tomo_shape = None



#if irank == 1:
#    Dopts = { 'hi':0, 'algo':'iradon', 'rank': rank} 
    
sino_shape = comm_intra.bcast(sino_shape,root=0)    
theta_shape = comm_intra.bcast(theta_shape,root=0)    
rot_center = comm_intra.bcast(rot_center,root=0)  
Dopts=comm_intra.bcast(Dopts,root=0)


num_slices =sino_shape[0]
num_rays = sino_shape[2]
tomo_shape = (num_slices, num_rays, num_rays)


#import time
#time.sleep(5)

shared_sino = xtomo.communicator.allocate_shared(sino_shape, comm = comm_intra)
#shared_sino = xtomo.communicator.allocate_shared(sino_shape, comm = comm_intra)
shared_theta = xtomo.communicator.allocate_shared(theta_shape, comm=comm_intra)
shared_tomo = xtomo.communicator.allocate_shared(tomo_shape, comm = comm_intra)

print('mpi_worker',rank)

# make sure all is transfered
comm_intra.barrier()

import numpy as np
print('passed barrier,  rank',rank,'recieved opts',Dopts, 'sino',np.sum(shared_sino),'theta',np.shape(shared_theta),np.sum(shared_theta))

# reconstruct
tomo_out, times_loop = rrr(shared_sino, shared_theta,  rot_center, shared_tomo, Dopts)



#print('rank',rank,' reconstructed')
comm_intra.barrier()
#print('rank',rank,' disconnecting')

#print(Dopts)
#status = MPI.Status()

#comm_intra.Disconnect()
#comm_intra.Free()
#comm_intra.Free()

#comm.Free()
comm_intra.Free()
#comm_intra.Disconnect()

del comm_intra
#comm.Free()
comm.Disconnect()


#print('rank',rank,' quitting')
#comm.Disconnect()
quit()
#comm.Free()


