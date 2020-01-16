
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI

try:
    comm = MPI.Comm.Get_parent()
    
    rank = comm.Get_rank()
except:
    raise ValueError('Could not connect to parent - ')



def rrr(Dopts):
    fname=Dopts['file_in']
    import h5py
    fid= h5py.File(fname, "r")
    sino  = fid['exchange/data']
    theta = fid['exchange/theta']
    num_slices = sino.shape[0]
    #num_angles = sino.shape[1]
    num_rays   = sino.shape[2]
    
    shape_tomo=(num_slices, num_rays, num_rays)

    
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
    algo=Dopts['algo']
    file_out=Dopts['file_out']
    
    from xtomo.IO import maptomofile
    tomo_out, rb = maptomofile(file_out, shape_tomo = shape_tomo, cstring=str(Dopts))
    
    #comm.barrier()
    #print('-'*50,'\nhello',rank,'tomo_out',tomo_out,flush=True)
    #algo='iradon'
    rot_center=None
    
    chunks=None
    
    from xtomo.loop_sino import recon 
    
    tomo, times_loop = recon(sino, theta, algo = algo, tomo_out=tomo_out, 
              rot_center = rot_center, max_iter = max_iter, tol=tol, 
              GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk,  
              reg = reg, tau = tau, verbose = verboseall, 
              ncore=ncore, crop=chunks, mpring=ringbuffer)

    if rank==0:
        from xtomo.IO import tomosave
        tomosave(tomo_out, 0,times_loop)
        print('flushed')

    #if file_out != '-1':
            
    return tomo, times_loop
    
 
    


comm_intra = comm.Merge(MPI.COMM_WORLD)    

irank = comm_intra.Get_rank()

#print('rank',rank, 'intra -- rank', comm_intra.Get_rank())
#comm.barrier()
#comm_intra.barrier()
#print("barrier, rank",rank )

Dopts=None
if irank == 1:
    Dopts = { 'hi':0, 'algo':'iradon', 'rank': rank} 
    
#comm.bcast(Dopts)
#Dopts=comm.bcast(Dopts,root=1)

#comm.barrier()

#comm.bcast(Dopts,root=0)
Dopts=comm_intra.bcast(Dopts,root=0)
#comm_intra.bcast(Dopts,root=MPI.ROOT)
#comm_intra.barrier()
#print('passed barrier,  rank',rank,'recieved opts',Dopts)
#print('passed barrier,  rank',rank,'recieved opts')


tomo, times_loop = rrr(Dopts)



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


