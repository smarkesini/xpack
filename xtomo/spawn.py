import sys
executable = sys.executable
import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI


def xtomo_reconstruct(data, theta, rot_center='None', Dopts=None, order='sino'):
    if order != 'sino':
       data=np.swapaxes(data,0,1)
    if type(Dopts)==type(None):
        Dopts={ 'algo':'iradon', 'GPU': True, 'n_workers' : 1 }            
    if Dopts['n_workers']==1:
        from xtomo.loop_sino_simple import reconstruct
        tomo = reconstruct(data, theta, rot_center, Dopts)
    else:
        from  xtomo.spawn import reconstruct_mpiv as recon
        tomo=recon(data,theta,rot_center, Dopts)
    return tomo



# reconstruct from file
def reconstruct_mpi(fname, n_workers, Dopts):
    
    Dopts['file_in']=fname
    from xtomo.IO import getfilename
    file_out=getfilename(Dopts['file_in'], Dopts['algo'])
    #print('file_out=',file_out)
    Dopts['file_out']=file_out
    

    import xtomo


    arg1=xtomo.__path__.__dict__["_path"][0]+'/mpi_worker_FIO.py'
    
    comm = MPI.COMM_WORLD.Spawn(
        executable,
        args = [arg1,], #args=[sys.argv[0], start_worker],
        maxprocs=n_workers)
    
    comm_intra = comm.Merge(MPI.COMM_WORLD)    
    
    # broadcast the options
    comm_intra.bcast(Dopts,root=0)
    
    # make sure we sent the data
    # doing the reconstruction
    comm_intra.barrier()
    
    

    comm_intra.Free()
    del comm_intra    
    comm.Disconnect()
    #comm.Free()


# reconstruct from numpy arrays to numpy array 

def reconstruct_mpiv(sino, theta, rot_center, Dopts):
    
    # Dopts['n_workers']=n_workers
    
    #Dopts['file_in']=fname
    #from xtomo.IO import getfilename
    #file_out=getfilename(Dopts['file_in'], Dopts['algo'])
    #print('file_out=',file_out)
    #Dopts['file_out']=file_out

    import xtomo
    import xtomo.communicator
    
    #    arg1=xtomo.__path__.__dict__["_path"][0]+'/mpi_worker.py'
    arg1=xtomo.__path__[0]+'/mpi_worker.py'

    
    comm = MPI.COMM_WORLD.Spawn(
        executable,
        args = [arg1,], #args=[sys.argv[0], start_worker],
        maxprocs=Dopts['n_workers'])
    
    comm_intra = comm.Merge(MPI.COMM_WORLD)    
    
    print('spawned size',comm_intra.Get_size()-1)
    # broadcast the data   
    sino_shape = sino.shape
    theta_shape = theta.shape
    num_slices =sino_shape[0]
    num_rays = sino_shape[2]
    tomo_shape = (num_slices, num_rays, num_rays)

    comm_intra.bcast(sino_shape,root=0)    
    comm_intra.bcast(theta_shape,root=0)    
    comm_intra.bcast(rot_center,root=0)    
    comm_intra.bcast(Dopts,root=0)


    # IO in shared memory

    shared_sino = xtomo.communicator.allocate_shared(sino_shape, comm = comm_intra)
    shared_theta = xtomo.communicator.allocate_shared(theta_shape, comm=comm_intra)
    shared_tomo = xtomo.communicator.allocate_shared(tomo_shape, comm = comm_intra)
    print('allocated shared')
    
    shared_sino[...]=sino
    shared_theta[...]=theta
 
    # make sure we sent the data
    comm_intra.barrier()

    # do the reconstruction in spawned mpi calls
    
    comm_intra.barrier()
    # import numpy as np
    # print("shared tomo and data",np.sum(shared_tomo),np.sum(shared_sino),np.sum(sino))

    # done with the job    

    comm_intra.Free()
    del comm_intra    
    comm.Disconnect()
    del shared_sino, shared_theta, xtomo
    
    return shared_tomo
    #comm.Free()
    

import tempfile
tempdir = tempfile.gettempdir()+'/xpack/'

import os.path as path
import os
try:
    os.mkdir(tempdir)
except: 
    None
        

import numpy as np

# memory mapping file
def mmdata(shp, fname, mode='w+'):
    filename = path.join(tempdir, fname)
    
    fp = np.memmap(filename, dtype='float32', mode=mode, shape=shp)
    return fp
    #shared_sino = xtomo.communicator.allocate_shared(sino_shape, comm = comm_intra)

def reconstruct_mpimm(sino, theta, rot_center, n_workers, Dopts, order='sino'):

    import xtomo
    import xtomo.communicator
    import numpy as np
    
    if order == 'proj':
        sino=np.swapaxes(sino,0,1)
    
    arg1=xtomo.__path__.__dict__["_path"][0]+'/mpi_worker_mm.py'
    
    comm = MPI.COMM_WORLD.Spawn(
        executable,
        args = [arg1,], #args=[sys.argv[0], start_worker],
        maxprocs=n_workers)
    
    comm_intra = comm.Merge(MPI.COMM_WORLD)    
    
    print('spawned size',comm_intra.Get_size())


    # broadcast the data   
    sino_shape = sino.shape
    theta_shape = theta.shape
    num_slices =sino_shape[0]
    num_rays = sino_shape[2]
    tomo_shape = (num_slices, num_rays, num_rays)

    comm_intra.bcast(sino_shape,root=0)    
    comm_intra.bcast(theta_shape,root=0)    
    comm_intra.bcast(rot_center,root=0)    
    comm_intra.bcast(Dopts,root=0)
    
    #print('broadcasted')

    # sinogram in shared memory, which is memory mapped

    shared_sino = mmdata(sino_shape, 'data', mode='w+')
    #print('sinogram on file')

    shared_theta = mmdata(theta_shape, 'theta', mode = 'w+')
    shared_tomo =mmdata(tomo_shape, 'tomo', mode='w+')
    
    shared_sino[...]=sino
    shared_theta[...]=theta  

    #print('tomo on file')

    # make sure the files exist
    comm_intra.barrier()

    # doing the reconstruction
    comm_intra.barrier()

    
    comm_intra.barrier()

    

    comm_intra.Free()
    del comm_intra    
    comm.Disconnect()
    return shared_tomo
    #comm.Free()

