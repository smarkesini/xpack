import sys
executable = sys.executable
import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI

def reconstruct_mpi(fname, n_workers, Dopts):
    
    Dopts['file_in']=fname
    from xtomo.IO import getfilename
    file_out=getfilename(Dopts['file_in'], Dopts['algo'])
    #print('file_out=',file_out)
    Dopts['file_out']=file_out
    

    import xtomo


    arg1=xtomo.__path__.__dict__["_path"][0]+'/mpi_worker.py'
    
    comm = MPI.COMM_WORLD.Spawn(
        executable,
        args = [arg1,], #args=[sys.argv[0], start_worker],
        maxprocs=n_workers)
    
    comm_intra = comm.Merge(MPI.COMM_WORLD)    
    
    
    comm_intra.bcast(Dopts,root=0)
    
    # make sure we sent the data
    # doing the reconstruction
    comm_intra.barrier()
    
    

    comm_intra.Free()
    del comm_intra    
    comm.Disconnect()
    #comm.Free()


def reconstruct_mpiv(sino, theta, rot_center, n_workers, Dopts):
    
    #Dopts['file_in']=fname
    #from xtomo.IO import getfilename
    #file_out=getfilename(Dopts['file_in'], Dopts['algo'])
    #print('file_out=',file_out)
    #Dopts['file_out']=file_out

    import xtomo
    import xtomo.communicator
    
    arg1=xtomo.__path__.__dict__["_path"][0]+'/mpi_worker_1.py'

    
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


    # sinogram in shared memory
    #shared_sino = xtomo.communicator.allocate_shared(sino_shape)
    #shared_theta = xtomo.communicator.allocate_shared(theta_shape)

    shared_sino = xtomo.communicator.allocate_shared(sino_shape, comm = comm_intra)
    print('allocated shared')
    shared_theta = xtomo.communicator.allocate_shared(theta_shape, comm=comm_intra)
    
    shared_sino[...]=sino
    shared_theta[...]=theta
    #shared_tomo = xtomo.communicator.allocate_shared(tomo_shape) 
    shared_tomo = xtomo.communicator.allocate_shared(tomo_shape, comm = comm_intra)

    # make sure we sent the data
    comm_intra.barrier()

    # doing the reconstruction
    
    comm_intra.barrier()
    import numpy as np
    print("shared tomo and data",np.sum(shared_tomo),np.sum(shared_sino),np.sum(sino))
    

    comm_intra.Free()
    del comm_intra    
    comm.Disconnect()
    return shared_tomo
    #comm.Free()

