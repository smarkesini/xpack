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

