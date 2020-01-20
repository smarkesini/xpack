import sys
executable = sys.executable

def reconstruct_mpi(fname, n_workers, Dopts):
    
    Dopts['file_in']=fname
    from xtomo.IO import getfilename
    file_out=getfilename(Dopts['file_in'], Dopts['algo'])
    #print('file_out=',file_out)
    Dopts['file_out']=file_out
    
    
    
    ## -spawn
    #n_workers  = 2 
    import mpi4py
    mpi4py.rc.threads = False # no multithreading...
    from mpi4py import MPI
    
    
    #start_worker = 'worker'
    
    #executable = '/home/smarchesini/anaconda3/bin/python /home/smarchesini/git/xpack/xtomo/testing.py'
    #executable = '/home/smarchesini/anaconda3/bin/python'
    
    #arg1='/home/smarchesini/git/xpack/xtomo/rr.py'
    import xtomo
    #arg1=xtomo.__path__.__dict__["_path"][0]+'/../worker.py'
    arg1=xtomo.__path__.__dict__["_path"][0]+'/worker.py'
    
    
    # Spawn workers
    comm = MPI.COMM_WORLD.Spawn(
        executable,
        args = [arg1,], #args=[sys.argv[0], start_worker],
        maxprocs=n_workers)
    
    comm_intra = comm.Merge(MPI.COMM_WORLD)    
    
    
    comm_intra.bcast(Dopts,root=0)
    
    #comm_intra.barrier() # make sure we sent the data
    # doing the reconstruction
    comm_intra.barrier()
    
    
    #comm_intra.Disconnect()
    comm_intra.Free()
    del comm_intra    
    comm.Disconnect()
    #comm.Free()

