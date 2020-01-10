#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import numpy as np
#from xtomo.IO import tomofile, tomosave, print_times

fname='/home/smarchesini/data/tomosim/shepp_logan_128_181_256_95.h5'

        
Dopts={ 'fname':fname, 'algo':'iradon',  'shmem':True, 'GPU':False, 'ncore':None,
       'max_chunk_slice':16, 'ringbuffer':0, 'verbose':True, 
       'max_iter':10, 'tol':5e-3, 'reg':1., 'tau':.05}

#---MPI spawn ---
#from xtomo.rr import rr as rr

#tomo, times_loop = rr(Dopts)


## -spawn
n_workers  = 2 
import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI


#start_worker = 'worker'

executable = '/home/smarchesini/anaconda3/bin/python /home/smarchesini/git/xpack/xtomo/testing.py'
#arg1='/home/smarchesini/git/xpack/xtomo/rr.py'
import xtomo
arg1=xtomo.__path__.__dict__["_path"][0]+'/../worker.py'

import sys
# Spawn workers
comm = MPI.COMM_WORLD.Spawn(
    sys.executable,
    args = [arg1,], #args=[sys.argv[0], start_worker],
    maxprocs=n_workers)

#parent = MPI.Comm.Get_parent()

#try:
#    parent_size=parent.Get_size()
#    
#except:
#    parent_size=0   

comm_intra = comm.Merge(MPI.COMM_WORLD)    

print("main")
#comm.barrier()
comm_intra.barrier()
print("main 1")
#comm.bcast(Dopts,root=MPI.ROOT)
#Dopts=
comm_intra.bcast(Dopts,root=0)

comm_intra.barrier()
# doing the reconstruction

#comm.barrier()

#comm_intra.bcast(Dopts,root=0)
#comm_intra.bcast(Dopts,root=MPI.ROOT)
print("main 2 ", "dopts",Dopts)
comm_intra.barrier()

print("main 3 ", "dopts",Dopts)
        
status = MPI.Status()
#comm_intra.Disconnect()

comm.Disconnect()

#
