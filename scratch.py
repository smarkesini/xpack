#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import numpy as np
#from xtomo.IO import tomofile, tomosave, print_times

fname='/home/smarchesini/data/tomosim/shepp_logan_128_181_256_95.h5'
       

Dopts={ 'fname':fname, 'algo':'iradon',  'shmem':True, 'GPU':False, 'ncore':None,
       'max_chunk_slice':16, 'ringbuffer':0, 'verbose':True, 
       'max_iter':10, 'tol':5e-3, 'reg':1., 'tau':.05}


from xtomo.IO import getfilename
file_out=getfilename(Dopts['fname'], Dopts['algo'])
print('file_out=',file_out)
Dopts['file_out']=file_out

#---MPI spawn ---
#from xtomo.rr import rr as rr

#tomo, times_loop = rr(Dopts)


## -spawn
n_workers  = 2 
import mpi4py
mpi4py.rc.threads = False # no multithreading...
from mpi4py import MPI


#start_worker = 'worker'

#executable = '/home/smarchesini/anaconda3/bin/python /home/smarchesini/git/xpack/xtomo/testing.py'
executable = '/home/smarchesini/anaconda3/bin/python '

#arg1='/home/smarchesini/git/xpack/xtomo/rr.py'
import xtomo
arg1=xtomo.__path__.__dict__["_path"][0]+'/../worker.py'

import sys
# Spawn workers
comm = MPI.COMM_WORLD.Spawn(
    sys.executable,
    args = [arg1,], #args=[sys.argv[0], start_worker],
    maxprocs=n_workers)

comm_intra = comm.Merge(MPI.COMM_WORLD)    


comm_intra.bcast(Dopts,root=0)

comm_intra.barrier()
# doing the reconstruction
comm_intra.barrier()


comm_intra.Free()
comm.Free()

