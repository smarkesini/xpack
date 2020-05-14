from xtomo.spawn import reconstruct_mpi

import os
fname=os.path.expanduser('~/data/tomosim/shepp_logan_128_181_256_95.h5')

n_workers= 2

"""
Dopts={ 'algo':'tv',  'shmem':True, 'GPU': 1 , 'ncore':None,
       'max_chunk_slice':16, 'ringbuffer':0, 'verbose':True, 
       'max_iter':10, 'tol':5e-3, 'reg':.5, 'tau':.05}
"""

Dopts={ 'algo':'iradon'}

reconstruct_mpi(fname, n_workers, Dopts)

