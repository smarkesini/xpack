#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:01:39 2020

@author: smarchesini
"""
# spawn jobs and share memory


import h5py
import numpy as np

fname = '/home/smarchesini/data/tomosim/shepp_logan_128_181_256_95.h5'
fid= h5py.File(fname, "r")
sino = fid['/exchange/data'][...]
theta = fid['/exchange/theta/'][...]

rot_center = None
if 'rot_center' in fid['/exchange/']:
    rot_center = fid['/exchange/rot_center']

sino=np.ascontiguousarray(np.swapaxes(sino,0,1))


n_workers= 2

"""
Dopts={ 'algo':'tv',  'shmem':True, 'GPU': 1 , 'ncore':None,
       'max_chunk_slice':16, 'ringbuffer':0, 'verbose':True, 
       'max_iter':10, 'tol':5e-3, 'reg':.5, 'tau':.05}
"""

Dopts={ 'algo':'TV'}


from  xtomo.spawn import reconstruct_mpimm as recon

tomo=recon(sino,theta,rot_center,n_workers,Dopts, order='proj')

