#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:06:22 2021

@author: smarchesini
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt

dname = '/tomodata/tomobank/tomo_00001/'
fname_in = 'tomo_00001.h5'
fname_out = 'tomo_00001_clean.h5'

# clean up the data
import os
if os.path.isfile(dname+fname_out)==False:
    from xtomo.prep import clean_raw
    clean_raw(dname+fname_in, dname+fname_out, max_chunks = 16)

fname = fname_out

# load the data
h5np=lambda fname,key: np.float32(h5py.File(fname, mode='r')[key][...])

theta =h5np(dname+fname,'/exchange/theta')

# get a small chunk to test
data = np.float32(h5py.File(dname+fname, mode='r')['/exchange/data'][896-32:896+32,...])
# data =h5np(dname+fname,'/exchange/data')
# data = np.swapaxes(data,0,1)


fid = h5py.File(dname+fname, mode='r')
rot_center = None
if 'rot_center' in fid['/exchange/']:
    rot_center = fid['/exchange/rot_center']
    


# %%
#  spawning mpi jobs on 2 gpus
from xtomo.spawn import xtomo_reconstruct
Dopts={ 'algo':'iradon', 'GPU': True, 'n_workers' : 2 }

tomo1=xtomo_reconstruct(data,theta,rot_center, Dopts)
plt.figure()
plt.imshow(tomo1[8])

# %%

Dopts={ 'algo':'TV', 'GPU': True, 'n_workers' : 1 , 'reg': .02, 'max_chunk_slice': 16}

tomo=xtomo_reconstruct(data,theta,rot_center, Dopts)

plt.figure()
plt.imshow(tomo[8])


'''
# %% no loops
from xtomo.wrap_algorithms import wrap
import cupy as cp
reconstruct=wrap(data.shape,theta,rot_center,'iradon' ,xp=cp)
gdata=cp.array(data)
tomo, nrm, timing= reconstruct(gdata,1)
plt.imshow(tomo[8])
plt.draw()
del gdata

# %% with loops (chunking) to fit in memory

from xtomo.loop_sino import recon
tomo, times_loop= recon(data, theta, rot_center=rot_center, algo = 'tv', reg=.02, tau=0.05)

plt.imshow(tomo[8])
plt.draw()
# %%
from xtomo.loop_sino_simple import recon
tomo, times_loop= recon(data, theta, rot_center=rot_center, algo = 'iradon')
plt.figure()
plt.imshow(tomo[8])

plt.draw()
'''

    



