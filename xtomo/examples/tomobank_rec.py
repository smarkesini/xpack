#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:06:22 2021

@author: smarchesini
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt

dname = '/tomodata/tomobank/tomo_00002/'
fname_in = 'tomo_00002.h5'
#dname = '/tomodata/tomobank/tomo_00004/'
#dname = '/tomodata/tomobank/tomo_00072/'
#fname_in = 'tomo_00072.h5'

fname_out = 'tomo_00002_clean.h5'


h5np=lambda fname,key: np.float32(h5py.File(fname, mode='r')[key][...])

theta =h5np(dname+fname_in,'/exchange/theta')

# %%
import os
try:
    os.remove(dname+fname_out)
except:
    pass

# clean up the data and swap axes to sinogram order
import os
if os.path.isfile(dname+fname_out)==False:
    from xtomo.prep import clean_raw
    clean_raw(dname+fname_in, dname+fname_out, max_chunks = 4, chunks=4, stripe_algo='bm3d_streak')
    # clean_raw(dname+fname_in, dname+fname_out, max_chunks = 16, chunks=16, stripe_algo='None')
    # clean_raw(dname+fname_in, dname+fname_out, max_chunks = 32, stripe_algo='vo_et_al')

# 


# %%
fname = fname_out

# fid = h5py.File(dname+fname, mode='r')

# get a small chunk to test
#data = np.float32(h5py.File(dname+fname, mode='r')['/exchange/data'][896-32:896+32,...])
#data = np.float32(h5py.File(dname+fname, mode='r')['/exchange/data'][...])
#theta = np.float32(h5py.File(dname+fname, mode='r')['/exchange/theta'][...])

data = h5np(dname+fname,'/exchange/data')
theta = h5np(dname+fname,'/exchange/theta')
# data = np.swapaxes(data,0,1)

num_angles=theta.size
subsample_angle=1
theta1=theta[0:num_angles:subsample_angle]
data1=data[:,0:num_angles:subsample_angle,:]



# theta=theta*np.pi/180.


try:
    rot_center = np.float32(h5py.File(dname+fname, mode='r')['/exchange/rot_center'][...])
except:
    rot_center = None
    
# tomo_00002 has a rotation center offset:
rot_center = data.shape[2]//2+6.5
    


## %%
#  spawning mpi jobs on 2 gpus
import xtomo
# from xtomo.spawn import xtomo_reconstruct
Dopts={ 'algo':'iradon', 'GPU': True, 'n_workers' : 1 }

tomo2=xtomo.recon(data1,theta1,rot_center, Dopts)
plt.figure()
plt.imshow(tomo2[1])
plt.title('iradon')

del tomo2
# %%

#Dopts={ 'algo':'TV', 'GPU': True, 'n_workers' : 2 ,  'max_chunk_slice': 20, 'Positivity': False, 'reg':0.0125}
Dopts={ 'algo':'TV', 'GPU': True, 'n_workers' : 1 ,  'max_chunk_slice': 16, 'Positivity': False, 'reg':0.0125, 'cgsmaxit':4}
#data1=data+np.max(data)
tomo4=xtomo.recon(data1,theta1,rot_center, Dopts)

plt.figure()
plt.imshow(tomo4[1])
plt.title('TV')


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

    



