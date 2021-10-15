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
fname = 'tomo_00001_clean.h5'


h5np=lambda fname,key: np.float32(h5py.File(fname, mode='r')[key][...])
# theta =np.float32(h5py.File(dname+fname, mode='r')['/exchange/theta'][...])
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

# %%
# testing spawning jobs
'''
n_workers= 2
Dopts={ 'algo':'iradon'}

# Dopts={ 'algo':'tv',  'shmem':True, 'GPU': 1 , 'ncore':None,
#        'max_chunk_slice':16, 'ringbuffer':0, 'verbose':True, 
#        'max_iter':10, 'tol':5e-3, 'reg':.5, 'tau':.05}


rot_center = None
from  xtomo.spawn import reconstruct_mpiv as recon

tomo=recon(data,theta,rot_center,n_workers,Dopts)
'''



