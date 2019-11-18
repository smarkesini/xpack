
from __future__ import print_function

#import os
#import sys
#import argparse
import numpy as np

import h5py
#import tomopy
#import dxchange

#from matplotlib import pylab
#import matplotlib.pyplot as plt
#import json
#import collections

#h5fname ='/tomodata/tomobank_clean/tomo_00001_preprocessed.h5'
#h5fname ='/tomodata/tomobank_clean/tomo_00072_preprocessed1.h5'
h5fname ='/fastdata/tomobank_clean/tomo_00072_preprocessed.h5'

#h5fname ='/fastdata/tomobank_clean/tomo_00001_preprocessed.h5'

# f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
#from communicator import comm #, rank, mpi_size

#f= h5py.File(h5fname, "r")
#dims=f['dims']
#csize=(dims[1]*dims[2]*16)*4
#f.close()

csize=0
f= h5py.File(h5fname, "r",rdcc_nbytes=csize)

#if type(comm)==type(None):
#    #print("h5 not parallel")
#    f= h5py.File(h5fname, "r")
#else:
#    f= h5py.File(h5fname, "r")
#    #print("h5parallel")
#    #f= h5py.File(h5fname, "r", driver='mpio', comm=comm)

def read_h5(dirname="data",chunks=None):   
    if type(chunks)==type(None):
        return f[dirname][:]
    else:
        return f[dirname][chunks[0]:chunks[1],...]

    #print(data.shape)
    #return data

def get_dims():
    return f['sino'].shape#[:]
    #return f['dims']#[:]
    #shape = read_h5(dirname="dims")
    #return shape

def get_sino(h5fname, chunks=None):
    return f['sino']

#    if type(chunks)==type(None):
#        return f['sino'][...]
#    else:
#        return f['sino'][chunks[0]:chunks[1],...]
    #return read_h5('sino',chunks)

def get_theta():
    return f['theta']#[:]
    #return read_h5('theta')


data_size = get_dims()
#print("data shape",data_size)
all_chunks = (0, data_size[0])

    
def get_data(kind,chunks=all_chunks):
#    if type(chunks)!=type(None):
#        return f[kind][:]
#    else:
#        return f[kind][chunks[0]:chunks[1],...]
        #return f[kind][]
    
    if   kind =='dims': 
        return f['sino'].shape
    elif kind =='theta':
        return f['theta']
#        theta = get_theta()
#        #print("theta type",type(theta),"theta dtype",theta.dtype,"theta shape",theta.shape, "theta", theta*180/np.pi)
#        return theta
    elif kind == 'sino': 
        return f['sino']
        
#        if type(chunks)!=type(None):
#            return get_sino(h5fname, (chunks[0],chunks[1]) )
#        else:
#            return get_sino(h5fname,(1,2))
    elif kind == 'tomo':
        print('no tomo')
        return None


 #----------------------
#
#h5fname ='/data/tomobank/tomo_00001/tomo_00001.h5'
#
#
#rot_center = 1024
##
#def get_theta():
#    proj, flat, dark, theta = dxchange.read_aps_32id(h5fname, sino=1)
#    return theta
##
#
#
#def get_dims():
#    """
#    Read array size of a specific group of Data Exchange file.
#
#    Parameters
#    ----------
#    fname : str
#        String defining the path of file or file name.
#    dataset : str
#        Path to the dataset inside hdf5 file where data is located.
#
#    Returns
#    -------
#    ndarray
#        Data set size.
#    """
#    'data'
#    grp = '/'.join(['exchange', 'data'])
#
#    with h5py.File(h5fname, "r") as f:
#        try:
#            data = f[grp]
#        except KeyError:
#            return None
#
#        shape = data.shape
#
#    shape=( shape[1],shape[0],shape[2])
#
#    return shape
#
#def get_sino(h5fname, chunks):
#
#    # Read APS 32-BM raw data.
#    proj, flat, dark, theta = dxchange.read_aps_32id(h5fname, sino=chunks)
#       
##    # Manage the missing angles:
##    if blocked_views is not None:
##        print("Blocked Views: ", blocked_views)
##        proj = np.concatenate((proj[0:blocked_views[0],:,:], proj[blocked_views[1]+1:-1,:,:]), axis=0)
##        theta = np.concatenate((theta[0:blocked_views[0]], theta[blocked_views[1]+1:-1]))
#
#    # Flat-field correction of raw data.
#    data = tomopy.normalize(proj, flat, dark, cutoff=1.4)
#    #data = tomopy.normalize(proj, flat, dark, cutoff=(1,2))
#
#    # remove stripes
#    data = tomopy.remove_stripe_fw(data,level=7,wname='sym16',sigma=1,pad=True)
#
#    #print("Raw data: ", h5fname)
#    #print("Center: ", rot_center)
#
#    data = tomopy.minus_log(data)
#
#    data = tomopy.remove_nan(data, val=0.0)
#    data = tomopy.remove_neg(data, val=0.00)
#    data[np.where(data == np.inf)] = 0.00
#
#    data=np.swapaxes(data,0,1)
#    
#    return  data
#
#all_chunks = None
#
#def get_data(kind,chunks=all_chunks):
#    if   kind =='dims': return get_dims()
#    elif kind =='theta': 
#        theta=get_theta()
#        theta = np.float32(theta)
#        #print("theta type",type(theta),"theta dtype",theta.dtype,"theta shape",theta.shape, "theta", theta*180/np.pi)
#        return theta
#    elif kind == 'sino': 
#        if type(chunks)!=type(None):
#            return get_sino(h5fname, (chunks[0],chunks[1]) )
#        else:
#            return get_sino(h5fname,(1,2))
#    elif kind == 'tomo':
#        print('no tomo')
#        return None
#
#data_size = get_data('dims')
##print("data shape",data_size)
#all_chunks = (0, data_size[0])
#"""
#"""
#
#nsino = 0.5
#start = int(data_size[0] * nsino)
#chunk = (start, start+2)
#
#print("chunk",chunk)
#
#theta = get_data('theta')
#
#data = get_data('sino', chunk)
#
#
##data = get_sino(h5fname, chunk)
#
#
###
#
#rec = tomopy.recon(data, theta, center=rot_center, sinogram_order=True, algorithm='gridrec')
#rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)
#
## proj, flat, dark, theta = dxchange.read_aps_32id(fname, sino=sino)
##print('size proj',proj.shape,"shape flat",flat.shape,"shape dark",dark.shape,"shape theta",theta.shape)
#
## Reconstruct
##rec,data,theta = reconstruct(fname, sino, rot_center, blocked_views)
#print("shape data",data.shape,"shape theta",theta.shape,"shape rec",rec.shape)
#
#pylab
#
##plt.imshow(data[:,0,:])
#plt.imshow(data[0,:,:])
#plt.show()
#tomo_slice = rec
##sino=np.swapaxes(data,0,1)
#
#
#plt.imshow(tomo_slice[0])
#plt.show()
#
