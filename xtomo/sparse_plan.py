#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparse_plan
"""

cache='.cache/'
import os
os.mkdir(cache)
import numpy as np

def get_hash_name(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter):
    if type(center) == type(None): center=num_rays//2 
    hs=hash(bytes(np.array([num_rays,theta.shape[0],center,k_r]))+bytes(theta)+bytes(kernel_type+dcfilter,encoding='utf8'))
    fname=cache+'sparse_'+tipe+'_'+str(hs)+'.npz'
    return fname

def save_sparse_plan(Sc,tipe, num_rays, theta, center=None, kernel_type = 'gaussian', k_r = 1, dcfilter='none_filter'):   
    fname=get_hash_name(tipe,num_rays, theta,  center, kernel_type, k_r, dcfilter)
    np.savez_compressed(fname, **Sc)
       
def load_sparse_plan(tipe, num_rays, theta, center=None, kernel_type = 'gaussian', k_r = 1, dcfilter='none_filter'):
    if type(center) == type(None): center=num_rays//2 
    fname=get_hash_name(tipe, num_rays, theta,  center, kernel_type, k_r, dcfilter)
     
    K=np.load(fname)
    return K

arrays_dict = {'val':Sc.data,'ind':Sc.indices, 'indptr':Sc.indptr}
"""
    arrays_dict = {}
    if matrix.format in ('csc', 'csr', 'bsr'):
        arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    elif matrix.format == 'dia':
        arrays_dict.update(offsets=matrix.offsets)
    elif matrix.format == 'coo':
        arrays_dict.update(row=matrix.row, col=matrix.col)
    else:
        raise NotImplementedError('Save is not implemented for sparse matrix of format {}.'.format(matrix.format))
    arrays_dict.update(
        format=matrix.format.encode('ascii'),
        shape=matrix.shape,
        data=matrix.data
    )
    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)
"""

