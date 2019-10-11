#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparse_plan
"""
import os
import numpy as np

cache='.cache/'

from timeit import default_timer as timer


if not os.path.isdir(cache):
    os.mkdir(cache)

def get_hash_name(tipe, num_rays, theta, center=None, kernel_type="gaussian", k_r=1, dcfilter=None):
    if type(center) == type(None): center=num_rays//2 
    if type(dcfilter) == type(None):  dcfilter=0
    
    hs=hash(bytes(np.array([num_rays,theta.shape[0],center,k_r]))+bytes(theta)+bytes(dcfilter)+bytes(tipe+kernel_type,encoding='utf8'))
    fname=cache+'sparse_'+tipe+'_'+str(hs)+'.npz'
    return fname

def save(Sc,tipe, num_rays, theta, center=None, kernel_type = 'gaussian', k_r = 1, dcfilter='none_filter'):   
    start=timer()
    
    fname=get_hash_name(tipe,num_rays, theta,  center, kernel_type, k_r, dcfilter)
    print('hash time',timer()-start)
    start=timer()
    K = {'val':Sc.data,'ind':Sc.indices, 'indptr':Sc.indptr,'shape':Sc.shape}
    print('copy to K',timer()-start)
    start=timer()    
    #a.d.
    np.savez(fname, **K)
    print('saving non',timer()-start,flush=True)
       
#def load_sparse_plan(tipe, num_rays, theta, center=None, kernel_type = 'gaussian', k_r = 1, dcfilter='none_filter'):
#def load(fname):         
#    K=np.load(fname)
#    return K

def load(tipe, num_rays, theta, center=None, kernel_type = 'gaussian', k_r = 1, dcfilter='none_filter'):
    fname=get_hash_name(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter)
    if os.path.isfile(fname):
        return np.load(fname)
    else:
        return None
    
    
        
    
# csr_matrix((data, indices, indptr), [shape=(M, N)])

#arrays_dict = {'val':Sc.data,'ind':Sc.indices, 'indptr':Sc.indptr}
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

