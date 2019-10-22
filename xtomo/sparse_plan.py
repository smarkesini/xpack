#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparse_plan
"""
import os
import numpy as np
import hashlib


cache='.cache/'

#from timeit import default_timer as timer


if not os.path.isdir(cache):
    os.mkdir(cache)

def get_hash_name(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter):
    if type(center) == type(None): center=num_rays//2 
    if type(dcfilter) == type(None):  dcfilter=0


    fmt='_{}_{}_'.format(num_rays,theta.shape[0])
    fname=cache+tipe+fmt
    
    # checksum the rest
    #b=bytes(np.array([num_rays,theta.shape[0],center,k_r]))+bytes(theta)+bytes(dcfilter)+bytes(tipe+kernel_type,encoding='utf8')
    #b=bytes(np.array([center,k_r]))+bytes(theta)+bytes(dcfilter)+bytes(kernel_type,encoding='utf8')
    b=bytes(np.array([center,k_r,theta]))+bytes(kernel_type,encoding='utf8')
    if tipe=='S': b+=bytes(np.array([dcfilter]))

    hs=hashlib.blake2b(b,digest_size=8).hexdigest()
    
    fname= fname+'_'+str(hs)+'.npz'
    return fname

def save(Sc,tipe, num_rays, theta, center, kernel_type, k_r, dcfilter):   

    
    fname=get_hash_name(tipe,num_rays, theta,  center, kernel_type, k_r, dcfilter)

    K = {'val':Sc.data,'ind':Sc.indices, 'indptr':Sc.indptr,'shape':Sc.shape}

    #K['theta']=theta
    #K['filter']=dcfilter
    
    np.savez(fname, **K)
       

import scipy
def load(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter):

    fname=get_hash_name(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter)
    #print('loading "',fname,'"')
    if os.path.isfile(fname):
        K=np.load(fname)
        #csr_matrix((data, indices, indptr), [shape=(M, N)])
        S=scipy.sparse.csr_matrix((K['val'],K['ind'], K['indptr']), 
                                      shape=(K['shape']))

        return S
    else:
        return None
       
    