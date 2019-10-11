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

def get_hash_name(tipe, num_rays, theta, center=None, kernel_type="gaussian", k_r=1, dcfilter=None):
    if type(center) == type(None): center=num_rays//2 
    if type(dcfilter) == type(None):  dcfilter=0
    
    b=bytes(np.array([num_rays,theta.shape[0],center,k_r]))+bytes(theta)+bytes(dcfilter)+bytes(tipe+kernel_type,encoding='utf8')
    hs=hashlib.sha1(b).hexdigest()

    fname=cache+tipe+'_'+str(hs)+'.npz'
    return fname

def save(Sc,tipe, num_rays, theta, center=None, kernel_type = 'gaussian', k_r = 1, dcfilter='none_filter'):   

    #start=timer()
    
    fname=get_hash_name(tipe,num_rays, theta,  center, kernel_type, k_r, dcfilter)
    print('saving fname',fname)
    #print('hash time',timer()-start)
    #start=timer()
    K = {'val':Sc.data,'ind':Sc.indices, 'indptr':Sc.indptr,'shape':Sc.shape}
    #print('copy to K',timer()-start)
    K['theta']=theta
    K['filter']=dcfilter
    
    #start=timer()    
    #a.d.
    np.savez(fname, **K)
    #print('saving time',timer()-start,flush=True)
       

import scipy
def load(tipe, num_rays, theta, center=None, kernel_type = 'gaussian', k_r = 1, dcfilter='none_filter'):

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
       
    