#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparse_plan
"""
import os
import numpy as np
import hashlib


try: 
    import cupy as cp
    def iscupy(x):
        if cp.get_array_module(x).__name__=='cupy':
            return True
        else:
            return False

    def tonp(x):
        if iscupy(x):
            return cp.asnumpy(x)
        else:
            return x
except:
    def tonp(x): return x

def xbytes(x): return bytes(tonp(x))


#cache='.cache/'
cache=os.path.expanduser('~/.cache/xpack/')


#from timeit import default_timer as timer


if not os.path.isdir(cache):
    os.mkdir(cache)

def get_hash_name(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter):
    if type(center) == type(None): center=num_rays//2 
    if type(dcfilter) == type(None):  dcfilter=np.array([0])
    

    fmt='_{}_{}_'.format(num_rays,theta.shape[0])
    fname=cache+tipe+fmt
    
    # checksum the rest
    #b=bytes(np.array([num_rays,theta.shape[0],center,k_r]))+bytes(theta)+bytes(dcfilter)+bytes(tipe+kernel_type,encoding='utf8')
    #b=bytes(np.array([center,k_r]))+bytes(theta)+bytes(dcfilter)+bytes(kernel_type,encoding='utf8')
    b=xbytes(theta)+bytes(np.array([center,k_r]))+bytes(kernel_type,encoding='utf8')
    if tipe=='S': b+=xbytes(dcfilter)

    #hs=hashlib.blake2b(b,digest_size=8,salt=bytes(0)).hexdigest()
    #md5Hash = hashlib.sha1()
    #md5Hash.update(b)
    #hs = md5Hash.hexdigest()
    #hs=hashlib.sha1(b).hexdigest()
    hs=hashlib.md5(b).hexdigest()
    
    #hs=hs[0::4]
    
    fname= fname+str(hs)+'.npz'
    return fname

def save(Sc,tipe, num_rays, theta, center, kernel_type, k_r, dcfilter):   

    
    fname=get_hash_name(tipe,num_rays, theta,  center, kernel_type, k_r, dcfilter)

    K = {'val':Sc.data,'ind':Sc.indices, 'indptr':Sc.indptr,'shape':Sc.shape}

    #K['theta']=theta
    #K['filter']=dcfilter
    
    np.savez(fname, **K)
       

# import scipy
def load(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter):

    fname=get_hash_name(tipe, num_rays, theta, center, kernel_type, k_r, dcfilter)
    #print('loading "',fname,'"')
    if os.path.isfile(fname):
        K=np.load(fname)
        
        keys = ('val', 'ind', 'indptr', 'shape')
        if bool(keys - K.keys()): 
            print('wrong sparse plan', K.keys(), K)
            return None
        #tr=True
        #csr_matrix((data, indices, indptr), [shape=(M, N)])
        #S=scipy.sparse.csr_matrix((K['val'],K['ind'], K['indptr']), shape=(K['shape']))

        return K
    else:
        return None
       
    