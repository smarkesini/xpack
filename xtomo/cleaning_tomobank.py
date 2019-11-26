import numpy as np
# import matplotlib.pyplot as plt
#import h5py
    
#import time
#h5fname_in ='/data/tomobank/tomo_00001/tomo_00001.h5'

import h5py
import time


import argparse
#import json
#import textwrap

ap = argparse.ArgumentParser( formatter_class=argparse.RawTextHelpFormatter)
#    epilog='_'*60+'\n Note option precedence (high to low):  individual options, opts dictionary, fopts  \n'+'-'*60)
ap.add_argument("-f", "--file_in",  type = str,default ='/tomodata/tomobank/tomo_00001/tomo_00001.h5', help="h5 file in")
ap.add_argument("-o", "--file_out",   type = str, default='0', help="file out, default 0, 0: autogenerate name, -1: skip saving")
ap.add_argument("-r", "--rot_center",  type = int, help="rotation center, int ")
ap.add_argument("-c", "--chunks",  type = int, help="chunk size, int ")
args = vars(ap.parse_args())



#import parse

h5fname_in = args['file_in'] #'/tomodata/tomobank/tomo_00001/tomo_00001.h5'
chunks = args['chunks']
rot_center = args['rot_center'] #1024
h5fname_out = args['file_out']
#5fname ='/tomodata/tomobank_clean/tomo_00072_preprocessed1.h5'

#def clean_raw(h5fname_in=h5fname_in, h5fname_out=None):
    
              
#rint("writing to ",h5fname)
# = h5py.File(h5fname, 'w')
#
"""
def write_h5(value,dirname="data", chunks=None):
    print("writing to :", h5fname, dirname, chunks)
    f.create_dataset(dirname, data = value, chunks=chunks)
        

from read_tomobank_dirty import get_data
# (1792, 1501, 2048)
data_shape = get_data('dims')
num_slices = data_shape[0]
num_angles = data_shape[1]
num_rays   = data_shape[2]
rot_center = 1024

write_h5([num_slices,num_angles,num_rays],dirname="dims")


dname_sino="exchange/sino"
dname_theta="exchange/theta"

theta = get_data('theta')

write_h5(theta,dirname=dname_theta)

max_chunks= 16*4



#chunks = [0, num_slices]
nchunks=int(np.ceil(num_slices/max_chunks))

chunks_init=np.arange(0,num_slices,max_chunks)
chunks_end  = np.append(chunks_init[1:],values=[num_slices],axis=0).reshape(int(nchunks),1)
chunks_init.shape=(int(nchunks),1)
chunks_end.shape=(int(nchunks),1)
chunks = np.concatenate((chunks_init,chunks_end),axis=1)

fsino=f.create_dataset('sino', (num_slices,num_angles,num_rays) , chunks=(num_slices,num_angles,max_chunks),dtype='float32')

start_loop_time =time.time()
for ii in range(nchunks):
    start_time =time.time()
    print("reading slices:",chunks[ii],'{}/{}'.format(ii,nchunks), flush=True)  
    sino = get_data('sino',chunks=chunks[ii])
    data = np.ascontiguousarray(sino)
    print("done reading, time=",time.time()-start_time ,"writing clean slices",flush=True)
    start_write_time=time.time()
    fsino[chunks[ii,0]:chunks[ii,1],...]=data
    print("done writing", time.time()-start_write_time, "total",time.time()-start_loop_time)
    #write_h5(data,dirname="sino",chunks=chunks[ii])

f.close()
#quit()
"""

def get_dims(h5fname=h5fname_in):
    """
    Read array size of a specific group of Data Exchange file.

    Parameters
    ----------
    fname : str
        String defining the path of file or file name.
    dataset : str
        Path to the dataset inside hdf5 file where data is located.

    Returns
    -------
    ndarray
        Data set size.
    """
    'data'
    grp = '/'.join(['exchange', 'data'])

    with h5py.File(h5fname, "r") as f:
        try:
            data = f[grp]
        except KeyError:
            return None

        shape = data.shape
    #swapping dimensions
    shape=( shape[1],shape[0],shape[2])

    return shape

import sys
def printbar(percent,string='    '):                
        sys.stdout.write('\r%s Progress: [%-60s] %3i%% ' %(string,'=' * (percent // 2), percent))
        sys.stdout.flush()
       
#import time
#from read_sigray_dirty import clean_sino

#h5fname ='/tomodata/sigray/Fly_data/Fly_preprocessed.h5'
#5fname_in ='/tomodata/sigray/Fly_data/Abs_Fly_tomo180_1p125um_4steps_34kV_4x4_2x_50s_042_whitedark.h5.h5'



def clean_sino(h5fname, chunks=None):
    import tomopy
    import dxchange

    
    # Read APS 32-BM raw data.
    #print('reading data')
    proj, flat, dark, theta = dxchange.read_aps_32id(h5fname, sino=chunks)
    #proj, flat, dark, theta = dxchange.read_aps_32id(h5fname)
       
#    # Manage the missing angles:
#    if blocked_views is not None:
#        print("Blocked Views: ", blocked_views)
#        proj = np.concatenate((proj[0:blocked_views[0],:,:], proj[blocked_views[1]+1:-1,:,:]), axis=0)
#        theta = np.concatenate((theta[0:blocked_views[0]], theta[blocked_views[1]+1:-1]))

    # Flat-field correction of raw data.
    data = tomopy.normalize(proj, flat, dark, cutoff=1.4)
    #data = tomopy.normalize(proj, flat, dark, cutoff=(1,2))

    # remove stripes
    #print('removing stripes, may take a while',flush=True)
    data = tomopy.remove_stripe_fw(data,level=7,wname='sym16',sigma=1,pad=True)

    #print("Raw data: ", h5fname)
    #print("Center: ", rot_center)

    data = tomopy.minus_log(data)

    data = tomopy.remove_nan(data, val=0.0)
    data = tomopy.remove_neg(data, val=0.00)
    data[np.where(data == np.inf)] = 0.00

    data=np.swapaxes(data,0,1)
    return  data, theta


def clean_raw(h5fname_in=h5fname_in, h5fname_out=None, max_chunks=None):
    import os
    
    #h5fname =  h5fname_out
    if type(h5fname_out) == type(None):
        file_out = os.path.splitext(os.path.splitext(os.path.splitext(h5fname_in)[0])[0])[0]
        h5fname_out=file_out+'_clean.h5'
    elif h5fname_out=='0':
        file_out = os.path.splitext(os.path.splitext(os.path.splitext(h5fname_in)[0])[0])[0]
        h5fname_out=file_out+'_clean.h5'
        
    #h5fname ='/tomodata/tomobank_clean/tomo_00072_preprocessed1.h5'
    dims = get_dims(h5fname=h5fname_in)
    #data_shape = get_data('dims')
    num_slices = dims[0]
    num_angles = dims[1]
    num_rays   = dims[2]
    #num_slices=num_slices//20
    
    print("will be writing to:",h5fname_out,'dimensions',dims)
    
    fid = h5py.File(h5fname_out, 'w')
    
    #dnames={'sino':"exchange/data", 'theta':"exchange/theta"}
    
    def write_h5(value,dirname="data"):
        fid.create_dataset(dirname, data = value)
    
    dnames={'sino':"exchange/data", 'theta':"exchange/theta", 'rot_center':'exchange/rot_center'}

   
    chunks = [0, num_slices]
    if max_chunks!=None:
        nchunks=int(np.ceil(num_slices/max_chunks))
    else: 
        nchunks=int(1)
        max_chunks=num_slices  
    
    start_loop_time =time.time()
    print("number of chunks=",nchunks)
    if nchunks>1:
        fsino=fid.create_dataset(dnames['sino'], (num_slices,num_angles,num_rays) , chunks=(num_slices,num_angles,max_chunks),dtype='float32')
        chunks_init=np.arange(0,num_slices,max_chunks)
        chunks_end  = np.append(chunks_init[1:],values=[num_slices],axis=0).reshape(int(nchunks),1)
        chunks_init.shape=(int(nchunks),1)
        chunks_end.shape=(int(nchunks),1)
        chunks = np.concatenate((chunks_init,chunks_end),axis=1)

        for ii in range(nchunks):
            #start_time =time.time()
            #printbar(ii*100//nchunks)
            #print("reading slices:",chunks[ii],'{}/{}'.format(ii,nchunks), flush=True)  
            sino, theta = clean_sino(h5fname=h5fname_in,chunks=chunks[ii])
            data = np.ascontiguousarray(sino)
            #print("done reading, time=",time.time()-start_time ,"writing clean slices",flush=True)
            #start_write_time=time.time()
            fsino[chunks[ii,0]:chunks[ii,1],...]=data
            string='{}/{}'.format(ii+1,nchunks)
            printbar((ii+1)*100//nchunks,string)
            #print("done writing", time.time()-start_write_time, "total",time.time()-start_loop_time)
            #write_h5(data,dirname="sino",chunks=chunks[ii])
        print("\n processed and saved in", time.time()-start_loop_time)
    else:        
        data, theta = clean_sino(h5fname=h5fname_in)
        data = np.ascontiguousarray(data)
        vname='sino'
        print("writing {} to: {}".format(vname,dnames[vname]))
        write_h5(data,dirname=dnames['sino'])
    

    vname='theta'
    print("writing {} to: {}".format(vname,dnames[vname]))
    
    write_h5(theta,dirname=dnames['theta'])

    
    if rot_center != None:
        vname='rot_center'
        print("writing {} to: {}".format(vname,dnames[vname]))
        write_h5(rot_center,dirname='exchange/rot_center')
    
    fid.close()
    return data,theta, rot_center, h5fname_out
    
clean_raw(h5fname_in=h5fname_in, h5fname_out= h5fname_out, max_chunks=chunks)

