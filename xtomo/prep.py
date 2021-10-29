import numpy as np
import h5py
import time
import numexpr as ne
import os
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
from .stripe_removal_original import remove_all_stripe as srm1

os.nice(60)

h5fname_in = '/tomodata/tomobank/tomo_00001/tomo_00001.h5'
chunks = 16
rot_center = None
h5fname_out = None

# import timer

#########################

credits={'vo_et_al': 'Vo, N. T., Atwood, R. C. & Drakopoulos, M.  Opt. Express, 26, 28396–28412, (2018).',
         'bm3d_streak':'Mäkinen, Y., Marchesini, S., Foi, A., "Ring artifact reduction via multiscale nonlocal collaborative filtering of spatially correlated noise", J. Synchrotron Rad. 28(3), pages 876-888, 2021.',
         'munch_et_al': 'Münch, B., Trtik, P., Marone, F. & Stampanoni, M. Opt. Express, 17, 8567–8591, (2009).'             
         }


# a complicated construct to initialize a function for parallel processing
_func0 = None    
def stripe_init(data,arg0=3,arg1=81,arg2=31):
  global _func0
  stripe_flt=lambda i: srm1(data[i], arg0, arg1, arg2)
  _func0 = stripe_flt

def stripe_initnt(data,arg0=3,arg1=81,arg2=31):
  global _func0
  #stripe_flt=lambda i: srm1.remove_all_stripe(data[:,i,:], arg0, arg1, arg2)
  def stripe_flt(i): 
      # in place
      data[:,i,:]=srm1(data[:,i,:], arg0, arg1, arg2)
      #return data[:,i,:]

  _func0 = stripe_flt


def stripe_flt(i):
    return _func0(i)

def stripe_clear():
    global _func0
    _func0 = None
    

def stripe_loop(data, nthreads=cpu_count()-4, arg0=3.,arg1=81,arg2=31, csz=2):
    data=np.ascontiguousarray(np.swapaxes(data,0,1))
    data1=np.empty_like(data)
     
    stripe_init(data, arg0,arg1,arg2)
     
    pool = Pool(processes=nthreads)        
    
    t0=timer()
    #data1 = np.asarray(pool.map(flt,(range(data.shape[0]))))
    
    #csz=1 # do 1 chunks per parallel job 
    for ii in range(0,data.shape[0],nthreads*csz):
        ii_end=np.min([ii+nthreads*csz,data.shape[0]-1])
        data1[ii:ii_end] = np.asarray(pool.map(stripe_flt,(range(ii,ii_end))))    
        #time.sleep(1) # let it cool otherwise it crashes
        print(ii,'-',ii_end,'/',data.shape[0],'time',timer()-t0, 'left=',(timer()-t0)/(ii_end)*(data.shape[0]-ii_end), flush=True)
    t1=timer()
    
    
    print('stripe removal ',t1-t0,'sec')
    t0=timer()
    # swap back
    data=data1
    data=np.ascontiguousarray(np.swapaxes(data1,0,1))
    t1=timer()
    print('swap axes ',t1-t0,'sec')
    return data
    


def stripe_loopnt(data, nthreads=cpu_count()-4, arg0=3.,arg1=81,arg2=31, csz=4):
    #data=np.ascontiguousarray(np.swapaxes(data,0,1))
    #data1=np.empty_like(data)
     
    stripe_initnt(data, arg0,arg1,arg2)
     
    pool = Pool(processes=nthreads)        
    
    t0=timer()
    t1=t0
    ii1=0
    #data1 = np.asarray(pool.map(flt,(range(data.shape[0]))))
    
    #csz=1 # do 1 chunks per parallel job 
    ii_end=np.min([nthreads*csz,data.shape[1]-1])
    for ii in range(0,data.shape[1],nthreads*csz):
        if ii==1:
            t1=timer() # the first iteration may be slower, so we only use this from now on
            ii1=ii_end
        ii_end=np.min([ii+nthreads*csz,data.shape[1]-1])
        #data[:,ii:ii_end,:] = np.swapaxes(np.asarray(pool.map(stripe_flt,(range(ii,ii_end)))),0,1)    
        pool.map(stripe_flt,(range(ii,ii_end)))    
        #time.sleep(1) # let it cool otherwise it crashes
        #print(ii,'-',ii_end,'/',data.shape[1],'time',timer()-t0, 'left=',(timer()-t0)/(ii_end)*(data.shape[0]-ii_end), flush=True)
        #print('({:d}-{:d})/{:d}'.format(ii,ii_end,data.shape[1]),
        #      'time:{:0.1f}, left=:{:0.1f}'.format(timer()-t0,(timer()-t1)/(ii_end-ii1)*(data.shape[0]-ii_end)), flush=True)

        # printbar(ii_end/data.shape[1]*100,string='stime:{:0.1f}, left=:{:0.1f}'.format(timer()-t0,(timer()-t0)/(ii_end)*(data.shape[1]-ii_end)))

    t1=timer()
    
    
    # print('stripe removal ',t1-t0,'sec')
    t0=timer()
    # swap back
    #data=data1
    #data=np.ascontiguousarray(np.swapaxes(data1,0,1))
    #t1=timer()
    #print('swap axes ',t1-t0,'sec')
    stripe_clear()
    return data
    
#########################

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
    #shape=( shape[1],shape[0],shape[2])

    return shape

import sys
def printbar(percent,string='    ', string2=''):        
        # print(percent)        
        sys.stdout.write('\r%s Progress: [%-50s] %3i%% %s' %(string,'=' * int(percent // 2), percent,string2))
        sys.stdout.flush()
       

from bm3d_streak_removal import *


def clean_sino(h5fname, chunks=None, algo='vo_et_al'):
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

    if algo == 'bm3d_streak':
        print('bm3d_streak')
        data = full_streak_pipeline(proj, flat, dark)
        data= ne.evaluate('-(a)',local_dict={'a':data})
      
    else:
        # Flat-field correction of raw data.
        data = tomopy.normalize(proj, flat, dark, cutoff=1.4)
        #data = tomopy.normalize(proj, flat, dark, cutoff=(1,2))
    
        # remove stripes
        #print('removing stripes, may take a while',flush=True)
        if algo=='munch_et_al':
            data = tomopy.remove_stripe_fw(data,level=7,wname='sym16',sigma=1,pad=True)
        elif algo=='vo_et_al':
            # data = tomopy.remove_all_stripe(data, snr = 3., la_size = 81, sm_size = 31 ,ncore = cpu_count()-2)
            data = stripe_loopnt(data)
            #nthreads=cpu_count()-4, arg0=3.,arg1=81,arg2=31, csz=4)
        
        #print("Raw data: ", h5fname)
        #print("Center: ", rot_center)
    
        data = tomopy.minus_log(data)
    
        data = tomopy.remove_nan(data, val=0.0)
        data = tomopy.remove_neg(data, val=0.00)
        data[np.where(data == np.inf)] = 0.00

    
    return  data, theta

def make_chunks(num_slices, max_chunks, chunks = None):

    if type(chunks)==type(None):
        chunk_offset = 0
        nslices = num_slices
    else:
        chunk_offset = (num_slices-chunks)//2
        nslices = chunks

    
    nchunks=int(np.ceil(nslices/max_chunks))
    chunks_init=np.arange(0,nslices,max_chunks)
    chunks_end  = np.append(chunks_init[1:],values=[nslices],axis=0).reshape(int(nchunks),1)
    chunks_init.shape=(int(nchunks),1)
    chunks_end.shape=(int(nchunks),1)
    chunks = np.concatenate((chunks_init,chunks_end),axis=1)
    chunks += chunk_offset
    
    #if type(chunks)!= type(None):
    #    continue
    
    return chunks, nchunks
        

def write_h5(fid,value,dirname="data"):
        fid.create_dataset(dirname, data = value)

def clean_raw(h5fname_in=h5fname_in, h5fname_out=None, max_chunks=None, stripe_algo='vo_et_al',chunks=None):

    import os
    print('stripe removal algo:', stripe_algo)
    try:
        print('\n credits: ', credits[stripe_algo], '\n')
    except:
        pass
    
    #h5fname =  h5fname_out
    if type(h5fname_out) == type(None):
        file_out = os.path.splitext(os.path.splitext(os.path.splitext(h5fname_in)[0])[0])[0]
        h5fname_out=file_out+'_clean.h5'
    elif h5fname_out=='0':
        file_out = os.path.splitext(os.path.splitext(os.path.splitext(h5fname_in)[0])[0])[0]
        h5fname_out=file_out+'_clean.h5'


    print("will be writing to:",h5fname_out)
    #h5fname ='/tomodata/tomobank_clean/tomo_00072_preprocessed1.h5'

    fid_in = h5py.File(h5fname_in, 'r')
    
    fid = h5py.File(h5fname_out, 'w')
    
    #dnames={'sino':"exchange/data", 'theta':"exchange/theta"}
    
    
    dnames={'sino':"exchange/data", 'theta':"exchange/theta", 'rot_center':'exchange/rot_cenrter'}
    start_loop_time =time.time()   
    #chunks = [0, num_slices]
    if max_chunks==None:
        #nchunks=int(1)
        #max_chunks=num_slices
        print('cleaning the whole sinogram')
        data, theta = clean_sino(h5fname=h5fname_in, algo=stripe_algo,chunks=chunks)
        data=np.swapaxes(data,0,1)
        data = np.ascontiguousarray(data)
        vname='sino'
        print("writing {} to: {}".format(vname,dnames[vname]))
        write_h5(data,dirname=dnames['sino'])

    else:
        #dims = get_dims(h5fname=h5fname_in)
        dims = fid_in['/exchange/data'].shape
        #data_shape = get_data('dims')
        num_slices = dims[1]
        num_angles = dims[0]
        num_rays   = dims[2]        
        chunks, nchunks = make_chunks(num_slices, max_chunks,chunks)
        num_slices = chunks[-1,1]-chunks[0,0]
        #nchunks=int(np.ceil(num_slices/max_chunks))
        print("number of slices=", chunks[-1,1]-chunks[0,0],', slices per chunk=',max_chunks,", number of chunks=",nchunks)
        fsino=fid.create_dataset(dnames['sino'], (num_slices,num_angles,num_rays) , chunks=(num_slices,num_angles,max_chunks),dtype='float32')
        #chunks_init=np.arange(0,num_slices,max_chunks)
        #chunks_end  = np.append(chunks_init[1:],values=[num_slices],axis=0).reshape(int(nchunks),1)
        #chunks_init.shape=(int(nchunks),1)
        #chunks_end.shape=(int(nchunks),1)
        #chunks = np.concatenate((chunks_init,chunks_end),axis=1)

        for ii in range(nchunks):
            #start_time =time.time()
            #printbar(ii*100//nchunks)
            #print("reading slices:",chunks[ii],'{}/{}'.format(ii,nchunks), flush=True)  
            data, theta = clean_sino(h5fname=h5fname_in,chunks=chunks[ii], algo=stripe_algo)
            
            data=np.swapaxes(data,0,1)
            # data = np.ascontiguousarray(data)
            fsino[chunks[ii,0]-chunks[0,0]:chunks[ii,1]-chunks[0,0],...]=data
            string='{}/{}'.format(ii+1,nchunks) 
            
            t_passed=time.time()-start_loop_time 
            t_left= (t_passed)/(ii+1)*(nchunks-ii-1)

            string2='t-passed={}, t-left={}'.format(t_passed,t_left)
            
            printbar((ii+1)*100//nchunks,string,string2)
            #print("done writing", time.time()-start_write_time, "total",time.time()-start_loop_time)
            #write_h5(data,dirname="sino",chunks=chunks[ii])
        #print('data0',data[0],'data_sum', np.sum(data))
        print("\n processed and saved in", time.time()-start_loop_time)

    

    vname='theta'
    print("writing {} to: {}".format(vname,dnames[vname]))
    
    #write_h5(theta,dirname=dnames['theta'])
    fid.create_dataset(dnames['theta'], data = theta)
    
    if rot_center != None:
        vname='rot_center'
        print("writing {} to: {}".format(vname,dnames[vname]))
        write_h5(rot_center,dirname='exchange/rot_center')
    
    fid.close()
    return data,theta, rot_center, h5fname_out
    
if __name__ == "__main__":
    import argparse
    #import json
    #import textwrap

    ap = argparse.ArgumentParser( formatter_class=argparse.RawTextHelpFormatter)
    #    epilog='_'*60+'\n Note option precedence (high to low):  individual options, opts dictionary, fopts  \n'+'-'*60)
    ap.add_argument("-f", "--file_in",  type = str,default ='/tomodata/tomobank/tomo_00001/tomo_00001.h5', help="h5 file in")
    ap.add_argument("-f", "--file_in",  type = str,default =None, help="h5 file in")
    ap.add_argument("-o", "--file_out",   type = str, default='0', help="file out, default 0, 0: autogenerate name, -1: skip saving")
    ap.add_argument("-r", "--rot_center",  type = int, help="rotation center, int ")
    ap.add_argument("-mc", "--max_chunks",  type = int, help="max chunk size, int ")
    ap.add_argument("-c", "--chunks",  type = int, help="chunk cropping, int ")
    args = vars(ap.parse_args())



    #import parse

    h5fname_in = args['file_in'] #'/tomodata/tomobank/tomo_00001/tomo_00001.h5'
    max_chunks = args['max_chunks']
    chunks     = args['chunks']
    rot_center = args['rot_center'] #1024
    h5fname_out = args['file_out']

    clean_raw(h5fname_in=h5fname_in, h5fname_out= h5fname_out, max_chunks=max_chunks, chunks=None, algo='vo_et_al')

