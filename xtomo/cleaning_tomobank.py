import numpy as np
import matplotlib.pyplot as plt
#import h5py
    
#import time
#h5fname_in ='/data/tomobank/tomo_00001/tomo_00001.h5'

import h5py
import time

h5fname ='/tomodata/tomobank_clean/tomo_00072_preprocessed1.h5'
print("writing to ",h5fname)

f = h5py.File(h5fname, 'w')
#

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


dname_sino="sino"
dname_theta="theta"

theta = get_data('theta')

write_h5(theta,dirname="theta")

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
