import numpy as np
#import matplotlib.pyplot as plt
#import h5py

#print("hello",flush=True)
#import tomopy
#import imageio
#import os
GPU=True
GPU=False

shmem=True
#shmem = False

algo='iradon'
#algo='sirt'
#algo = 'tv'
#algo='tomopy-gridrec'




#num_angles =    11




rot_center = None


simulate=True
if simulate:
    # from simulate_data import get_data as gdata  
    from simulate_data import init 
    obj_size = 1024//4
    num_slices = 16*32# size//2
    #num_angles =    obj_size//2
    num_angles =  180*6+1
    num_rays   = obj_size
    obj_width=0.95
    
    fid, dnames = init(num_slices,num_rays,num_angles,obj_width)
    
    theta = fid[dnames['theta']]
    sino  = fid[dnames['sino' ]]
    true_obj = fid[dnames['tomo' ]]
    #def get_data(x,chunks=None):    


    #    return gdata(num_slices,num_rays,num_angles,obj_width,x,chunks=chunks) 
else:
    from read_tomobank import get_data
    theta = get_data('theta')
    sino = get_data('sino')
    true_obj = None

    #rot_center = 1403
    #rot_center = 1024
    #from read_sigray import get_data


#theta = get_data('theta')
#sino = get_data('sino')

max_iter = 10

from reconstruct import reconstruct

#print('sino type',type(sino))
tomo, times_loop = reconstruct(sino, theta, algo = 'iradon' ,rot_center = rot_center, max_iter = max_iter)


num_slices = tomo.shape[0]
num_rays = tomo.shape[1]
num_angles = sino.shape[1]

import fubini
msk_tomo,msk_sino=fubini.masktomo(num_rays, np, width=.95)
#msk_tomo=msk_tomo[0,...]


t2i = lambda x: x[num_slices//2,:,:].real
tomo0c=t2i(tomo)*msk_tomo[0,...]
#plt.imshow(np.abs(tomo0c))
#plt.show()
import imageio
imageio.imwrite("test.png", tomo0c)
#
#plt.imshow((data[0]))
#img = None
#for f in range(num_slices):
#    im=tomo[f,:,:]*msk_tomo
#    if img is None:
#        img = plt.imshow(im)
#    else:
#        img.set_data(im)
#    plt.pause(.01)
#    plt.draw()
#

    #quit()

try:
    #true_obj = get_data('tomo')[...]
    true_obj = true_obj[...]
    print("comparing with truth, summary coming...\n\n")
except:
    true_obj = None



if type(true_obj) == type(None): 
    print("no tomogram to compare")
    #quit()

else:
    
    print("phantom shape",true_obj.shape, "n_angles",num_angles, 'algorithm:', algo,"GPU:",GPU,"max_iter:",max_iter)
    print("reading tomo, shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)
    
    
    scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
    rescale = lambda x,y: scale(x,y)*x
    ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
    ssnr2    = lambda x,y: ssnr(x,y)**2
    
    
    print("times full tomo", times_loop)
    print("solver time=", times_loop['solver'], "snr=", ssnr(true_obj,tomo))
    
    #tomo0=tomo_chunk
    
    
    #print("psirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo1))
    
    ## tomo to cropped image
    #t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
    ## vector to tomo
    #v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))
    
    #print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))
    
    #if GPU:
    #    cupy = xp
    #    mempool = cupy.get_default_memory_pool()
    #    pinned_mempool = cupy.get_default_pinned_memory_pool()
    #    print(mempool.used_bytes())
    #    del data,iradon,theta
    #    try: del radon
    #    except: None
    #        
    #    mempool.free_all_blocks()
    #    pinned_mempool.free_all_blocks()
    #    print(mempool.used_bytes())
    #
