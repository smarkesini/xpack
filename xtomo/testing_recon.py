import numpy as np
from reconstruct import recon, recon_file

import argparse
#import json
#import textwrap


ap = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    epilog='_'*60+'\n Note option precedence (high to low):  individual options, opts dictionary, fopts  \n'+'-'*60)

import parse
args=parse.main()

#print("hello", args)

#
#ap.add_argument("-f", "--file_in",  type = str, help="h5 file in")
#ap.add_argument("-o", "--file_out",   type = str, help="file out, default none, 0 will autogenerate name")
#ap.add_argument("-sim", "--simulate",  type = bool, help="use simulated data, bool")
#ap.add_argument("-sim_shape", "--sim_shape",  type = int,nargs='+', help="simulated data shape, bool")
#ap.add_argument("-rot_center", "--rot_center",  type = int, help="rotation center, int ")
#ap.add_argument("-a", "--algo",  type = str, help="algorithm: 'iradon' (default), 'sirt', 'cgls', 'tv', 'tvrings ")
#ap.add_argument("-G", "--GPU",  type = bool, help="turn on GPU, bool")
#ap.add_argument("-S", "--shmem",  type = bool, help="turn on shared memory MPI, bool")
#ap.add_argument("-maxiter", "--maxiter", type = int, help="maxiter, default 10")
#ap.add_argument("-max_chunk", "--max_chunk_slice",  type = int, help="max chunks per mpi rank")
#ap.add_argument("-reg", "--reg",  type = float, help="regularization parameter")
#ap.add_argument("-tau", "--tau", type = float, help="soft thresholding parameter")
#ap.add_argument("-v", "--verbose",   type = float, help="verbose float (0...1), default 1")
#ap.add_argument('-opts', '--options', type=json.loads, help="e.g. \'{\"algo\":\"iradon\", \"maxiter\":10, \"tol\":1e-2, \"reg\":1, \"tau\":.05} \' ")
#ap.add_argument('-fopts', '--foptions', type=str, help="file with options ")
#
#Dopts={ 'maxiter':10 ,'algo':'iradon', 'shmem':True, 'GPU':True, 'max_chunk_slice':16, 'verbose':True }
#
#args = vars(ap.parse_args())
#opts = args['options']
#
#if args['foptions']!=None:
#    fopts = json.load(open(args['foptions'],'r'))
#    Dopts.update(fopts)
#
#if opts!=None:
#    Dopts.update(opts)
#
#for key in Dopts:
#    if args[key]==None: args[key]=Dopts[key]
#if args['file_in']==None: args['simulate']=True

sim_shape=args['sim_shape']#[256, 181, 256]
sim_width=args['sim_width']



#print("args", args)

GPU   = args['GPU']
algo  = args['algo']
shmem = args['shmem']
rot_center = args['rot_center']
simulate   = args['simulate']
max_chunk  = args['max_chunk_slice']
max_iter   = args['maxiter']

reg = args['reg']
tau = args['tau']
verboseall = args['verbose']


#print("testing_recon max_iter",max_iter)
#algo='tvrings'

from communicator import rank, mpi_size
if rank==0: print(args)

if type(args['file_in']) is not type(None):
    fname=args['file_in']
elif simulate:
    # from simulate_data import get_data as gdata  
    from simulate_data import init
    #print("running simulations",sim_shape)
    
    #obj_size = 256
    num_slices = sim_shape[0] #16*32# size//2
    #num_angles =    obj_size//2
    num_angles =  sim_shape[1] #180*6+1
    num_rays   = sim_shape[2]# obj_size
    obj_width=sim_width#0.95
    
    fid, dnames = init(num_slices,num_rays,num_angles,obj_width)
    #print("fid",fid.filename)
    theta = fid[dnames['theta']]
    sino  = fid[dnames['sino' ]]
    true_obj = fid[dnames['tomo' ]]
    #def get_data(x,chunks=None):
    args['file_in']=fid.filename
    fname=fid.filename
    args['file_out']='0'
    #print("about to be running simulations",sim_shape)
   
    tomo, times_loop = recon_file(fname,dnames=None, algo = algo ,rot_center = rot_center, max_iter = max_iter, GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk, reg = reg, tau = tau)


'''

#simulate=True
if simulate:
    # from simulate_data import get_data as gdata  
    from simulate_data import init
    
    #obj_size = 256
    num_slices = sim_shape[0] #16*32# size//2
    #num_angles =    obj_size//2
    num_angles =  sim_shape[1] #180*6+1
    num_rays   = sim_shape[2]# obj_size
    obj_width=sim_width#0.95
    
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



#print('sino type',type(sino))
#tomo, times_loop = reconstruct(sino, theta, algo = 'iradon' ,rot_center = rot_center, max_iter = max_iter)
tomo, times_loop = recon(sino, theta, algo = algo ,rot_center = rot_center, max_iter = max_iter, GPU=GPU,shmem=shmem, max_chunk_slice = max_chunk,  reg = reg, tau = tau)

'''

if type(args['file_out']) is not type(None):  
        import os, sys
        
        cstring = ' '.join(sys.argv)
        
        file_out = args['file_out']
        if file_out == '0': 
            file_out = os.path.splitext(fname)[0]
            file_out=file_out+'_'+algo+'_recon.tif'
            print('file out was 0, changed to:',file_out)
            
        print('file out',args['file_out'])
              
        if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
            from tifffile import imsave
            imsave(file_out,tomo, description = cstring)
            quit()                      
        import h5py
        fname=args['file_out']
        fid = h5py.File(fname, 'w')
        fid.create_dataset('exchange/tomo', data = tomo)
        fid.create_dataset('exchange/command', data =' '.join(sys.argv) )
        fid.close()
        #quit()
        #from tifffile import imsave
    
    #tomo, times_loop =
    

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

#import imageio
#imageio.imwrite("test.png", tomo0c)
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
    
    print("phantom shape",true_obj.shape, "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    #print("reading tomo, shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)
    
    
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
