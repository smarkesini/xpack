import numpy as np
from reconstruct_mp import  recon_file #, recon

import argparse

from timeit import default_timer as timer
time0=timer()
#import json
#import textwrap


ap = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    epilog='_'*60+'\n Note option precedence (high to low):  individual options, opts dictionary, fopts  \n'+'-'*60)

import parse
args=parse.main()
print(args)
sim_shape=args['sim_shape']
sim_width=args['sim_width']




GPU   = args['GPU']
algo  = args['algo']
shmem = args['shmem']
rot_center = args['rot_center']
simulate   = args['simulate']
max_chunk  = args['max_chunk_slice']
max_iter   = args['maxiter']
tol   = args['tol']
#print('tolerance',tol)
reg = args['reg']
tau = args['tau']
verboseall = args['verbose']
chunks = args['chunks']

ncore = args['ncore']

#print("testing_recon max_iter",max_iter)
#algo='tvrings'

from communicator import rank, mpi_size
if rank==0: print(args)

if type(args['file_in']) is not type(None):
    fname=args['file_in']
    import h5py
    fid= h5py.File(fname, "r")
    sino  = fid['exchange/data']

    num_angles =  sino.shape[1]
    num_rays   =  sino.shape[2]
    num_slices =  sino.shape[0]

    fid.close()    
    #if type(dnames) == type(None):
    #    dnames=dnames_get()
    #sino  = fid[dnames['sino']]
    
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
    
    if type( args['file_out'])== type(None):    args['file_out']='0'
      

tomo, times_loop, dshape = recon_file(fname,dnames=None, algo = algo ,rot_center = rot_center, max_iter = max_iter, tol=tol, GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk, reg = reg, tau = tau, verbose=verboseall,ncore=ncore, chunks=chunks)


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
times_begin=timer()

if (type(args['file_out']) is not type(None)) and args['file_out']!='-1':  
        import os, sys
        
        cstring = ' '.join(sys.argv)
        tstring = str(times_loop)
        #print('cstring',cstring,'sysargv:',str(sys.argv) )
        #print('tstring,',tstring)        
        file_out = args['file_out']
        if file_out == '0': 
            file_out = os.path.splitext(fname)[0]
            file_out=file_out+'_'+algo+'_recon.tif'
            print('file out was 0, changed to:',file_out)
            args['file_out']=file_out
        else: print('file out',file_out)
              
        if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
            from tifffile import imsave
            imsave(file_out,tomo, description = cstring+' '+tstring)
        else:
            import h5py
            fname=args['file_out']
            fid = h5py.File(fname, 'w')
            fid.create_dataset('exchange/tomo', data = tomo)
            fid.create_dataset('mish/command', data =cstring )
            fid.create_dataset('mish/times', data =tstring )
            
            fid.close()
        #quit()
        #from tifffile import imsave
    
    #tomo, times_loop =

time_saving=timer()-times_begin
    

num_slices = tomo.shape[0]
num_rays = tomo.shape[1]
num_angles = dshape[1]

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
#print('checking for truth')
time_tot=timer()-time0
bold='\033[1m'
endb= '\033[0m'

print(bold+"tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
print("times full tomo", times_loop)
#print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
print(bold+"loop+setup time=", times_loop['loop']+times_loop['setup'], 'saving',time_saving, 'total', time_tot,endb, end='')




try:
    #true_obj = get_data('tomo')[...]
#    crop=chunks
#    if crop!=None:
#        if len(crop)==1:
#            loop_offset=true_obj.shape[0]//2-crop//2
#        else:
#            loop_offset=crop[0]
#    true_obj = true_obj[loop_offset:loop_offset+num_slices,...]
    true_obj = true_obj[...]

    #print("comparing with truth, summary coming...")
except:
    print("no tomogram to compare")

    quit()
    #print("no truth, quitting \n")
    #true_obj = None


if type(true_obj) == type(None): 
    print("no tomogram to compare")
    #quit()

else:
    


    import fubini
    msk_tomo,msk_sino=fubini.masktomo(num_rays, np, width=.95)
#msk_tomo=msk_tomo[0,...]


#t2i = lambda x: x[num_slices//2,:,:].real
    t2i = lambda x: x[num_slices//2,:,:].real
    tomo0c=t2i(tomo)*msk_tomo[0,...]
    #plt.imshow(np.abs(tomo0c))
    #print("phantom shape",true_obj.shape, "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    #print("reading tomo, shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)

    
    scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
    rescale = lambda x,y: scale(x,y)*x
    ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
    ssnr2    = lambda x,y: ssnr(x,y)**2
    #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)

    #bold='\033[1m'
    #endb= '\033[0m'
    #print(bold+"tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    #print("times full tomo", times_loop)
    #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
    print(bold+"snr=", ssnr(true_obj,tomo),endb)



#bold='\033[1m'
#    endb= '\033[0m'
#    print(bold+"tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
#    print("times full tomo", times_loop)
#    print("loop+setup time=", times_loop['loop']+times_loop['setup'],endb)
    

    
    #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
    
    
    
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
