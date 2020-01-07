import numpy as np
#from reconstruct import  recon_file #, recon
#from xtomo.loop_sino import  recon_file #, recon

from xtomo.communicator import rank, mpi_size

#import argparse

from timeit import default_timer as timer
from xtomo.IO import tomofile, tomosave, print_times
from xtomo.loop_sino import chunktomo
from xtomo.mish import compare_tomo

bold='\033[1m'
endb= '\033[0m'

#def recon():
if __name__ == "__main__":
    time0=timer()
    
    from xtomo.parse import parse
    args=parse()
    #print(args)
    sim_shape=args['sim_shape']
    sim_width=args['sim_width']
    time_file = args['time_file']
    GPU   = args['GPU']
    algo  = args['algo']
    shmem = args['shmem']
    rot_center = args['rot_center']
    simulate   = args['simulate']
    max_chunk  = args['max_chunk_slice']
    max_iter   = args['maxiter']
    tol   = args['tol']
    reg = args['reg']
    tau = args['tau']
    verboseall = args['verbose']
    chunks = args['chunks']
    ncore = args['ncore']
    ringbuffer = args['ring_buffer']
    file_out = args['file_out']
    

    if rank==0: print(args)
    
    if type(args['file_in']) is not type(None):
        fname=args['file_in']
        import h5py
        fid= h5py.File(fname, "r")
        sino  = fid['exchange/data']
        theta = fid['exchange/theta']
        if rot_center==None:
            try:
                rot_center = np.round( fid['exchange/rot_center'][()] )
                if rank == 0: print('rotation center from file',rot_center)
            except:
                pass
    
        #fid.close()    
    
    elif simulate:
        # from simulate_data import get_data as gdata  
        from xtomo.simulate_data import init            
        fid, dnames = init(sim_shape[0],sim_shape[2],sim_shape[1],sim_width)
                    
        #print("fid",fid.filename)
        theta = fid[dnames['theta']]
        sino  = fid[dnames['sino' ]]
        true_obj = fid[dnames['tomo' ]]
        #def get_data(x,chunks=None):
        args['file_in']=fid.filename
        fname=fid.filename
        
        if type( args['file_out'])== type(None):    args['file_out']='0'
    

    num_slices =  sino.shape[0]
    num_angles =  sino.shape[1]
    num_rays   =  sino.shape[2]


    num_slices_cropped, chunks = chunktomo(num_slices, chunks)
    
    times_of=timer()
    tomo_out, ring_buffer = tomofile(file_out, file_in=fname, algo=algo, shape_tomo=(num_slices_cropped,num_rays,num_rays), ring_buffer=ringbuffer)
    ringbuffer=ring_buffer
    
    times_of=timer()-times_of

    if rank == 0: print('output file setup time',times_of)
    
    if rank == 0: print('theta ', theta[0:4]*180/np.pi,'...',theta[-3:]*180/np.pi)

    #print('ring buffer',ring_buffer)
    import xtomo.loop_sino
    tomo, times_loop = xtomo.loop_sino.recon(sino, theta, algo = algo, tomo_out=tomo_out, 
          rot_center = rot_center, max_iter = max_iter, tol=tol, 
          GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk,  
          reg = reg, tau = tau, verbose = verboseall, 
          ncore=ncore, crop=chunks, mpring=ringbuffer)
    
    ###################################################################
    if rank>0: quit()
    dshape = sino.shape
    
    
    #times_loop['outfile']=times_of
    print(bold+"tomo shape",(num_slices_cropped,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    print("times full tomo", times_loop,flush=True)
    #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
    print(bold+"loop+setup time=", times_loop['loop']+times_loop['setup'], 'saving',+times_loop['write'], 'total',timer()-time0,endb)
    
    
    times_begin=timer()
    
    
    
    if file_out != '-1':
        tomosave(tomo_out, ring_buffer,times_loop)
    
    #time_saving=timer()-times_begin
    times_loop['write']+=timer()-times_begin
    times_loop['tot']=timer()-time0

    
    num_slices = tomo.shape[0]
    #num_rays = tomo.shape[1]
    #num_angles = sino.shape[1]
        
    times_loop['outfile']=times_of
    print(bold+"tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    print("times full tomo", times_loop,flush=True)
    #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
    print(bold+"loop+setup time=", times_loop['loop']+times_loop['setup'], 'saving',times_loop['write'], 'total', times_loop['tot'],endb, end='')
    
    
    if time_file == 1:
        print_times(fname,num_slices, num_rays, num_angles, args, times_loop)

    if 'exchange/tomo' not in fid:
        print("no tomogram to compare")
        #quit()
    else:
        print(endb+"\n comparing with tomo in the file", end=' ')
    #'exchange/tomo' in 
    true_obj = None
    try:
        if type(chunks)!=type(None):
            true_obj = fid['exchange/tomo'][chunks[0]:chunks[1],...]
        else:
            true_obj = fid['exchange/tomo'][...]
        compare_tomo(tomo,true_obj,chunks)

    except:
        print("no tomogram to compare")


       

    
#if __name__ == "__main__":
    """
    tomo, times_loop = xtomo.loop_sino.recon(sino, theta, algo = algo, tomo_out=tomo_out, 
          rot_center = rot_center, max_iter = max_iter, tol=tol, 
          GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk,  
          reg = reg, tau = tau, verbose = verboseall, 
          ncore=ncore, crop=chunks, mpring=ringbuffer)
    """
    
#    recon()
    """
    
        #fid = h5py.File(fname, 'w')
        
    # DID NOT SAVE
    
    #if ringbuffer <2 and (type(args['file_out']) is not type(None)) and args['file_out']!='-1': # did not save during iterations
    
        
        #   description 
    #args['file_out']=-1
    #if (type(args['file_out']) is not type(None)) and args['file_out']!='-1':  
    #        import os, sys
            
    #        cstring = ' '.join(sys.argv)
    #        tstring = str(times_loop)
    #        #print('cstring',cstring,'sysargv:',str(sys.argv) )
    #        #print('tstring,',tstring)        
    #        file_out = args['file_out']
    #        if file_out == '0': 
    #            file_out = os.path.splitext(fname)[0]
    #            file_out=file_out+'_'+algo+'_recon.tif'
    #            print('file out was 0, changed to:',file_out)
    #            args['file_out']=file_out
    #        else: print('file out',file_out)
    #              
            tstring = str(times_loop)
    
            if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
                from tifffile import imsave
                imsave(file_out,tomo, description = cstring+' '+tstring)
                #im = memmap(file_ou, shape=(num_slices,num_rays,num_rays), dtype='float32')
            else:
                import h5py
                fname=args['file_out']
                fid = h5py.File(fname, 'w')
                fid.create_dataset('exchange/tomo', data = tomo)
                fid.create_dataset('mish/command', data =cstring )
                fid.create_dataset('mish/times', data =tstring )
                
                fid.close()
    """            
