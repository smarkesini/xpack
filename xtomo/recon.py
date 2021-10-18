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
                #rot_center = np.round( fid['exchange/rot_center'][()] )
                rot_center =  fid['exchange/rot_center'][()] 
                
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
    #if rank>0: quit()
    if rank==0: 
        dshape = sino.shape
        
        
        #times_loop['outfile']=times_of
        print(bold+"tomo shape",(num_slices_cropped,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
        print("times full tomo", times_loop,flush=True)
        #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
        print(bold+"loop+setup time=", times_loop['loop']+times_loop['setup'], 'saving',+times_loop['write'], 'total',timer()-time0,endb)
        
        
        times_begin=timer()
        if algo[0:min(len(algo),6)]=='tomopy' and type(tomo_out)!=type(None):
            #print('saving tomopy output')
            tomo_out[...]=tomo[...]
            tomo_out.flush()
        
        
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
