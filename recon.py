import numpy as np
#from reconstruct import  recon_file #, recon
#from xtomo.loop_sino import  recon_file #, recon

from xtomo.communicator import rank, mpi_size, mpi_barrier

#import argparse

from timeit import default_timer as timer

def tomofile(file_out, file_in=None, algo='iradon', shape_tomo=(1,1,1), ring_buffer=0, rank=0):
    tomo_out=None
    
    fname=file_in
    if file_out=='-1': # not saving
        if ring_buffer>1: 
            if rank==0: print('output file was "-1", no ring buffer without output file')
            ring_buffer=np.mod(ring_buffer,2)
    
    
    if (type(file_out) is not type(None)) and file_out!='-1':  
            if rank==0: print("setting up output file")
    
            import os, sys
            
            cstring = ' '.join(sys.argv)
            #file_out = args['file_out']

            if file_out == '0' or file_out=='*': 
                file_out = os.path.splitext(fname)[0]
                file_out=file_out+'_'+algo+'_recon.tif'
                if rank ==0: print('file out was 0, changed to:',file_out)
                #args['file_out']=file_out
            elif file_out == '0.h5' or file_out=='*.h5':
                file_out = os.path.splitext(fname)[0]
                file_out=file_out+'_'+algo+'_recon.h5'
                if rank ==0: print('file out was *.h5, changed to:',file_out)
                #args['file_out']=file_out
            #else: print('file out',file_out)
            
            
            ## file out mapping
            if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
                from tifffile import memmap
    
                if rank == 0:       
                    if os.path.exists(file_out): 
                        if rank ==0: print('file exist, overwriting')
                        tomo_out = memmap(file_out) # file should already exist
                        if tomo_out.shape==shape_tomo:
                            #print("reusing")
                            tomo_out = memmap(file_out) # file  already exist
                        else:
                            if rank ==0: print("new shape",tomo_out.shape)
                            tomo_out = memmap(file_out, shape=shape_tomo, dtype='float32',description=cstring)
                    else:
                        tomo_out = memmap(file_out, shape=shape_tomo, dtype='float32',description=cstring)
    
                    #print('rank 0 created file')
                    mpi_barrier()
                    # print('rank 0 created file and passed barrier')
    
                else: 
                    #print('rank',rank,'wating for barrier')
                    mpi_barrier()
                    #print('rank',rank,'barrier done')
                    tomo_out = memmap(file_out) # file should already exist
                    
            elif os.path.splitext(file_out)[-1] in ('.h5','.hdf5'):
                import h5py
                # fname=args['file_out']
                if rank==0:
                    print('creating h5 file',file_out,end=' ')
                    fid = h5py.File(file_out, 'w')
                    fid.create_dataset('mish/command', data =cstring )            
                    #tomo_out=fid.create_dataset('/exchange/tomo', (num_slices,num_rays,num_rays) , chunks=(max_chunk,num_rays,num_rays),dtype='float32')
                    tomodset=fid.create_dataset('/exchange/tomo', shape_tomo , dtype='float32')
                    tomo_out=tomodset[...]
                    mpi_barrier()
                else:
                    mpi_barrier() # file should already exist
                    fid = h5py.File(file_out, 'a')
    
                    tomo_out=fid['/exchange/tomo']
    return tomo_out, ring_buffer

#if __name__ == "__main__":
def recon():

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
    #print('tolerance',tol)
    reg = args['reg']
    tau = args['tau']
    verboseall = args['verbose']
    chunks = args['chunks']
    print('chunks type', type(chunks))
    if type(chunks)==type(None):
        chunks_val=-1
    else:
        chunks_val = chunks[0]
    #print(chunks_val[0])
    ncore = args['ncore']
    
    ringbuffer = args['ring_buffer']
    file_out = args['file_out']
    
    #print("testing_recon max_iter",max_iter)
    #algo='tvrings'
    
    
    if rank==0: print(args)
    
    if type(args['file_in']) is not type(None):
        fname=args['file_in']
        import h5py
        fid= h5py.File(fname, "r")
        sino  = fid['exchange/data']
        theta = fid['exchange/theta']
        num_angles =  sino.shape[1]
        num_rays   =  sino.shape[2]
        num_slices =  sino.shape[0]
    
        #fid.close()    
    
    elif simulate:
        # from simulate_data import get_data as gdata  
        from xtomo.simulate_data import init
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
    
    # compute the crop points:
    if type(chunks)!=type(None):
        chunks = np.array(chunks)
        if chunks.size==1: 
            chunks=np.int64(chunks)
            chunks=np.array([0,chunks[0]])+(num_slices-chunks[0])//2
            
        chunks=np.clip(np.array(chunks),0,num_slices)
        num_slices_cropped=np.int(chunks[1]-chunks[0])
        print('='*50)
        print(num_slices_cropped, type(num_slices_cropped), num_slices, type(num_slices))
        print('='*50)
        
    else:
        num_slices_cropped=num_slices
        

    
    #tomo_out=None
    
    times_of=timer()
    tomo_out, ring_buffer = tomofile(file_out, file_in=fname, algo=algo, shape_tomo=(num_slices_cropped,num_rays,num_rays), ring_buffer=ringbuffer)

    times_of=timer()-times_of
    if rank ==0: print('output file setup time',times_of)
    
          
    #print('ringbuffer',ringbuffer)
    """
    tomo, times_loop, dshape = recon_file(fname,dnames=None, tomo_out=tomo_out, algo = algo,
                                          rot_center = rot_center, 
                                          max_iter = max_iter, tol=tol, 
                                          GPU = GPU, shmem = shmem, 
                                          max_chunk_slice=max_chunk, 
                                          reg = reg, tau = tau, verbose=verboseall,
                                          ncore=ncore, chunks=chunks,mpring=ringbuffer)
    """
    import xtomo.loop_sino
    tomo, times_loop = xtomo.loop_sino.recon(sino, theta, algo = algo, tomo_out=tomo_out, 
          rot_center = rot_center, max_iter = max_iter, tol=tol, 
          GPU = GPU, shmem = shmem, max_chunk_slice=max_chunk,  
          reg = reg, tau = tau, verbose = verboseall, 
          ncore=ncore, crop=chunks, mpring=ringbuffer)
    
    ###################################################################
    if rank>0: quit()
    dshape = sino.shape
    
    bold='\033[1m'
    endb= '\033[0m'
    
    #times_loop['outfile']=times_of
    print(bold+"tomo shape",(num_slices_cropped,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    print("times full tomo", times_loop,flush=True)
    #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
    print(bold+"loop+setup time=", times_loop['loop']+times_loop['setup'], 'saving', 'total',endb)
    
    
    times_begin=timer()
    
    
    
    #print("done recon")
    #print(type(tomo_out))

    def tomosave(tomo_out, ring_buffer):
        if ringbuffer >1:
            import os.path
            file_out = tomo_out.filename
        
            if os.path.splitext(file_out)[-1] in ('.h5','.hdf5'):
                tstring = str(times_loop)
                fid.create_dataset('mish/times', data =tstring )            
                #fid.close()        
            elif os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
                #tomo=tomo_out
                
                print('flushing',end=' ')
                if type(tomo_out)==np.memmap:
                    tomo_out.flush()
                if type(tomo)==np.memmap:
                    tomo.flush()
                print('\r flushed, t=',timer()-times_begin)#,end=' ')
        elif type(tomo_out)!=type(None):
                
                #tomo_out.close()
        #        pass
    #    elif  (type(file_out) is not type(None)) and file_out!='-1':
            print('saving...',end=' ')
            tstring = str(times_loop)
            # did not save during iterations
            #if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
            tomo_out.flush()
            del tomo_out
            #from tifffile import imsave
                #imsave(file_out,tomo, description = cstring+' '+tstring)

    if file_out != '-1':
        tomosave(tomo_out, ring_buffer)
    
        
    #print("flushed")
        
            
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
    
    times_loop['outfile']=times_of
    print(bold+"tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
    print("times full tomo", times_loop,flush=True)
    #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
    print(bold+"loop+setup time=", times_loop['loop']+times_loop['setup'], 'saving',time_saving, 'total', time_tot,endb, end='')
    
    
    if time_file == 1:
        root_name=os.path.expanduser('~/data/')
        fname = 'runtime_data.txt'
        f = open(root_name + fname, 'a+')
        print('\n Created new file', fname, 'if it does not exist; otherwise appending.')
        
        dataset = args['file_in']
        f.write('\nTesting: %s using %s algorithm ' % (dataset, algo))
        if type(max_chunk) is not type(None):
            f.write('with max_chunk = %d ' % (max_chunk))
        if type(chunks) is not type(None):
            f.write('and chunk = %d' % (chunks_val))
        
        f.write('\nTomogram shape = (%d, %d, %d) \nNumber of angles = %d \nMPI size = %d' % (num_slices, num_rays, num_rays, num_angles, mpi_size))
        f.write('\nLoop time = %f, Setup time = %f, Saving time = %f, Total time = %f\n\n' % (times_loop['loop'], times_loop['setup'], time_saving, time_tot))
        f.write('--------------------------------------------------------------------\n')
    
    
    if 'exchange/tomo' not in fid:
        print("no tomogram to compare")
        #quit()
    else:
        print(endb+"\n comparing with tomo in the file", end=' ')
    #'exchange/tomo' in 
    true_obj = None
    try:
        #true_obj = get_data('tomo')[...]
    #    crop=chunks
    #    if crop!=None:
    #        if len(crop)==1:
    #            loop_offset=true_obj.shape[0]//2-crop//2
    #        else:
    #            loop_offset=crop[0]
    #    true_obj = true_obj[loop_offset:loop_offset+num_slices,...]
        #true_obj = true_obj[...]
        if type(chunks)!=type(None):
            true_obj = fid['exchange/tomo'][chunks[0]:chunks[1],...]
        else:
            true_obj = fid['exchange/tomo'][...]
    
        
    
        #print("comparing with truth, summary coming...")
    except:
        print("no tomogram to compare")
    
        #quit()
        #print("no truth, quitting \n")
        #true_obj = None
    
    #print('still alive')
    
    if type(true_obj) == type(None): 
        print("no tomogram to compare")
        #quit()
    
    else:
    #    print('still alive 2')
    
        #import fubini
        from xtomo.fubini import masktomo
        msk_tomo,msk_sino=masktomo(num_rays, np, width=.95)
    #msk_tomo=msk_tomo[0,...]
    
    #    if ringbuffer>1:
    #        print('loading back the file for comparison, some patience...')
    
    #t2i = lambda x: x[num_slices//2,:,:].real
        #t2i = lambda x: x[num_slices//2,:,:].real
        #tomo0c=t2i(tomo)*msk_tomo[0,...]
        #plt.imshow(np.abs(tomo0c))
        #print("phantom shape",true_obj.shape, "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
        #print("reading tomo, shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, "max_iter",max_iter)
    
        print('\r'+'*'*60)
        #print('type',type(tomo))
        print('tomo norm',np.linalg.norm(np.reshape(tomo,(-1))))
        print('\r'+'*'*60)
        
        #scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
        scale   = lambda x,y: np.dot(np.reshape(x,(-1)), np.reshape(y,(-1)))/np.linalg.norm(x)**2
        #scale   = lambda x,y: np.dot(x.flat(), y.flat())/np.linalg.norm(x)**2
        rescale = lambda x,y: scale(x,y)*x
        ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
        #ssnr2    = lambda x,y: ssnr(x,y)**2
        
        print('\r ...', end=' ')
        snr=ssnr(true_obj,tomo)
        print('\r done computing',end=' ')
        #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
    
        #bold='\033[1m'
        #endb= '\033[0m'
        #print(bold+"tomo shape",(num_slices,num_rays,num_rays), "n_angles",num_angles, ', algorithm:', algo,", max_iter:",max_iter,",mpi size:",mpi_size,",GPU:",GPU)
        #print("times full tomo", times_loop)
        #print("loop+setup time=", times_loop['loop']+times_loop['setup'], "snr=", ssnr(true_obj,tomo),endb)
        print('\r',' '*50+'\r' +bold+"snr=", snr,endb,'\n')
    
    
    
if __name__ == "__main__":
    recon()