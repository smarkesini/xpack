import numpy as np

from .communicator import rank, mpi_size, mpi_barrier

bold='\033[1m'
endb= '\033[0m'


# generate output file name from file_in and algo, tif by default, h5 if file_out=*.h5
def getfilename(file_in, algo='', file_out='*'):
    import os

    fname=file_in
    
    if file_out == '0' or file_out=='*': 
        file_out = os.path.splitext(fname)[0]
        file_out=file_out+'_'+algo+'_recon.tif'
        if rank ==0: print('file out was */0, changed to:',file_out)
        #args['file_out']=file_out
    elif file_out == '0.h5' or file_out=='*.h5':
        file_out = os.path.splitext(fname)[0]
        file_out=file_out+'_'+algo+'_recon.h5'
        if rank ==0: print('file out was *.h5, changed to:',file_out)
    
    return file_out

# map to file_out
def maptomofile(file_out, shape_tomo=(1,1,1), ring_buffer=0, cstring=None):
    tomo_out=None
    #print(bold+"setting up output file"+endb)
    
    #fname=file_in
    if file_out=='-1': # not saving
        if ring_buffer>1: 
            return tomo_out, ring_buffer
    
    if (type(file_out) is not type(None)) and file_out!='-1':  
            if rank==0: print(bold+"setting up output file"+endb)
    
            import os

            if type(cstring)==type(None):
                import sys            
                cstring = ' '.join(sys.argv)
            
 
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
                        else: #,bigtiff=True
                            if rank ==0: print("new shape",tomo_out.shape)
                            tomo_out = memmap(file_out, shape=shape_tomo, dtype='float32',bigtiff=True, description=cstring)
                    else:
                        tomo_out = memmap(file_out, shape=shape_tomo, dtype='float32',bigtiff=True,description=cstring)
    
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



# map to file_out
def tomofile(file_out, file_in=None, algo='iradon', shape_tomo=(1,1,1), ring_buffer=0):
    tomo_out=None
    #fname=file_in
    if file_out=='-1': # not saving
        tomo_out=None
        if ring_buffer==2: ring_buffer=0
        elif ring_buffer==3: ring_buffer=1
        elif ring_buffer==6: ring_buffer=4
        elif ring_buffer==7: ring_buffer=5
        #if ring_buffer>1: 
        #    return tomo_out, ring_buffer
        return tomo_out, ring_buffer

            
    if (type(file_out) is not type(None)) and file_out!='-1':  
            if rank==0: print("setting up output file")
    
            #file_out = args['file_out']
            file_out = getfilename(file_in, algo=algo, file_out=file_out)
            import sys            
            cstring = ' '.join(sys.argv)
            tomo_out, ring_buffer = maptomofile(file_out, shape_tomo, ring_buffer, cstring)
    return tomo_out, ring_buffer

def tomosave(tomo_out, ring_buffer,times_loop):
    ringbuffer=ring_buffer
    if ringbuffer >1:
        import os.path
        file_out = tomo_out.filename
        
        if os.path.splitext(file_out)[-1] in ('.h5','.hdf5'):
            import h5py
            tstring = str(times_loop)
            fid = h5py.File(file_out, 'a')
        
            fid.create_dataset('mish/times', data =tstring )            
            #fid.close()        
        elif os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
            #tomo=tomo_out
            
            print('saving',end=' ')
            if type(tomo_out)==np.memmap:
                tomo_out.flush()

    elif type(tomo_out)!=type(None):
            
            #tomo_out.close()
    #        pass
#    elif  (type(file_out) is not type(None)) and file_out!='-1':
        print('flushing...',end=' ')
        tstring = str(times_loop)
        # did not save during iterations
        #if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
        tomo_out.flush()
        del tomo_out
        #from tifffile import imsave

def print_times(fname,num_slices, num_rays, num_angles, args, times_loop):
    import os
    import datetime
    max_chunk = args['max_chunk_slice']
    algo=args['algo']
    chunks = args['chunks']
    
    root_name=os.path.expanduser('~/data/')
    fname = 'runtime_data.txt'
    f = open(root_name + fname, 'a+')
    print('\n Created new file', fname, 'if it does not exist; otherwise appending.')
    
    dataset = args['file_in']
    f.write('\nCurrent time %s.' % (datetime.datetime.now()))
    f.write('\nTesting: %s using %s algorithm ' % (dataset, algo))
    
    if type(max_chunk) is not type(None):
        f.write('with max_chunk = %d ' % (max_chunk))
    if type(chunks) is not type(None):
        if type(chunks)==int: f.write('and chunk = %d' % (chunks[0]))
        else: f.write('and chunk = %d' % (chunks[1]-chunks[0]))
    
    f.write('\nTomogram shape = (%d, %d, %d) \nNumber of angles = %d \nMPI size = %d' % (num_slices, num_rays, num_rays, num_angles, mpi_size))
   #f.write('\nLoop time = %f, Setup time = %f, Saving time = %f, Total time = %f\n\n' % (times_loop['loop'], times_loop['setup'], time_saving, time_tot))
    f.write('\nLoop time = %f, Setup time = %f, Saving time = %f, Total time = %f\n\n' % (times_loop['loop'], times_loop['setup'],times_loop['write'], times_loop['tot']))

    f.write('--------------------------------------------------------------------\n')
  
