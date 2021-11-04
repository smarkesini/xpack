import numpy as np

from .communicator import rank, mpi_size, mpi_barrier, comm
import h5py
import os
#from xtomo.communicator import rank, mpi_size, comm

bold='\033[1m'
endb= '\033[0m'


# generate output file name from file_in and algo, tif by default, h5 if file_out=*.h5
def getfilename(file_in, algo='', file_out='*'):
    import os

    fname=file_in
    
    if file_out == '0' or file_out=='*' or file_out == '0.tif' or file_out == '*.tif': 
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
    fid = None
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
                        if rank ==0: print('file exist, overwriting', file_out)
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
                fid = h5py.File(file_out, 'w', driver='mpio', comm=comm)
                #print('hello 0')
                tomodset=fid.create_dataset('/exchange/tomo', shape_tomo , dtype='float32')
                #print('hello 1')
                tomo_out=fid['/exchange/tomo']
                #print('hello 2')
                fid.create_dataset('/mish/command', (1) ,dtype=h5py.string_dtype('utf-8',len(cstring)))
                fid.create_dataset('/mish/times', (1) ,dtype=h5py.string_dtype('utf-8',300))
                
                #
                # fid.create_dataset('mish/command', data =cstring )   
                # fid.create_dataset('mish/command', shape = (len(cstring),1),dtype='S')
                # print('hello 3')
                
                
                #fid = h5py.File(file_out, 'w')
                if rank==0:
                    #pass
                    # print('trying here')
                    #print('saving command to h5 file',file_out,end=' ')
                    fid['mish/command'][...]=cstring
                    #fid['mish/command']=cstring
                    #fid.create_dataset('mish/command', data =cstring )            
                    #print('created command' )
                    # fid = h5py.File(file_out, 'w')
                    #fid.create_dataset('mish/command', data =cstring )            
                    #tomo_out=fid.create_dataset('/exchange/tomo', (num_slices,num_rays,num_rays) , chunks=(max_chunk,num_rays,num_rays),dtype='float32')
                    #tomodset=fid.create_dataset('/exchange/tomo', shape_tomo , dtype='float32')
                    #tomo_out=tomodset
                    #mpi_barrier()
                else:
                    pass
                    #fid.create_dataset('mish/command', data =cstring )      
                    # fid.create_dataset('mish/command')
                    #mpi_barrier() # file should already exist
                    
                    #fid = h5py.File(file_out, 'a')
    
                    #tomo_out=fid['/exchange/tomo']
    #print('tomo out',type(tomo_out))
    #print('#########hello????????????????????????')
    #print('')
    return tomo_out, ring_buffer, fid



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
            # print('\n ---',file_out,'\n')
            import sys            
            cstring = ' '.join(sys.argv)
            #if rank==0: print('file_out', file_out)
            #if rank==0: print('cstring', cstring)
            tomo_out, ring_buffer, fid_out = maptomofile(file_out, shape_tomo, ring_buffer, cstring)
    #print('hello')
    return tomo_out, ring_buffer, fid_out

def tomosave(tomo_out, ring_buffer,times_loop):
    # print('@@@',type(tomo_out), np.sum(tomo_out),ring_buffer)
    ringbuffer=ring_buffer
    if ringbuffer >1:
        import os.path
        try:
            file_out = tomo_out.filename
        except:
            import re
            file_out = re.findall(r'"(.*?)"', str(tomo_out.file), re.DOTALL)[0]
        
        if os.path.splitext(file_out)[-1] in ('.h5','.hdf5'):
            #print('-----------')
            # print('--++--------')
            import h5py
            tstring = str(times_loop)
            fid = h5py.File(file_out, 'a')
           
            #fid.create_dataset('mish/times', data =tstring )     
            fid.flush()
            #print('-----------')
            #print('++++++++++++')
            #fid.close()        
        elif (os.path.splitext(file_out)[-1] in ('.tif','.tiff')) or type(tomo_out)==np.memmap:
            #tomo=tomo_out
            
            print('saving',end=' ')
            tomo_out.flush()
#            if type(tomo_out)==np.memmap:
#                tomo_out.flush()

    elif type(tomo_out)!=type(None):
            
            #tomo_out.close()
    #        pass
        #print('----',type(tomo_out), np.sum(tomo_out))
#    elif  (type(file_out) is not type(None)) and file_out!='-1':
        #print('@@@',type(tomo_out), np.sum(tomo_out),ring_buffer)
        #print('flushing...',end=' ')
        import os.path
        tstring = str(times_loop)
        # did not save during iterations
        #if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
        #print('+++@@@',type(tomo_out), np.sum(tomo_out),ring_buffer)
        
        #print(type(tomo_out) == H5py._hl.dataset.Dataset)
        #if os.path.splitext(file_out)[-1] in ('.tif','.tiff'):
        #    print('flushing...',end=' ')
        #    tomo_out.flush()
        import h5py
        
        if type(tomo_out)!=h5py._hl.dataset.Dataset:
            #print('timings')
            tomo_out.flush()
        else:
            tomo_out.file['mish/times'][0:len(tstring)]=tstring
            
            #print('saving times')
            #tomo_out.file.create_dataset('mish/times', (1) ,dtype=h5py.string_dtype('utf-8',len(tstring)))
            # tomo_out
            #tomo_out.file.create_dataset('mish/times', data = tstring)
            #print('saving times')
            #tomo.file.create_dataset('mish/times1', data='cstring')
        # try:
        #     # print('+++---',type(tomo_out), np.sum(tomo_out))
        #     #del tomo_out
        # except:
        #     pass
        
        #print('^^^@@@',type(tomo_out), np.sum(tomo_out),ring_buffer)
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
  
