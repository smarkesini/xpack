#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:28:16 2019

@author: anu
"""

import h5py
import time
import numexpr as ne
import numpy as np
import tomopy
import os
import argparse
import dxchange
import warnings
#import json
#import textwrap

ap = argparse.ArgumentParser( formatter_class=argparse.RawTextHelpFormatter)
#    epilog='_'*60+'\n Note option precedence (high to low):  individual options, opts dictionary, fopts  \n'+'-'*60)
ap.add_argument("-f", "--file_in",  type = str,default ='/tomodata/tomobank/tomo_00001/tomo_00001.h5', help="h5 file in")
ap.add_argument("-o", "--file_out",   type = str, default='0', help="file out, default 0, 0: autogenerate name, -1: skip saving")
ap.add_argument("-r", "--rot_center",  type = int, help="rotation center, int ")
ap.add_argument("-c", "--chunks",  type = int, help="chunk size, int ")
args = vars(ap.parse_args())



#import parse

h5fname_in = args['file_in'] #'/tomodata/tomobank/tomo_00001/tomo_00001.h5'
chunks = args['chunks']
rot_center = args['rot_center'] #1024
h5fname_out = args['file_out']
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
    shape=( shape[1],shape[0],shape[2])

    return shape

import sys
def printbar(percent,string='    '):                
        sys.stdout.write('\r%s Progress: [%-60s] %3i%% ' %(string,'=' * (percent // 2), percent))
        sys.stdout.flush()
       
#import time
#from read_sigray_dirty import clean_sino

#h5fname ='/tomodata/sigray/Fly_data/Fly_preprocessed.h5'
#5fname_in ='/tomodata/sigray/Fly_data/Abs_Fly_tomo180_1p125um_4steps_34kV_4x4_2x_50s_042_whitedark.h5.h5'



def cleaning(
    filename,
    bffilename = None,
    inputPath = '/', #input path, location of the data set to reconstruct
    outputPath = None,# define an output path (default is inputPath), a sub-folder will be created based on file name
    outputFilename = None, #file name for output tif files (a number and .tiff will be added). default is based on input filename
    fulloutputPath = None, # definte the full output path, no automatic sub-folder will be created
    doFWringremoval = True,  # Fourier-wavelet ring removal
    ringSigma = 3, # damping parameter in Fourier space (Fourier-wavelet ring removal)
    ringLevel = 8, # number of wavelet transform levels (Fourier-wavelet ring removal)
    ringWavelet = 'db5', # type of wavelet filter (Fourier-wavelet ring removal)
    ringNBlock = 0, # used in Titarenko ring removal (doTIringremoval)
    ringAlpha = 1.5, # used in Titarenko ring removal (doTIringremoval)
    ringSize = 5, # used in smoothing filter ring removal (doSFringremoval)
    butterworth_cutoff = 0.25, #0.1 would be very smooth, 0.4 would be very grainy (reconstruction)
    butterworth_order = 2, # for reconstruction
    npad = None, # amount to pad data before reconstruction
    projused = None, # should be slicing in projection dimension (start,end,step) Be sure to add one to the end as stop in python means the last value is omitted 
    sinoused = None, # should be sliceing in sinogram dimension (start,end,step). If first value is negative, it takes the number of slices from the second value in the middle of the stack.
    angle_offset = 0, # this is the angle offset from our default (270) so that tomopy yields output in the same orientation as previous software (Octopus)
    anglelist = None, # if not set, will assume evenly spaced angles which will be calculated by the angular range and number of angles found in the file. if set to -1, will read individual angles from each image. alternatively, a list of angles can be passed.
    cor = None, # center of rotation (float). If not used then cor will be detected automatically
    corFunction = 'pc', # center of rotation function to use - can be 'pc', 'vo', or 'nm'
    voInd = None, # index of slice to use for cor search (vo)
    voSMin = -40, # min radius for searching in sinogram (vo)
    voSMax = 40, # max radius for searching in sinogram (vo)
    voSRad = 10, # search radius (vo)
    voStep = 0.5, # search step (vo)
    voRatio = 2.0, # ratio of field-of-view and object size (vo)
    voDrop = 20, # drop lines around vertical center of mask (vo)
    nmInd = None, # index of slice to use for cor search (nm)
    nmInit = None, # initial guess for center (nm)
    nmTol = 0.5, # desired sub-pixel accuracy (nm)
    nmMask = True, # if True, limits analysis to circular region (nm)
    nmRatio = 1.0, # ratio of radius of circular mask to edge of reconstructed image (nm)
    nmSinoOrder = False, # if True, analyzes in sinogram space. If False, analyzes in radiograph space
    useNormalize_nf = False, # normalize based on background intensity (nf)
    bfexposureratio = 1 #ratio of exposure time of bf to exposure time of sample
    ):

    start_time = time.time()
    print("Start {} at:".format(filename)+time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))

    outputFilename = os.path.splitext(filename)[0] if outputFilename is None else outputFilename
    outputPath = inputPath+'rec'+os.path.splitext(filename)[0]+'/' if outputPath is None else outputPath+'rec'+os.path.splitext(filename)[0]+'/'
    fulloutputPath = outputPath if fulloutputPath is None else fulloutputPath
    tempfilenames = [fulloutputPath+'tmp0.h5',fulloutputPath+'tmp1.h5']
    filenametowrite = fulloutputPath+outputFilename
    print(filenametowrite)

    print("cleaning up previous temp files", end="")
    for tmpfile in tempfilenames:
        try:
            os.remove(tmpfile)
        except OSError:
            pass

    print(", reading metadata")

    datafile = h5py.File(inputPath+filename, 'r')
    gdata = dict(dxchange.reader._find_dataset_group(datafile).attrs)
    numslices = int(gdata['nslices'])
    numangles = int(gdata['nangles'])
    print('There are ' + str(numslices) + ' sinograms and ' + str(numangles) + ' projections')
    angularrange = float(gdata['arange'])
    numrays = int(gdata['nrays'])
    npad = int(np.ceil(numrays * np.sqrt(2)) - numrays)//2 if npad is None else npad
    if projused is not None and (projused[1] > numangles-1 or projused[0] < 0): #allows program to deal with out of range projection values
        if projused[1] > numangles:
            print("End Projection value greater than number of angles. Value has been lowered to the number of angles " + str(numangles))
            projused = (projused[0], numangles, projused[2])
        if projused[0] < 0:
            print("Start Projection value less than zero. Value raised to 0")
            projused = (0, projused[1], projused[2])
    if projused is None:
        projused = (0,numangles,1)
    else:
    #if projused is different than default, need to chnage numangles and angularrange
    #dula attempting to do this with these two lines, we'll see if it works! 11/16/17
        testrange = range(projused[0],projused[1],projused[2])
        #+1 because we need to compensate for the range functions last value always being one less than the second arg
        angularrange = (angularrange/(numangles-1))*(projused[1]-projused[0])
        # want angular range to stay constant if we keep the end values consistent 
        numangles = len(testrange)   

# ndark = int(gdata['num_dark_fields'])
# ind_dark = list(range(0, ndark))
# group_dark = [numangles - 1]
    inter_bright = int(gdata['i0cycle'])
    if inter_bright > 0:
        group_flat = list(range(0, numangles, inter_bright))
        if group_flat[-1] != numangles - 1:
            group_flat.append(numangles - 1)
    elif inter_bright == 0:
        group_flat = [0, numangles - 1]
    else:
        group_flat = None

    # figure out the angle list (a list of angles, one per projection image)
    dtemp = datafile[list(datafile.keys())[0]]
    fltemp = list(dtemp.keys())
    firstangle = float(dtemp[fltemp[0]].attrs.get('rot_angle',0))
    anglegap = angularrange/(numangles-1)
    firstangle += anglegap*projused[0] #accounting for projused argument
    if anglelist is None:
        #the offset angle should offset from the angle of the first image, which is usually 0, but in the case of timbir data may not be.
        #we add the 270 to be inte same orientation as previous software used at bl832
        angle_offset = 270 + angle_offset - firstangle
        anglelist = tomopy.angles(numangles, angle_offset, angle_offset-angularrange)
    elif anglelist==-1:
        anglelist = np.zeros(shape=numangles)
        for icount in range(0,numangles):
            anglelist[icount] = np.pi/180*(270 + angle_offset - float(dtemp[fltemp[icount]].attrs['rot_angle']))
    
    #figure out how user can pass to do central x number of slices, or set of slices dispersed throughout (without knowing a priori the value of numslices)
    if sinoused is None:
        sinoused = (0,numslices,1)
    elif sinoused[0]<0:
        sinoused=(int(np.floor(numslices/2.0)-np.ceil(sinoused[1]/2.0)),int(np.floor(numslices/2.0)+np.floor(sinoused[1]/2.0)),1)   
   
    if cor is None:
        print("Detecting center of rotation", end="")
        if angularrange>300:
            lastcor = int(np.floor(numangles/2)-1)
        else:
            lastcor = numangles-1
        # I don't want to see the warnings about the reader using a deprecated variable in dxchange
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tomo, flat, dark, floc = dxchange.read_als_832h5(inputPath+filename,ind_tomo=(0,lastcor))
            if bffilename is not None:
                tomobf, flatbf, darkbf, flocbf = dxchange.read_als_832h5(inputPath+bffilename)
                flat = tomobf
        tomo = tomo.astype(np.float32)
        if useNormalize_nf:
            tomopy.normalize_nf(tomo, flat, dark, floc, out=tomo)
            if bfexposureratio != 1:
                tomo = tomo*bfexposureratio
        else:
            tomopy.normalize(tomo, flat, dark, out=tomo)
            if bfexposureratio != 1:
                tomo = tomo*bfexposureratio

        if corFunction == 'vo':
            # same reason for catching warnings as above
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cor = tomopy.find_center_vo(tomo, ind=voInd, smin=voSMin, smax=voSMax, srad=voSRad, step=voStep,
                                        ratio=voRatio, drop=voDrop)
        elif corFunction == 'nm':
            cor = tomopy.find_center(tomo, tomopy.angles(numangles, angle_offset, angle_offset-angularrange),
                                     ind=nmInd, init=nmInit, tol=nmTol, mask=nmMask, ratio=nmRatio,
                                     sinogram_order=nmSinoOrder)
        elif corFunction == 'pc':
            cor = tomopy.find_center_pc(tomo[0], tomo[1], tol=0.25)
        else:
            raise ValueError("\'corFunction\' must be one of: [ pc, vo, nm ].")
        print(", {}".format(cor))
    else:
        print("using user input center of {}".format(cor))


    tomo, flat, dark, floc = dxchange.read_als_832h5(inputPath+filename,
                        ind_tomo=range(projused[0],projused[1],projused[2]),
                        sino=(sinoused[0],sinoused[1], sinoused[2]))

    tomo = tomo.astype(np.float32,copy=False)
    tomopy.normalize(tomo, flat, dark, out=tomo)
    mx = np.float32(0.01)
    ne.evaluate('where(tomo>mx, tomo, mx)', out=tomo)
    tomopy.minus_log(tomo, out=tomo)
    tomo = tomopy.remove_stripe_fw(tomo, sigma=ringSigma, level=ringLevel, pad=True, wname=ringWavelet)
    tomo = tomopy.pad(tomo, 2, npad=npad, mode='edge')
    
    tomo = np.swapaxes(tomo,0,1)
    theta = anglelist
    
    print('It took {:.3f} s to process {}'.format(time.time()-start_time,inputPath+filename))
    
    return tomo, theta, cor

def clean_raw(h5fname_in=h5fname_in, h5fname_out=None):

    import os
    #h5fname =  h5fname_out

    if type(h5fname_out) == type(None):
        file_out = os.path.splitext(os.path.splitext(os.path.splitext(h5fname_in)[0])[0])[0]
        h5fname_out=file_out+'_clean.h5'
    elif h5fname_out=='0':
        file_out = os.path.splitext(os.path.splitext(os.path.splitext(h5fname_in)[0])[0])[0]
        h5fname_out=file_out+'_clean.h5'

    print("will be writing to:",h5fname_out)
    #h5fname ='/tomodata/tomobank_clean/tomo_00072_preprocessed1.h5'

    fid = h5py.File(h5fname_out, 'w')
    #dnames={'sino':"exchange/data", 'theta':"exchange/theta"}

    
    def write_h5(value,dirname="data"):
        fid.create_dataset(dirname, data = value)

    dnames={'sino':"exchange/data", 'theta':"exchange/theta", 'rot_center':'exchange/rot_center'}
    start_loop_time =time.time()   
#    chunks = [0, num_slices]

    #nchunks=int(1)
    #max_chunks=num_slices

    print('cleaning the whole sinogram')
    h5fname = h5fname_in
    data, theta, rot_center = cleaning(h5fname)
    data = np.ascontiguousarray(data)
    vname='sino'
    print("writing {} to: {}".format(vname,dnames[vname]))
    write_h5(data,dirname=dnames['sino'])
    vname='theta'
    print("writing {} to: {}".format(vname,dnames[vname]))
    write_h5(theta,dirname=dnames['theta'])


    if rot_center != None:
        vname='rot_center'
        print("writing {} to: {}".format(vname,dnames[vname]))
        write_h5(rot_center,dirname='exchange/rot_center')

    fid.close()

    return data, theta, rot_center, h5fname_out



#from reconstruct import recon

#h5fname_in = 'rock.h5'
#h5fname_out = 'rock_output_data.h5'
#data, theta, cor = cleaning(h5fname_in)

data, theta, rot_center, h5fname_out = clean_raw(h5fname_in, h5fname_out)
