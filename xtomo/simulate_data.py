#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import h5py
import os.path
import tomopy

#from testing_setup import setup_tomo
import numpy as np
#from timeit import default_timer as timer
#from fubini import radon_setup as radon_setup

from timeit import default_timer as timer 

xp = np

#file_name="/data/tomosim/shepp_logan.h5"
#file_name="/data/tomosim/shepp_logan"

root_name="/data/tomosim/shepp_logan"



    #f= h5py.File(h5fname, "r",rdcc_nbytes=csize)

def write_h5(value,file_name,dirname="data"):
    print("writing to :", file_name, dirname)
    with h5py.File(file_name, 'a') as f:
        f.create_dataset(dirname, data = value)
        f.close()
        


def read_h5(file_name,dirname="data",chunks=None):   

    with h5py.File(file_name, "r") as f:
        #print("reading from:", dirname)
        if type(chunks)==type(None):
            data = f[dirname][:]
        else:
            data = f[dirname][chunks[0]:chunks[1],...]

    #print(data.shape)
    return data
 
def generate_Shepp_Logan(cube_shape):

   return tomopy.misc.phantom.shepp3d(size=cube_shape, dtype='float32')


def setup_tomo (num_slices, num_angles, num_rays, k_r=1, kernel_type = 'gaussian',width=.5):

    #num_rays=num_rays//2
    num_rays_obj=np.int(np.floor(num_rays*width/2)*2)
    true_obj_shape = (num_slices, num_rays_obj, num_rays_obj)
    true_obj = generate_Shepp_Logan(true_obj_shape)
    
    
    pad_1D        = (num_rays-num_rays_obj)//2
    padding_array = ((0, 0), (pad_1D, pad_1D), (pad_1D, pad_1D))

    true_obj = xp.pad(true_obj, padding_array, 'constant', constant_values = 0)
    
    #theta    = xp.arange(0., 180., 180. / num_angles,dtype='float32')*xp.pi/180.

    return true_obj


def fname(num_slices,num_rays,num_angles,obj_width,root_name = root_name):    
    file_name="{}_{}_{}_{}_{}.h5".format(root_name,num_slices,num_angles,num_rays,int(obj_width*100))
    return file_name


def simulate(num_slices,num_rays,num_angles,obj_width,root_name = root_name):
        
    #grp="sim_{}_{}_{}_{}".format(num_slices,num_angles,num_rays,int(obj_width*100))
    #grp="sim"

    #dname_tomo="{}/tomo".format(grp)
    #dname_sino="{}/sino".format(grp)
    #dname_theta="{}/theta".format(grp)
    dname_tomo="tomo"
    dname_sino="sino"
    dname_theta="theta"
    
    #file_name="{}_{}_{}_{}_{}.h5".format(root_name,num_slices,num_angles,num_rays,int(obj_width*100))
    file_name=fname(num_slices,num_rays,num_angles,obj_width,root_name = root_name)
    

    print("will be writing to :",file_name, dname_tomo)
    print("will be writing to :",file_name, dname_sino)
    print("will be writing to :",file_name, dname_theta,flush=True)
    if np.mod(num_angles,2)==0:
        theta    = np.arange(0., 180., 180. / num_angles,dtype='float64')*np.pi/180.
    else:
        print("all the way to 180")
        theta    = np.linspace(0, 180., num= num_angles)*np.pi/180.
        theta=theta.astype('float32')
        #theta    = np.arange(0., 180., 180. / num_angles,dtype='float64')*xp.pi/180.
        
    
    print('setting up the phantom:{}'.format((num_slices,num_rays,num_rays)),"num angles", num_angles,"filling",obj_width,'%',flush=True)
    
    #print("setting up the phantom,...")
    start=timer()
    true_obj=setup_tomo(num_slices, num_angles, num_rays, np, width=obj_width)
    end = timer()
    time_phantom=(timer()- start)
    print("phantom setup time=", time_phantom)
    
    
    
    #xp=np
    true_obj=np.array(true_obj)
    theta=np.array(theta)
    
    #true_obj=true_obj[num_slices//2:num_slices//2+1,:,:]
    #num_slices=1
    
    print("setting up radon. ", end = '')
    start=timer()
    #from testing_setup import setup_tomo
    #from fubini import radon_setup as radon_setup
    from fubini import radon_setup

    radon,iradon = radon_setup(num_rays, theta, xp=np, kernel_type = 'gaussian', k_r =1,width=obj_width)
    time_radonsetup=(timer() - start)
    print("time=", time_radonsetup)
    
    print("doing radon. ", flush = True)
    start=timer()
    sinogram = radon(true_obj)
    end = timer()
    time_radon=(end - start)
    print("time=", time_radon)
    
    write_h5(theta,file_name,dirname=dname_theta)
    write_h5(true_obj,file_name,dirname=dname_tomo)

    #write_h5(true_obj,file_name,dirname=dname_tomo)
    write_h5(sinogram,file_name,dirname=dname_sino)
    print("done simulating")
    
    return dname_tomo,dname_sino,dname_theta

#theta1= read_h5(file_name,dirname=dname_theta)
#tomo1= read_h5(file_name,dirname=dname_tomo)

csize=0
global fid
fid = None

def init(num_slices,num_rays,num_angles,obj_width,which_data, root_name = root_name):
    file_name = fname(num_slices,num_rays,num_angles,obj_width,root_name = root_name)
    if not os.path.isfile(file_name):
        print("data doesn't exist, generating...")
        simulate(num_slices,num_rays,num_angles,obj_width,root_name = root_name)
 
    global fid
    fid= h5py.File(file_name, "r",rdcc_nbytes=csize)
    
def fclose():
    global fid
    fid.close()

def get_data(num_slices,num_rays,num_angles,obj_width,which_data, root_name = root_name, chunks=None):
    
    file_name = fname(num_slices,num_rays,num_angles,obj_width,root_name = root_name)
    
    #file_name="{}_{}_{}_{}_{}.h5".format(root_name,num_slices,num_angles,num_rays,int(obj_width*100))
    #print(file_name)
    if not os.path.isfile(file_name):
        print("data doesn't exist, generating...")
        simulate(num_slices,num_rays,num_angles,obj_width,root_name = root_name)
        
    global fid     
    if type(fid)==type(None):
        init(num_slices,num_rays,num_angles,obj_width,which_data, root_name = root_name)
    #print("tipe fid",type(fid))
   
    if which_data =='theta':
        #theta= read_h5(file_name,dirname="theta",chunks=chunks)
        #return theta
        return fid['theta']
    elif which_data =='sino':
        return fid['sino']
        #sino= read_h5(file_name,dirname="sino",chunks=chunks)
        #return sino
    elif which_data =='tomo':
        return fid['tomo']
        #tomo = read_h5(file_name,dirname="tomo",chunks=chunks)
        #return tomo
    print("{} doesn't exist".format(which_data))



#    try: 
#        theta=read_h5(file_name,dirname=dname_theta)
#    except:
#        print("simulating first",flush=True)
#        from simulate_data import simulate
#        simulate(num_slices,num_rays,num_angles,obj_width)
#        theta=read_h5(file_name,dirname=dname_theta)



 
"""
obj_size = 1024*2//2
num_slices = 32# size//2

num_slices = 8# size//2
num_angles =  obj_size//2
#num_angles =  960
num_rays   = obj_size
obj_width=0.95

#file_name="{}_{}_{}_{}_{}.h5".format(root_name,num_slices,num_angles,num_rays,int(obj_width*100))

simulate(num_slices,num_rays,num_angles,obj_width,root_name=root_name)
"""
 


