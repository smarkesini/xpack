#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:25:06 2019

@author: anu
"""

#Gridrec algorithm based on pseudocode using Gaussian kernel
#from numba import jit
import matplotlib.pyplot as plt
import numpy as np
#import h5py
#import dxchange
import tomopy
#from scipy import 
import imageio
import os
import cupy as cp

#start_time = time.time()


#Read in file
#filename  = '20190524_085542_clay_testZMQ.h5'

#inputPath = '/Users/anu/Desktop/Summer/'
#file      = h5py.File(inputPath+filename, 'r')
#gdata     = dict(dxchange.reader._find_dataset_group(file).attrs)

#num_slices = int(gdata['nslices']) #number of slices
#num_angles = int(gdata['nangles']) #number of angles
#num_rays   = int(gdata['nrays'])
#Nr = len(r)


#print("The total number of slices is", num_slices, "and the total number of angles is", num_angles)

def gaussian_kernel(x, sigma, xp): #using sigma=2 since this is the default value for sigma
    kernel1 = xp.exp(-1/2 * (x/sigma)**2)
    return kernel1 
    
def keiser_bessel(x, k_r, beta,xp):
#    kernel1 = np.i0(beta*np.sqrt(1-(2*x/(k_r))**2))/np.abs(np.i0(beta)) 
    kernel1 = (xp.abs(x) <= k_r/2) * xp.i0(beta*xp.sqrt((xp.abs(x) <= k_r/2)*(1-(2*x/(k_r))**2)))/xp.abs(xp.i0(beta)) 
    return kernel1

def K2(x, y, kernel_type, xp, k_r=2, beta=1, sigma=2): #k_r and beta are parameters for keiser-bessel
    if kernel_type == 'gaussian':
        kernel2 = gaussian_kernel(x,sigma,xp) * gaussian_kernel(y,sigma,xp)
        return kernel2
    elif kernel_type == 'kb':
        kernel2 = keiser_bessel(x,k_r,beta,xp) * keiser_bessel(y,k_r,beta,xp)
        return kernel2
    else:
        print("Invalid kernel type")

#STACK SINOGRAMS
#data = dxchange.read_als_832h5(filename)
#print("---- Data read -----")
#stack_sino = data[0]

#np.set_printoptions(threshold=sys.maxsize)
#
#k_r = 3
    
def create_unique_folder(name):
   
    id_index = 0
    while True:
        id_index += 1
        folder = "Sim_" + name + "_id_" + str(id_index) + "/"
        
        if not os.path.exists(folder):
            os.makedirs(folder)
            break

    return folder


def save_cube(cube, base_fname):

    for n in range(0, cube.shape[0]):
        tname = base_fname + "_" + str(n) + '.png'
        imageio.imwrite(tname, cube[n,:,:].astype(np.uint8))
        

def generate_Shepp_Logan(cube_shape):

   return tomopy.misc.phantom.shepp3d(size=cube_shape, dtype='float32')


def forward_project(cube, theta):

    return tomopy.sim.project.project(cube, theta, pad = False, sinogram_order=False)

    
def clip(a, lower, upper):

    return min(max(lower, a), upper)  


def grid_rec_one_slice(qt, theta_array, num_rays, k_r, kernel_type): #use for backward proj
    
    qxy = np.zeros((num_rays + k_r * 2 + 1, num_rays + k_r * 2 + 1), dtype=np.complex64)
    for q in range(num_rays):
        ind = 0
        
        for theta in theta_array:
            px = -(q - num_rays/2)*np.sin(theta)+(num_rays/2)
            py = (q - num_rays/2)*np.cos(theta)+(num_rays/2)  
            
            for ii in range(-k_r, k_r+1):
                
                for jj in range(-k_r, k_r+1):
                    kernel = K2(px-round(px)-ii, py-round(py)-jj, kernel_type, k_r)
                    x_index = int(clip(round(px)+ii, 0, num_rays - 1))
                    y_index = int(clip(round(py)+jj, 0, num_rays - 1))
                    qxy[x_index, y_index] += qt[int(ind),q]*kernel
                    
            ind = ind+1
            
    return qxy[k_r:-k_r, k_r: -k_r] 



def grid_rec_one_slice2(qt, theta_array, num_rays, k_r, kernel_type, xp, mode = "python"): #vectorized version of grid_rec_one_slice
    
    pad = k_r

    qxy = xp.zeros((num_rays + pad * 2 + 1, num_rays + pad * 2 + 1), dtype=xp.complex64)

    kernel_x = xp.stack([xp.stack([xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)), ] * theta_array.shape[0]) , ] * num_rays)  #this is 2D k_r*k_r size
    kernel_y = xp.stack([xp.stack([xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)).T, ] * theta_array.shape[0]) , ] * num_rays)
        
    px = xp.around(-(xp.array([range(num_rays), ] *  theta_array.shape[0]) - num_rays/2).T * (xp.sin(theta_array)) + (num_rays/2)).astype(int) + pad #adding k_r accounts for the padding
    py = xp.around((xp.array([range(num_rays), ] * theta_array.shape[0]) - num_rays/2).T * (xp.cos(theta_array)) + (num_rays/2)).astype(int) + pad #adding k_r accounts for the padding 

    kernel = K2(kernel_x + xp.reshape(px - xp.around(px), (px.shape[0], px.shape[1], 1, 1)), kernel_y + xp.reshape(py - xp.around(py), (py.shape[0], px.shape[1], 1, 1)), kernel_type, xp) * xp.reshape(qt.T, (qt.shape[1], qt.shape[0], 1, 1))

    qxy = Overlap(qxy, kernel, px, py, k_r, mode)


    return qxy[pad:-pad, pad: -pad] 

    
def grid_rec_one_slice_transpose(qxy, theta_array, num_rays, k_r, kernel_type): #use for forward proj
    
    qt = np.zeros((theta_array.shape[0], num_rays), dtype=np.complex64)
    padding_array = ((k_r, k_r), (k_r, k_r))
    qxy = np.lib.pad(qxy, padding_array, 'constant', constant_values=0)
    
    for q in range(num_rays):
        ind = 0
        
        for theta in theta_array:
            px = -(q - num_rays/2)*np.sin(theta)+(num_rays/2) + k_r
            py = (q - num_rays/2)*np.cos(theta)+(num_rays/2) + k_r    
            qti = 0

            for ii in range(-k_r, k_r+1):
                
                for jj in range(-k_r, k_r+1):
                    kernel = K2(px-round(px)-ii, py-round(py)-jj, kernel_type)
                    x_index = int(clip(round(px)+ii, 0, num_rays - 1))
                    y_index = int(clip(round(py)+jj, 0, num_rays - 1))

                    qti += qxy[x_index, y_index]*kernel; 
                    
            qt[int(ind),q] = qti
            
            ind = ind+1
            
    return qt


def grid_rec_one_slice_transpose2(qxy, theta_array, num_rays, k_r, kernel_type, xp, mode): #vectorized version of grid_rec_one_slice_transpose
    
    qt = xp.zeros((theta_array.shape[0] + k_r * 2 + 2, num_rays + k_r * 2 + 2), dtype = xp.complex64)

    padding_array = ((k_r + 1, k_r + 1), (k_r + 1, k_r + 1))
    qxy = xp.lib.pad(qxy, padding_array, 'constant', constant_values = 0)
    
    kernel_x = [[xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)), ] * theta_array.shape[0] , ] * num_rays  #this is 2D k_r*k_r size
    kernel_y = [[xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)).T, ] * theta_array.shape[0] , ] * num_rays

    px = xp.around(-(xp.array([range(num_rays), ] *  theta_array.shape[0]) - num_rays/2).T * (xp.sin(theta_array)) + (num_rays/2)).astype(int) + k_r + 1 #adding k_r accounts for the padding
    py = xp.around((xp.array([range(num_rays), ] * theta_array.shape[0]) - num_rays/2).T * (xp.cos(theta_array)) + (num_rays/2)).astype(int) + k_r + 1#adding k_r accounts for the padding

    kernel = K2(kernel_x + xp.reshape(px - xp.around(px), (px.shape[0], px.shape[1], 1, 1)), kernel_y + xp.reshape(py - xp.around(py), (py.shape[0], py.shape[1], 1, 1)), kernel_type)

    qt = Overlap_transpose(qt, qxy, kernel, px, py, k_r)

    return qt[k_r + 1 :-k_r - 1, k_r + 1: -k_r - 1]


def Overlap_transpose(image, frames_multiply, frames, coord_x, coord_y, k_r):

    for q in range(coord_x.shape[1]):
        
        for i in range(coord_x.shape[0]):
            image[q + 1: q + k_r*2 + 2, i + 1: i + k_r*2 + 2] += frames[i, q] * frames_multiply[coord_x[i, q] - k_r: coord_x[i, q] + k_r + 1, coord_y[i, q] - k_r: coord_y[i, q] + k_r + 1]
    
    return image



def Overlap(image, frames, coord_x, coord_y, k_r, mode = None):

    if mode is "python":
        return Overlap_CPU(image, frames, coord_x, coord_y, k_r)

    elif mode is "cuda":
        return Overlap_GPU(image, frames, coord_x, coord_y, k_r)

    else:
        return None


def Overlap_CPU(image, frames, coord_x, coord_y, k_r):

    for q in range(coord_x.shape[0]):
        
        for i in range(coord_x.shape[1]):
            image[coord_x[q, i] - k_r: coord_x[q, i] + k_r + 1, coord_y[q, i] - k_r: coord_y[q, i] + k_r + 1] += frames[q, i]

    return image



convolve_raw_kernel = None
if convolve_raw_kernel is None:
    with open("convolve.cu", 'r') as myfile:
        convolve_raw_kernel = myfile.read()

def Overlap_GPU(image, frames, coord_x, coord_y, k_r):

    n_angles = coord_x.shape[1]
    n_rays = coord_x.shape[0]

    img_x = image.shape[0]
    img_y = image.shape[1]

    nthreads = 128
    nblocks = ((n_rays * n_angles) / nthreads) + 1

    import cupy as cp


    image = image.astype(cp.complex64)
    frames = frames.astype(cp.complex64)

    coord_x = coord_x.astype(cp.int32)
    coord_y = coord_y.astype(cp.int32)


    cp.RawKernel(convolve_raw_kernel, "Convolve") \
        ((int(nblocks),), (int(nthreads),), \
        (image, frames, coord_x, coord_y, k_r, n_angles, n_rays, img_x, img_y))

    return image


def gridrec(sinogram_stack, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj
 
    tomo_stack = xp.zeros((sinogram_stack.shape[0], num_rays + 1, num_rays + 1),dtype=xp.complex64)
 
    ramlak_filter = xp.abs(xp.array(range(num_rays)) - num_rays/2)
    ramlak_filter = ramlak_filter.reshape((1, ramlak_filter.size))
    
    
    for i in range(sinogram_stack.shape[0]): #loops through each slice
        
        print("Reconstructing slice " + str(i))
         
        sinogram = xp.fft.fftshift(sinogram_stack[i],axes=1)
        
        sinogram = xp.fft.fft(sinogram,axis=1)  

        sinogram = xp.fft.fftshift(sinogram,axes=1)
        sinogram *= ramlak_filter

        tomogram = grid_rec_one_slice2(sinogram, theta_array, num_rays, k_r, kernel_type, xp, mode)

        tomo_stack[i] = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(tomogram)))
        
    return tomo_stack


def gridrec_transpose(tomo_stack, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for forward proj

    sinogram_stack = np.zeros((tomo_stack.shape[0], theta_array.shape[0], num_rays),dtype=np.complex64)

    ktilde = gaussian_kernel(np.array([range(-k_r, k_r+1),] * (k_r * 2 + 1)), sigma = k_r)
    ktilde *= ktilde.T

    print(ktilde)
    
    print(ktilde.shape)

    padding_array = ((num_rays//2 - k_r - 1, num_rays//2 - k_r), (num_rays//2 - k_r - 1, num_rays//2 - k_r))

    ktilde = np.lib.pad(ktilde, padding_array, 'constant', constant_values=0)
    
    deapodization_factor = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ktilde)))
    
    deapodization_factor = 1./deapodization_factor
  
    for i in range(tomo_stack.shape[0]): #loops through each slice
        
        tomo_slice = tomo_stack[i] * deapodization_factor
        # forward fft centered
        tomo_slice = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(tomo_slice)))

        sinogram = grid_rec_one_slice_transpose2(tomo_slice, theta_array, num_rays, k_r, kernel_type, xp, mode)
        print(sinogram.shape)
        
        sinogram = np.fft.ifftshift(sinogram,axes=1)
        
        sinogram = np.fft.ifft(sinogram,axis=1)        

        sinogram = np.fft.ifftshift(sinogram,axes=1)
        
        sinogram_stack[i]=sinogram
    
    return sinogram_stack

def force_data_types(input_data):

    input_data["data"]  = input_data["data"].astype(np.complex64)
    input_data["theta"] = input_data["theta"].astype(np.float64)	
    #input_data["rays"]  = input_data["rays"].astype(np.uint)

def memcopy_to_device(host_pointers):
    
    for key, value in host_pointers.items():
         
        host_pointers[key] = cp.asarray(value)
        
def memcopy_to_host(device_pointers):
    
    for key, value in device_pointers.items():
        
        device_pointers[key] = cp.asnumpy(value)

def tomo_reconstruct(data, theta, rays, k_r, kernel_type, algorithm, gpu_accelerated):
    input_data = {"data": data,
                  "theta": theta,
                  "rays": rays}
    mode = None    

    force_data_types(input_data)

    if gpu_accelerated == True:
        mode = "cuda"
        xp = __import__("cupy")
        memcopy_to_device(input_data)

    else:
        mode = "python"
        xp = __import__("numpy")
        
    if algorithm == "gridrec":
       output_stack = gridrec(input_data["data"], input_data["theta"], rays, k_r, kernel_type, xp, mode)    

    elif algorithm == "gridrec_transpose":
       output_stack = gridrec_transpose(input_data["data"], input_data["theta"], rays, k_r, kernel_type, xp, mode)

    output_data = {"result": output_stack}
    
    if gpu_accelerated == True:
        memcopy_to_host(output_data)

    return output_data["result"]

def sirt(A,At,b,end): #A is forward projection, b is the projections
    
    x = np.zeros((A.shape[1],1))
    rows = A.shape[0]
    cols = A.shape[1]
    C = np.zeros((cols,cols))
    for i in range(cols):
        C[i][i] = 1/(np.sum(A,axis=0)[i])
        
    R = np.zeros((rows,rows))
    
    for j in range(rows):
        R[j][j] = 1/(np.sum(A,axis=1)[i])

    CATR = C * At * R
    
    for i in range(1,end):
        x = x + CATR * (b - A * x)
        
    return x
