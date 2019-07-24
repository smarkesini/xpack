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
import sys
from scipy import misc
#from scipy import special
import os
import time

start_time = time.time()


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

def gaussian_kernel(x, sigma=2): #using sigma=2 since this is the default value for sigma
    kernel1 = np.exp(-1/2 * (x/sigma)**2)
    return kernel1 
    
def keiser_bessel(x, k_r, beta):
#    kernel1 = np.i0(beta*np.sqrt(1-(2*x/(k_r))**2))/np.abs(np.i0(beta)) 
    kernel1 = (np.abs(x) <= k_r/2) * np.i0(beta*np.sqrt((np.abs(x) <= k_r/2)*(1-(2*x/(k_r))**2)))/np.abs(np.i0(beta)) 
    return kernel1

def K2(x, y, kernel, k_r=2, beta=1): #k_r and beta are parameters for keiser-bessel
    if kernel == 'gaussian':
        kernel2 = gaussian_kernel(x) * gaussian_kernel(y)
        return kernel2
    elif kernel == 'kb':
        kernel2 = keiser_bessel(x,k_r,beta) * keiser_bessel(y,k_r,beta)
        return kernel2
    else:
        print("Invalid kernel type")

#STACK SINOGRAMS
#data = dxchange.read_als_832h5(filename)
#print("---- Data read -----")
#stack_sino = data[0]

np.set_printoptions(threshold=sys.maxsize)

k_r = 3
    
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
        tname = base_fname + "_" + str(n) + '.tiff'
        misc.imsave(tname, cube[n,:,:])

def generate_Shepp_Logan(cube_shape):

   return tomopy.misc.phantom.shepp3d(size=cube_shape, dtype='float32')


def forward_project(cube, theta):

    return tomopy.sim.project.project(cube, theta, pad = False, sinogram_order=False)
    
def clip(a, lower, upper):

    return min(max(lower, a), upper)  

def grid_rec_one_slice(qt, theta_array, num_rays):
    print(theta_array.shape)
    qxy = np.zeros((num_rays, num_rays), dtype=np.complex128)
    for q in range(num_rays):
        ind = 0
        for theta in theta_array:
            px = -(q - num_rays/2)*np.sin(theta)+(num_rays/2)
            py = (q - num_rays/2)*np.cos(theta)+(num_rays/2)         
            for ii in range(-k_r, k_r):
                for jj in range(-k_r, k_r):
#                    kernel = K2(px-round(px)-ii, py-round(py)-jj, 'kb', k_r) #using keiser-bessel kernel
                    kernel = K2(px-round(px)-ii, py-round(py)-jj, 'gaussian') #using gaussian kernel
                    x_index = int(clip(round(px)+ii, 0, num_rays - 1))
                    y_index = int(clip(round(py)+jj, 0, num_rays - 1))
                    qxy[x_index, y_index] += qt[int(ind),q]*kernel
            ind = ind+1
    return qxy 

def Overlap(image, frames, coord_x, coord_y, k_r):

    for q in range(coord_x.shape[0]):
        for i in range(coord_x.shape[1]):
            image[coord_x[q, i] - k_r: coord_x[q, i] + k_r , coord_y[q, i] - k_r: coord_y[q, i] + k_r] += frames[q, i]

    return image


def grid_rec_one_slice2(qt, theta_array, num_rays): #vectorized version of grid_rec_one_slice

    qxy = np.zeros((num_rays + k_r * 2, num_rays + k_r * 2), dtype=np.complex128)

    kernel_x = [[np.array([range(- k_r, k_r), ] * k_r * 2), ] * theta_array.shape[0] , ] * num_rays  #this is 2D k_r*k_r size
    kernel_y = [[np.array([range(- k_r, k_r), ] * k_r * 2).T, ] * theta_array.shape[0] , ] * num_rays

    px = np.round(-(np.array([range(num_rays), ] *  theta_array.shape[0]) - num_rays/2).T * (np.sin(theta_array)) + (num_rays/2)).astype(int) + k_r #adding k_r accounts for the padding
    py = np.round((np.array([range(num_rays), ] * theta_array.shape[0]) - num_rays/2).T * (np.cos( theta_array)) + (num_rays/2)).astype(int) + k_r #adding k_r accounts for the padding

    kernel = K2(kernel_x + np.reshape(px - np.round(px), (px.shape[0], px.shape[1], 1, 1)), kernel_y + np.reshape(py - np.round(py), (py.shape[0], px.shape[1], 1, 1)), 'gaussian') * np.reshape(qt.T, (qt.shape[1], qt.shape[0], 1, 1))
#    kernel = K2(kernel_x + np.reshape(px - np.round(px), (px.shape[0], px.shape[1], 1, 1)), kernel_y + np.reshape(py - np.round(py), (py.shape[0], px.shape[1], 1, 1)), 'kb') * np.reshape(qt.T, (qt.shape[1], qt.shape[0], 1, 1))

    qxy = Overlap(qxy, kernel, px, py, k_r)

    return qxy[k_r:-k_r, k_r: -k_r] 

   
def gridrec(sinogram_stack, theta_array, num_rays):
    tomo_stack = np.zeros((sinogram_stack.shape[0], num_rays, num_rays),dtype=np.complex128)
    ramlak_filter = np.abs(np.array(range(num_rays)) - num_rays/2)
    ramlak_filter = ramlak_filter.reshape((1, ramlak_filter.size))

    
    for i in range(sinogram_stack.shape[0]): #loops through each slice
        
        
        print("Reconstructing slice " + str(i))
        
        sinogram = np.fft.fftshift(sinogram_stack[i],axes=1)
        
        sinogram = np.fft.fft(sinogram,axis=1)        

        sinogram = np.fft.fftshift(sinogram,axes=1)
        sinogram *= ramlak_filter

        tomogram = grid_rec_one_slice2(sinogram, theta_array, num_rays)

        print(tomogram.shape)

        tomo_stack[i] = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(tomogram)))

    return tomo_stack

base_folder = create_unique_folder("shepp_logan")

size = 128

num_slices = size
num_angles = 512
num_rays   = size

true_obj_shape = (num_slices, num_rays, num_rays)

true_obj = generate_Shepp_Logan(true_obj_shape)


pad_1D = 128
padding_array = ((0,0), (pad_1D, pad_1D), (pad_1D, pad_1D))

num_rays = num_rays + pad_1D*2

print("True obj shape", true_obj.shape)

save_cube(true_obj, base_folder + "true")

#padding...
true_obj = np.lib.pad(true_obj, padding_array, 'constant', constant_values=0)

#angles in radians
theta = np.arange(0, 180., 180. / num_angles)*np.pi/180.

simulation = forward_project(true_obj, theta)
print("simulation size", simulation.shape)

#The simulation generates a stack of projections, meaning, it changes the dimensions order
save_cube(simulation, base_folder + "sim_project")

simulation = np.swapaxes(simulation,0,1)

save_cube(simulation, base_folder + "sim_slice")

#Here I take only a subset of slices to not reconstruct the whole thing...
sub_slice = 15

tomo_stack = gridrec(simulation[64:65], theta, num_rays)

#tomopy_stack = tomopy.recon(simulation[64:65], theta, center=None, sinogram_order=True, algorithm="gridrec") #compare tomopy reconstruction to our reconstruction

tomo_stack = tomo_stack[:, pad_1D: -pad_1D, pad_1D: -pad_1D]

save_cube(abs(tomo_stack), base_folder + "rec")

print("A simulation of size", simulation.shape, "with kernel radius", k_r, "took", time.time() - start_time, "seconds to run")




