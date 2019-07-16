#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:25:06 2019

@author: anu
"""

#Gridrec algorithm based on pseudocode using Gaussian kernel
#from numba import jit
#import matplotlib.pyplot as plt
import numpy as np
#import h5py
#import dxchange
import tomopy
import sys
from scipy import misc
#from scipy import special
import os

#Read in file
#filename  = '20190524_085542_clay_testZMQ.h5'
#inputPath = '/Users/anu/Desktop/Summer/'
#file      = h5py.File(inputPath+filename, 'r')
#gdata     = dict(dxchange.reader._find_dataset_group(file).attrs)
print("Hi this is a test change")
#stack_sino = np.random.rand(128,128,128)

#num_slices = int(gdata['nslices']) #number of slices
#num_angles = int(gdata['nangles']) #number of angles
#num_rays   = int(gdata['nrays'])
#Nr = len(r)

size = 64

num_slices = size
num_angles = size
num_rays   = size

#print("The total number of slices is", num_slices, "and the total number of angles is", num_angles)

def gaussian_kernel(x, sigma=2): #using sigma=2 since this is the default value for sigma
    kernel1 = np.exp(-1/2 * (x/sigma)**2)
    return kernel1 
    

def keiser_bessel(x, k_r, beta):
    if np.abs(x) <= k_r:
        kernel1 = np.i0(beta*np.sqrt(1-(2*x/(k_r-1))**2))/np.abs(np.i0(beta)) 
    elif np.abs(x) > k_r:
        kernel1 = 0
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



#stack_sino = stack_sino.reshape([num_rays,num_angles,num_slices])
#print("-----Sinogram reshaped -------")
#qt = np.fft.fft(stack_sino, axis=2)
#print("------ Ran FFT ------")
#qt = np.fft.fft(stack_sino, axis=1)

## Need to dig into fft shift to figure out how to set the center for the shift.
#qt = np.fft.fftshift(qt, axes=2)
#print("------ Shifted along second axis.--------")

np.set_printoptions(threshold=sys.maxsize)


#Create the Ram-Lak filter and apply it to the fft (qt). From pseudo code this is optional.
# First cut I would ignoreit.
# vec    = (np.linspace(0,r,r))*(-r/2)
# ramlak = np.abs(vec)
# qt     = np.multiply(qt, ramlak)
# print(qt)

# Next initialize the matrix of all zeros that will be filled
#qxy = np.zeros((num_rays, num_rays))

# Convolution for one slice
k_r = 2
    
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

    return tomopy.sim.project.project(cube, theta, pad=False, sinogram_order=False)


def clip(a, lower, upper):

    return min(max(lower, a), upper)  

def grid_rec_one_slice(qt, theta_array, num_rays):

    qxy = np.zeros((num_rays, num_rays))

    for q in range(num_rays):
        for theta in theta_array:
            px = (q - num_rays/2)*np.cos(theta)+(num_rays/2)
            py = (q - num_rays/2)*np.sin(theta)+(num_rays/2)
            for ii in range(-k_r, k_r):
                for jj in range(-k_r, k_r):

#                    gaussian_kernel = np.exp(-0.5*(px-round(px)-ii)**2/(2**2)-0.5*(py-round(py)-jj)**2/(2**2))
#                    kernel = K2(px-round(px)-ii, py-round(py)-jj, 'kb', k_r) #using keiser-bessel kernel
                    kernel = K2(px-round(px)-ii, py-round(py)-jj, 'gaussian') #using gaussian kernel
#                    print(kernel)
                    x_index = int(clip(round(px)+ii, 0, num_rays - 1))
                    y_index = int(clip(round(py)+jj, 0, num_rays - 1))

                    qxy[x_index, y_index] += qt[int(theta),q]*kernel

    return qxy    


def gridrec(sinogram_stack, theta_array, num_rays):

    tomo_stack = np.zeros((sinogram_stack.shape[0], num_rays, num_rays))
#    sinogram = np.fft.fft(sinogram_stack, axis=1)
#    sinogram = np.fft.fftshift(sinogram, axes=1)
#
    ramlak_filter = np.abs(np.array(range(num_rays)) - num_rays/2)
    ramlak_filter = ramlak_filter.reshape((1, ramlak_filter.size))
#    
#    sinogram *= ramlak_filter

    print("Ram-Lak filter values:", ramlak_filter)

    for i in range(sinogram_stack.shape[0]): #loops through each slice

        print("Reconstructing slice " + str(i))

        sinogram = np.fft.fft(sinogram_stack[i], axis=1)
        print("the shape of sinogram is",sinogram.shape)
        sinogram = np.fft.fftshift(sinogram, axes=1)

        sinogram *= ramlak_filter

        tomogram = grid_rec_one_slice(sinogram, theta_array, num_rays)
#        tomogram = grid_rec_one_slice(sinogram[i], theta_array, num_rays)
        tomo_stack[i] = np.fft.fftshift(np.fft.ifft2(tomogram)) 

    return tomo_stack


base_folder = create_unique_folder("shepp_logan")


true_obj_shape = (num_slices, num_angles, num_rays)

true_obj = generate_Shepp_Logan(true_obj_shape)

save_cube(true_obj, base_folder + "true")

theta = np.arange(0, 180., 180. / num_angles)*np.pi/180.
print(theta)

simulation = forward_project(true_obj, theta)

#we put these back in degrees...
theta = theta*180./np.pi
print(theta)

#I am not sure about this now. Do we normalize the degrees to fit our cube size? I am doing that now...
theta = theta*(num_angles/180.)
print(theta)

#The simulation generates a stack of projections, meaning, it changes the dimensions order
save_cube(simulation, base_folder + "sim_project")

simulation = np.swapaxes(simulation,0,1)

save_cube(simulation, base_folder + "sim_slice")

#Here I take only a subset of slices to not reconstruct the whole thing...
sub_slice = 15

tomo_stack = gridrec(simulation[0: sub_slice], theta, num_rays)
#tomo_stack = gridrec(simulation, theta, num_rays)

#tomo_stack = tomopy.recon(simulation, theta, center=None, sinogram_order=True, algorithm="gridrec")
#print(tomo_stack)

save_cube(tomo_stack, base_folder + "rec")




