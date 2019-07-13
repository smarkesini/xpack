#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:25:06 2019

@author: anu
"""

#Gridrec algorithm based on pseudocode using Gaussian kernel
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
import h5py
import dxchange
import tomopy
from scipy import misc

#Read in file
#filename  = '20190524_085542_clay_testZMQ.h5'
#inputPath = '/Users/anu/Desktop/Summer/'
#file      = h5py.File(inputPath+filename, 'r')
#gdata     = dict(dxchange.reader._find_dataset_group(file).attrs)
print("Hi this is a test change")
stack_sino = np.random.rand(128,128,128)

#num_slices = int(gdata['nslices']) #number of slices
#num_angles = int(gdata['nangles']) #number of angles
#num_rays   = int(gdata['nrays'])
#Nr = len(r)

size = 64

num_slices = size
num_angles = size
num_rays   = size

print("The total number of slices is", num_slices, "and the total number of angles is", num_angles)

def K(x):
    sigma = 2
    kernel1 = np.exp(-1/2 * (x/sigma)**2)
    return kernel1

def KB2(x,y):
    kernel2 = np.outer(K(x),K(y))
    return kernel2

#STACK SINOGRAMS
#data = dxchange.read_als_832h5(filename)
#print("---- Data read -----")
#stack_sino = data[0]



#stack_sino = stack_sino.reshape([num_rays,num_angles,num_slices])
print("-----Sinogram reshaped -------")
qt = np.fft.fft(stack_sino, axis=2)
print("------ Ran FFT ------")
#qt = np.fft.fft(stack_sino, axis=1)

## Need to dig into fft shift to figure out how to set the center for the shift.
qt = np.fft.fftshift(qt, axes=2)
print("------ Shifted along second axis.--------")
import sys
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
   
    while True:
        id_index += 1
        folder = "Sim_" + dataset_name + "_id_" + str(id_index) + "/"
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

    return tomopy.sim.project.project(np.real(cube), theta, pad=False, sinogram_order=False)


@jit 
#def grid_rec_one_slice(qxy, qt):
def grid_rec_one_slice(qt, num_rays):

    qxy = np.zeros((num_rays, num_rays))

    for q in range(num_rays):
        for theta in range(num_angles):
            px = (q - num_rays/2)*np.cos(theta)+(num_rays/2)
            py = (q - num_rays/2)*np.sin(theta)+(num_rays/2)
            for ii in range(-k_r, k_r):
                for jj in range(-k_r, k_r):

                    gaussian_kernel = np.exp(-0.5*(px-round(px)-ii)**2/(2**2)-0.5*(py-round(py)-jj)**2/(2**2))

                    qxy[int(round(px)+ii-2), int(round(py)+jj-2)] += qt[theta,q]*gaussian_kernel

    return qxy                    


base_folder = create_unique_folder("shepp_logan")


true_obj_shape = (num_slices, num_angles, num_rays)

true_obj = generate_Shepp_Logan(original_data_shape)

save_cube(true_obj, base_folder + "true")

theta = np.arange(0, 180., 180. / num_angles)*np.pi/180.

simulation = forward_project(true_obj, theta)

save_cube(true_obj, base_folder + "sim")

sinogram = qt[0]

print("\n\n ---------- Input ---------- \n\n")
print(sinogram)

qxy = grid_rec_one_slice(sinogram, num_rays)


print("\n\n ---------- Finished one pass of grid rec ---------- \n\n")
result = np.fft.fftshift(np.fft.ifft2(qxy), axes=1)

print("\n\n ---------- Output ---------- \n\n")

print(result)




