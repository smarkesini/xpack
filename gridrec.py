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

num_slices = 128
num_angles = 128
num_rays   = 128

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


@jit 
#def grid_rec_one_slice(qxy, qt):
def grid_rec_one_slice(num_rays,qt):

    qxy = np.zeros((num_rays, num_rays))
#    num_rays, _ = qxy.shape
    for q in range(num_rays):
        for theta in range(num_angles):
            px = (q - num_rays/2)*np.cos(theta)+(num_rays/2)
            py = (q - num_rays/2)*np.sin(theta)+(num_rays/2)
            for ii in range(-k_r, k_r):
                for jj in range(-k_r, k_r):
                    ## qt[q,theta,0] to get just the first slice.
                    gaussian_kernel = np.exp(-0.5*(px-round(px)-ii)**2/(2**2)-0.5*(py-round(py)-jj)**2/(2**2))
                    #print(gaussian_kernel)
                    #print(int(round(px)+ii-2), int(round(py)+jj-2))
                    #print(qt[0,theta,q])
                    qxy[int(round(px)+ii-2), int(round(py)+jj-2)] += qt[0,theta,q]*gaussian_kernel
                    #print(qxy[int(round(px)+ii-2), int(round(py)+jj-2)])
#                    print(int(round(px)+ii-2), int(round(py)+jj-2))
    return qxy                    
#print("----- Finished convolution for one slice -----")
#print(qxy)
#print(qt)
#qxy = grid_rec_one_slice(qxy,qt)
qxy = grid_rec_one_slice(num_rays,qt)

print(qxy)
print("----- Finished one pass of grid rec ------")
result = np.fft.fftshift(np.fft.ifft2(qxy), axes=1)
print(result.shape)
#print(result)