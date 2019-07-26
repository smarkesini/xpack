#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:14:36 2019

@author: anu
"""

import gridrec
import numpy as np
import matplotlib.pyplot as plt
#import h5py
#import dxchange
import sys
#from scipy import special
import time

start_time = time.time()

np.set_printoptions(threshold=sys.maxsize)

k_r = 2

base_folder = gridrec.create_unique_folder("shepp_logan")

size = 128

num_slices = size
num_angles = size
#num_angles = 512
num_rays   = size

true_obj_shape = (num_slices, num_rays, num_rays)

true_obj = gridrec.generate_Shepp_Logan(true_obj_shape)


pad_1D = 64
padding_array = ((0,0), (pad_1D, pad_1D), (pad_1D, pad_1D))

num_rays = num_rays + pad_1D*2

print("True obj shape", true_obj.shape)

gridrec.save_cube(true_obj, base_folder + "true")

#padding...
true_obj = np.lib.pad(true_obj, padding_array, 'constant', constant_values=0)

#angles in radians
theta = np.arange(0, 180., 180. / num_angles)*np.pi/180.

simulation = gridrec.forward_project(true_obj, theta)
print("simulation size", simulation.shape)

#The simulation generates a stack of projections, meaning, it changes the dimensions order
gridrec.save_cube(simulation, base_folder + "sim_project")

simulation = np.swapaxes(simulation,0,1)

gridrec.save_cube(simulation, base_folder + "sim_slice")

#Here I take only a subset of slices to not reconstruct the whole thing...
sub_slice = 15

#Calls forward projection
tomo_stack = gridrec.gridrec(simulation[64:65], theta, num_rays, k_r, 'kb')
plt.imshow(abs(tomo_stack[0]))
plt.show()

#Calls backward projection
sino_stack = gridrec.gridrec_transpose(true_obj[64:65], theta, num_rays, k_r, 'gaussian')
plt.imshow(abs(sino_stack[0]).real)
plt.show()

#tomopy_stack = tomopy.recon(simulation[64:65], theta, center=None, sinogram_order=True, algorithm="gridrec") #compare tomopy reconstruction to our reconstruction

if pad_1D > 0: tomo_stack = tomo_stack[:, pad_1D: -pad_1D, pad_1D: -pad_1D]

gridrec.save_cube(abs(tomo_stack), base_folder + "rec")
gridrec.save_cube(abs(sino_stack), base_folder + "sino")

print("A simulation of size", simulation.shape, "with kernel radius", k_r, "took", time.time() - start_time, "seconds to run")

