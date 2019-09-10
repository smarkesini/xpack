import gridrec
import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os

def sirt(sino_data, theta_array, num_rays, k_r, kernel_type, gpu_accelerated, max_iter, factor):

    image = np.zeros((sino_data.shape[0], sino_data.shape[1], sino_data.shape[2]))
    
    if gpu_accelerated == False:
        mode = "python"
        xp = __import__("numpy")
        
    print("image", image.shape)
    

    for i in range(max_iter):
        
        print("------ Iteration", i, "------")
#        sino_sim = gridrec.forward_project(image, theta_array)
#        sino_sim = np.swapaxes(sino_sim,0,1) 
        
        radon,iradon,radont=gridrec.radon_setup(num_rays, theta, center=num_rays//2, xp=np, kernel_type = 'gaussian', k_r =1)

#       sino_sim = gridrec.tomo_reconstruct(image, theta_array, num_rays, k_r, kernel_type, "gridrec_transpose", gpu_accelerated)
        
        sino_sim = radon(image)
        
        residual = sino_data - sino_sim
        
#        step = gridrec.tomo_reconstruct(residual, theta_array, num_rays, k_r, kernel_type, "gridrec", gpu_accelerated)
        
        step = iradon(residual)
        step = step[:,:step.shape[1],:step.shape[2]]
        step *= np.abs(factor)
        
        image = image + step
        
        plt.imshow(abs(image[size//2]))
        plt.show()

#    return image[sino_data.shape[0]//2]
    return image

k_r = 1
size = 64
num_slices = size
num_angles = size*2
num_rays   = size

base_folder = gridrec.create_unique_folder("shepp_logan")

true_obj_shape = (num_slices, num_rays, num_rays)
true_obj = gridrec.generate_Shepp_Logan(true_obj_shape)

pad_1D        = size//2
padding_array = ((0, 0), (pad_1D, pad_1D), (pad_1D, pad_1D))
num_rays      = num_rays + pad_1D*2

print("True obj shape", true_obj.shape)

true_obj = np.lib.pad(true_obj, padding_array, 'constant', constant_values = 0)
theta    = np.arange(0, 180., 180. / num_angles)*np.pi/180.

simulation = gridrec.forward_project(true_obj, theta)
#simulation = np.swapaxes(simulation,0,1)

#xp              = __import__("numpy")
#mode            = 'python'
kernel_type     = "gaussian"
gpu_accelerated = False

radon,iradon,radont = gridrec.radon_setup(num_rays, theta, center=num_rays//2, xp=np, kernel_type = 'gaussian', k_r =1)

#recon = radon(true_obj)
#plt.imshow(np.abs(recon[size//2]))
#plt.show()
#
#backward = iradon(recon)
#plt.imshow(np.abs(backward[size//2]))
#plt.show()

image = sirt(simulation, theta, num_rays, k_r, kernel_type, gpu_accelerated, 3, factor=1)