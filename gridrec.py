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
from timeit import default_timer as timer
import scipy
#import cupy as cp

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

def gaussian_kernel(x, k_r, sigma, xp): #using sigma=2 since this is the default value for sigma
#    kernel1 = xp.exp(-1/2 * (x/sigma)**2 *16)
    kernel1 = xp.exp(-1/2 * 8 * (x/sigma)**2 )
    kernel1 = kernel1* (xp.abs(x) <= k_r)
    return kernel1 
    
def keiser_bessel(x, k_r, beta,xp):
#    kernel1 = np.i0(beta*np.sqrt(1-(2*x/(k_r))**2))/np.abs(np.i0(beta)) 
    kernel1 = (xp.abs(x) <= k_r/2) * xp.i0(beta*xp.sqrt((xp.abs(x) <= k_r/2)*(1-(2*x/(k_r))**2)))/xp.abs(xp.i0(beta)) 
    return kernel1

def K1(x, k_r, kernel_type, xp,  beta=1, sigma=2):
    if kernel_type == 'gaussian':
        kernel2 = gaussian_kernel(x,k_r,sigma,xp) 
        return kernel2
    elif kernel_type == 'kb':
        kernel2 = keiser_bessel(x,k_r,beta,xp) 
        return kernel2
    else:
        print("Invalid kernel type")
        
    
def K2(x, y, kernel_type, xp, k_r=2, beta=1, sigma=2): #k_r and beta are parameters for keiser-bessel
    if kernel_type == 'gaussian':
        kernel2 = gaussian_kernel(x,k_r,sigma,xp) * gaussian_kernel(y,k_r,sigma,xp)
        return kernel2
    elif kernel_type == 'kb':
        kernel2 = keiser_bessel(x,k_r,beta,xp) * keiser_bessel(y,k_r,beta,xp)
        return kernel2
    else:
        print("Invalid kernel type")

def deapodization(num_rays,kernel_type,xp,k_r=2, beta=1, sigma=2):
    #stencil=np.array([range(-k_r, k_r+1),]
    sampling=2
    step=1./sampling
    stencil=np.array([np.arange(-k_r, k_r+1-step*(sampling-1),step),])
   
    ktilde=K1(stencil,k_r, kernel_type, xp,  beta, sigma)
    
    padding_array = ((0,0),(num_rays*sampling//2 - k_r*sampling , num_rays*sampling//2 - k_r*sampling-1))
    ktilde1=np.lib.pad(ktilde, padding_array, 'constant', constant_values=0)
    ktilde1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ktilde1)))
    #print("ratio i/r=", np.max(abs(deapodization_factor.imag))/np.max(abs(deapodization_factor.real)))

    ktilde2=(ktilde1[:,num_rays*(sampling)//2-num_rays//2:num_rays*(sampling)//2+num_rays//2]).real
    
    apodization_factor = ktilde2 * ktilde2.T
    
    return 1./apodization_factor

def deapodization_shifted(num_rays,kernel_type,xp,k_r=2, beta=1, sigma=2):
    #stencil=np.array([range(-k_r, k_r+1),]
    sampling=2
    step=1./sampling
    stencil=np.array([np.arange(-k_r, k_r+1-step*(sampling-1),step),])
   
    ktilde=K1(stencil,k_r, kernel_type, xp,  beta, sigma)
    
    padding_array = ((0,0),(num_rays*sampling//2 - k_r*sampling , num_rays*sampling//2 - k_r*sampling-1))
    ktilde1=np.lib.pad(ktilde, padding_array, 'constant', constant_values=0)
    # ktilde1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ktilde1)))
    ktilde1 = np.fft.fftshift(np.fft.ifft(ktilde1))
    #print("ratio i/r=", np.max(abs(deapodization_factor.imag))/np.max(abs(deapodization_factor.real)))

    ktilde2=(ktilde1[:,num_rays*(sampling)//2-num_rays//2:num_rays*(sampling)//2+num_rays//2]).real
    
    apodization_factor = ktilde2 * ktilde2.T
    
    return 1./apodization_factor


def deapodization_simple(num_rays,kernel_type,xp,k_r=2, beta=1, sigma=2):
    ktilde=K1(np.array([range(-k_r, k_r+1),] ), k_r, kernel_type, xp,  beta, sigma)
    padding_array = ((0,0),(num_rays//2 - k_r , num_rays//2 - k_r-1))
    ktilde = np.lib.pad(ktilde, padding_array, 'constant', constant_values=0)
    ktilde = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ktilde)))

    deapodization_factor = ktilde * ktilde.T 
#    print("ratio i/r=", np.max(abs(deapodization_factor.imag))/np.max(abs(deapodization_factor.real)))
 
    return 1./deapodization_factor.real


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
        tname = base_fname + "_" + str(n) + '.jpeg'
        imageio.imwrite(tname, cube[n,:,:].astype(np.float32))
        

def generate_Shepp_Logan(cube_shape):

   return tomopy.misc.phantom.shepp3d(size=cube_shape, dtype='float32')


def forward_project(cube, theta):

    #return tomopy.sim.project.project(cube, theta, pad = False, sinogram_order=False)
    return tomopy.sim.project.project(cube, theta, pad = False, sinogram_order=True)

    
def clip(a, lower, upper):

    return min(max(lower, a), upper)  


def grid_rec_one_slice(qt, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj
    
    qxy = np.zeros((num_rays + k_r * 2 + 1, num_rays + k_r * 2 + 1), dtype=np.complex64)
    #qxy = np.zeros((num_rays + k_r * 2 , num_rays + k_r * 2), dtype=np.complex64)

    stencil=np.array(range(-k_r, k_r+1))
    
    start = timer()

    for q in range(num_rays):
        ind = 0
        
        for theta in theta_array:
            px = -(q - num_rays/2)*np.sin(theta)+(num_rays/2)+k_r
            py = (q - num_rays/2)*np.cos(theta)+(num_rays/2)+k_r  
            #ky= gaussian_kernel(py-round(py)-np.arange(-k_r,k_r+1), k_r, xp);
            
            ky=K1(round(py)-py+stencil,k_r, kernel_type, xp)
             
            
            for ii in range(-k_r, k_r+1):
                #kx= gaussian_kernel(px-round(px)-ii, k_r, xp);
                kx=K1(round(px)-px+ii,k_r, kernel_type, xp)

                for jj in range(-k_r, k_r+1):
                    #kernel = K2(px-round(px)-ii, py-round(py)-jj, kernel_type, xp, k_r)
                    kernel = kx * ky[jj+k_r]
                    #x_index = int(clip(round(px)+ii, 0, num_rays - 1))
                    #y_index = int(clip(round(py)+jj, 0, num_rays - 1))
                    x_index = int(round(px)+ii)
                    y_index = int(round(py)+jj)
                    
                    qxy[x_index, y_index] += qt[int(ind),q]*kernel
                    
            ind = ind+1
    
    end = timer()
    print("time 4 loop=",end - start)

        
    return qxy[k_r:-k_r-1, k_r: -k_r-1] 


def grid_rec_one_slice0(qt, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj
    
    start = timer()

    #qxy = np.zeros((num_rays + k_r * 2 + 1, num_rays + k_r * 2 + 1), dtype=np.complex64)
    qxy = np.zeros((num_rays + k_r * 2 + 1, num_rays + k_r * 2 + 1), dtype=np.complex64)
    #qxy = np.zeros((num_rays + k_r * 2 , num_rays + k_r * 2), dtype=np.complex64)
    #print("shape",(qxy[k_r:-k_r-1, k_r: -k_r-1]).shape)
    #print("shape qxy",(qxy).shape)
    
    stencil=np.array([range(-k_r, k_r+1),]);

    end = timer()
    print("regridding loop setup time=",end - start) 
    start = timer()
    for q in range(num_rays):
        ind = 0
        
        for theta in theta_array:
            px = -(q - num_rays/2)*np.sin(theta)+(num_rays/2)+k_r
            py = (q - num_rays/2)*np.cos(theta)+(num_rays/2)+k_r  
            #ky= gaussian_kernel(round(py)-py+stencil, k_r, xp);
            ky=K1(round(py)-py+stencil,k_r, kernel_type, xp)
            #kx= gaussian_kernel(round(px)-px+stencil, k_r, xp);
            kx=K1(round(px)-px+stencil,k_r, kernel_type, xp)
            
            kernel=qt[int(ind),q]*ky*kx.T
            
            qxy[int(round(px))+stencil.T,int(round(py))+stencil]+=kernel                    
            ind = ind+1
    #return qxy[k_r+1:-k_r, k_r+1: -k_r]
    end = timer()
    print("regridding loop time=",end - start)         
    return qxy[k_r:-k_r-1, k_r: -k_r-1] 


def grid_rec_one_sliceSpMV(qt, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj
    
    # setting up sparse array
    
    start = timer()
    num_angles=theta_array.shape[0]
    
    stencil=np.array([range(-k_r, k_r+1),])
    stencilx=np.reshape(stencil,[1,1,2*k_r+1,1])
    stencily=np.reshape(stencil,[1,1,1,2*k_r+1])
    
    # coordinates of the points on the  grid
    px = - (xp.sin(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
    py =   (xp.cos(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
    
    # input index
    pind = xp.array(range(num_rays*num_angles))
    
    # reshape coordinates and index
    px=np.reshape(px,[num_angles,num_rays,1,1])
    py=np.reshape(py,[num_angles,num_rays,1,1])
    pind=np.reshape(pind,[num_angles,num_rays,1,1])
    
    # compute kernel
    kx=K1(stencilx - (px - xp.around(px)),k_r, kernel_type, xp); 
    ky=K1(stencily - (py - xp.around(py)),k_r, kernel_type, xp); 
    Kval=(kx*ky)

    # expand coordinates with stencils
    kx=stencilx+xp.around(px)
    ky=stencily+xp.around(py)

    # row  index (where the output goes)
    Krow=((kx)*(num_rays+2*k_r+1)+ky)
    # column index (where to get the input)
    Kcol=(pind+kx*0+ky*0)
    
    # create sparse array
    S=scipy.sparse.csr_matrix((Kval.flatten(), (Krow.flatten(), Kcol.flatten())), shape=((num_rays+2*k_r+1)**2, num_angles*num_rays))

    end = timer()
    
    print("Spmv setup time=",end - start)

    #compute output
    start = timer()
    

    # do the SpMV and reshape
    qxy=np.reshape(S*(qt).flatten(),(num_rays+2*k_r+1,num_rays+2*k_r+1))
    
    end = timer()
    
    print("Spmv time=",end - start)
        
    return qxy[k_r:-k_r-1, k_r: -k_r-1] 



def grid_rec_one_slice2(qt, theta_array, num_rays, k_r, kernel_type, xp, mode = "python"): #vectorized version of grid_rec_one_slice


    start = timer()
    
    pad = k_r

    qxy = xp.zeros((num_rays + pad * 2 + 1, num_rays + pad * 2 + 1), dtype=xp.complex64)

    kernel_x = xp.stack([xp.stack([xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)), ] * theta_array.shape[0]) , ] * num_rays)  #this is 2D k_r*k_r size
    kernel_y = xp.stack([xp.stack([xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)).T, ] * theta_array.shape[0]) , ] * num_rays)
        
    
    px = ((xp.array([range(num_rays), ] *  theta_array.shape[0]) - num_rays/2).T * (xp.sin(theta_array)) + (num_rays//2)) + pad #adding k_r accounts for the padding
    py = ((xp.array([range(num_rays), ] * theta_array.shape[0]) - num_rays/2).T * (xp.cos(theta_array)) + (num_rays//2)) + pad #adding k_r accounts for the padding 

    kernel = K2(kernel_x - xp.reshape(px - xp.around(px), (px.shape[0], px.shape[1], 1, 1)), kernel_y - xp.reshape(py - xp.around(py), (py.shape[0], px.shape[1], 1, 1)), kernel_type, xp) * xp.reshape(qt.T, (qt.shape[1], qt.shape[0], 1, 1))

    mode="python"

    end = timer()
    print("regridding 0 loop setup time=",end - start)

    start = timer()
    
    qxy = Overlap(qxy, kernel, xp.around(px).astype(int), xp.around(py).astype(int), k_r, mode)

    end = timer()
    print("regridding 0 loop=",end - start)
    
    return qxy[pad:-pad-1, pad: -pad-1] 

def gridding_setup(num_rays, theta_array, xp=np, kernel_type = 'gaussian', k_r = 2):
    
    num_angles=theta_array.shape[0]
    
    stencil=xp.array([range(-k_r, k_r+1),])
    stencilx=xp.reshape(stencil,[1,1,2*k_r+1,1])
    stencily=xp.reshape(stencil,[1,1,1,2*k_r+1])

    # coordinates of the points on the  grid
    px = - (xp.sin(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
    py =   (xp.cos(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
    
    # output index
    pind = xp.array(range(num_rays*num_angles))
    
    # reshape coordinates and index
    px=xp.reshape(px,[num_angles,num_rays,1,1])
    py=xp.reshape(py,[num_angles,num_rays,1,1])
    pind=xp.reshape(pind,[num_angles,num_rays,1,1])
    
    # compute kernels
    kx=K1(stencilx - (px - xp.around(px)),k_r, kernel_type, xp); 
    ky=K1(stencily - (py - xp.around(py)),k_r, kernel_type, xp); 
    Kval=(kx*ky)
    
    
    # expand coordinates with stencils
    kx=stencilx+xp.around(px)
    ky=stencily+xp.around(py)

  # row  index (where the output goes)
    Krow=((kx)*(num_rays+2*k_r+1)+ky)
    # column index (where to get the input)
    Kcol=(pind+kx*0+ky*0)
    
    # create sparse array   
    S=scipy.sparse.csr_matrix((Kval.flatten(), (Krow.flatten(), Kcol.flatten())), shape=((num_rays+2*k_r+1)**2, num_angles*num_rays))

    # note that we swap column and rows to get the transpose
    ST=scipy.sparse.csr_matrix((Kval.flatten(),(Kcol.flatten(), Krow.flatten())), shape=(num_angles*num_rays, (num_rays+2*k_r+1)**2))
    
       
    return S, ST

def radon_setup(num_rays, theta_array, xp=np, kernel_type = 'gaussian', k_r = 2):

    S, ST = gridding_setup(num_rays, theta_array, xp, kernel_type , k_r)
    

    
    #deapodization_factor = deapodization(num_rays, kernel_type, xp, k_r)
    deapodization_factor = deapodization_shifted(num_rays, kernel_type, xp, k_r)
    deapodization_factor=xp.reshape(deapodization_factor,(1,num_rays,num_rays))
    
    # get the filter
    num_angles=theta_array.shape[0]
    print("num_angles", num_angles)
    ramlak_filter = (xp.abs(xp.array(range(num_rays)) - num_rays/2)+1./num_rays)/(num_rays**2)/num_angles/9.8
    
    # this is to avoid one fftshift
    ramlak_filter*=(-1)**np.arange(num_rays);
    
    
    # removing the highest frequency
    ramlak_filter[0]=0;

    #ramlak_filter = ramlak_filter.reshape((1, ramlak_filter.size))
    ramlak_filter = ramlak_filter.reshape((1, 1, ramlak_filter.size))
    
    
            
    
    R  = lambda tomo:  radon(tomo, deapodization_factor/num_rays/3.5*1.046684, ST, k_r, num_angles )
    
    
    #IR = lambda sino: iradon(sino, deapodization_factor, S,  k_r, ramlak_filter )
    IR = lambda sino: iradon(sino, deapodization_factor, S,  k_r, ramlak_filter )
    #iradon(sinogram_stack, deapodization_factor, S, k_r , hfilter): 
   
    return R,IR
    

    

def grid_rec_one_slice_transposeSpMV(qxy, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj
    
    #qt = np.zeros((theta_array.shape[0], num_rays), dtype=np.complex64)
    
    
    padding_array = ((k_r, k_r+1), (k_r, k_r+1))
    qxy = np.lib.pad(qxy, padding_array, 'constant', constant_values=0)

    # setting up sparse array
    num_angles=theta_array.shape[0]
    
    stencil=np.array([range(-k_r, k_r+1),])
    stencilx=np.reshape(stencil,[1,1,2*k_r+1,1])
    stencily=np.reshape(stencil,[1,1,1,2*k_r+1])

    # coordinates of the points on the  grid
    px = - (xp.sin(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
    py =   (xp.cos(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
    
    # output index
    pind = xp.array(range(num_rays*num_angles))
    
    # reshape coordinates and index
    px=np.reshape(px,[num_angles,num_rays,1,1])
    py=np.reshape(py,[num_angles,num_rays,1,1])
    pind=np.reshape(pind,[num_angles,num_rays,1,1])
    
    # compute kernels
    kx=K1(stencilx - (px - xp.around(px)),k_r, kernel_type, xp); 
    ky=K1(stencily - (py - xp.around(py)),k_r, kernel_type, xp); 
    Kval=(kx*ky)
    
    
    # expand coordinates with stencils
    kx=stencilx+xp.around(px)
    ky=stencily+xp.around(py)

  # row  index (where the output goes)
    Krow=((kx)*(num_rays+2*k_r+1)+ky)
    # column index (where to get the input)
    Kcol=(pind+kx*0+ky*0)
    
    # create sparse array   
    #S=scipy.sparse.csr_matrix((Kval.flatten(), (Krow.flatten(), Kcol.flatten())), shape=((num_rays+2*k_r+1)**2, num_angles*num_rays))
    # note that we swap column and rows to get the transpose
    ST=scipy.sparse.csr_matrix((Kval.flatten(),(Kcol.flatten(), Krow.flatten())), shape=(num_angles*num_rays, (num_rays+2*k_r+1)**2))
    
    # do the SpMV and reshape
    qt=np.reshape(ST*(qxy).flatten(),(num_angles,num_rays))

    return qt


    
def grid_rec_one_slice_transpose(qxy, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for forward proj
    
    qt = np.zeros((theta_array.shape[0], num_rays), dtype=np.complex64)
#    padding_array = ((k_r, k_r), (k_r, k_r))
#    qxy = np.lib.pad(qxy, padding_array, 'constant', constant_values=0)
    
    for q in range(num_rays):
        ind = 0
        
        for theta in theta_array:
            px = -(q - num_rays/2)*np.sin(theta)+(num_rays/2) # + k_r 
            py = (q - num_rays/2)*np.cos(theta)+(num_rays/2) # + k_r
            
#            This is what we had after the meeting
#            px = -(q  - (num_rays)/2)*np.sin(theta)+((num_rays+4+2*k_r)/2)
#            py = (q  - (num_rays)/2)*np.cos(theta)+((num_rays+4+2*k_r)/2)
            
            qti = 0

            for ii in range(-k_r, k_r+1):
                
                for jj in range(-k_r, k_r+1):
                    kernel = K2(px-round(px)-ii, py-round(py)-jj, kernel_type, xp)
                    x_index = int(round(px) +ii)
                    y_index = int(round(py) +jj)
#                    if x_index>=0 & x_index<num_rays-, y_index --> do this instead of clipping
#                    if (x_index >= 0 and x_index <= num_rays+2*k_r-1) and (y_index >= 0 and y_index <= num_rays+2*k_r-1):
                    if (x_index >= 0 and x_index < num_rays) and (y_index >= 0 and y_index < num_rays):

                        qti += qxy[x_index, y_index]*kernel
                    else:
                        qti += 0
                        
            qt[int(ind),q] = qti
            
            ind = ind+1
            
            
    return qt

def grid_rec_one_slice_transpose0(qxy, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj
    
    qt = np.zeros((theta_array.shape[0], num_rays), dtype=np.complex64)
    
    padding_array = ((k_r, k_r+1), (k_r, k_r+1))
    qxy = np.lib.pad(qxy, padding_array, 'constant', constant_values=0)

    #print("shape",(qxy[k_r:-k_r-1, k_r: -k_r-1]).shape)
    stencil=np.array([range(-k_r, k_r+1),]);

    for q in range(num_rays):
        ind = 0
        
        for theta in theta_array:
            px = -(q - num_rays/2)*np.sin(theta)+(num_rays/2)+k_r
            py = (q - num_rays/2)*np.cos(theta)+(num_rays/2)+k_r  
            #ky= gaussian_kernel(round(py)-py+stencil, k_r, xp);
            ky= K1(round(py)-py+stencil, k_r, kernel_type, xp)
            #kx= gaussian_kernel(round(px)-px+stencil, k_r, xp);
            kx= K1(round(px)-px+stencil, k_r, kernel_type, xp)
            
            kernel=ky*kx.T
            
            qt[int(ind),q]=np.sum(qxy[int(round(px))+stencil.T,int(round(py))+stencil]*kernel)
                                
            ind = ind+1
    #return qxy[k_r+1:-k_r, k_r+1: -k_r]         
    return qt



def grid_rec_one_slice_transpose2(qxy, theta_array, num_rays, k_r, kernel_type, xp, mode): #vectorized version of grid_rec_one_slice_transpose
    
    qt = xp.zeros((theta_array.shape[0] + k_r * 2 +1, num_rays + k_r * 2 + 1), dtype = xp.complex64)

    padding_array = ((k_r + 1, k_r + 1), (k_r + 1, k_r + 1))
    qxy = xp.lib.pad(qxy, padding_array, 'constant', constant_values = 0)
    
    kernel_x = [[xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)), ] * theta_array.shape[0] , ] * num_rays  #this is 2D k_r*k_r size
    kernel_y = [[xp.array([range(- k_r, k_r + 1), ] * (k_r * 2 + 1)).T, ] * theta_array.shape[0] , ] * num_rays

    px = xp.around(-(xp.array([range(num_rays), ] *  theta_array.shape[0]) - num_rays/2).T * (xp.sin(theta_array)) + (num_rays/2)).astype(int) + k_r +1 #adding k_r accounts for the padding
    py = xp.around((xp.array([range(num_rays), ] * theta_array.shape[0]) - num_rays/2).T * (xp.cos(theta_array)) + (num_rays/2)).astype(int) + k_r +1#adding k_r accounts for the padding

    kernel = K2(kernel_x + xp.reshape(px - xp.around(px), (px.shape[0], px.shape[1], 1, 1)), kernel_y + xp.reshape(py - xp.around(py), (py.shape[0], py.shape[1], 1, 1)), kernel_type,xp)

    qt = Overlap_transpose(qt, qxy, kernel, px, py, k_r)

    return qt[k_r  +1:-k_r - 1, k_r + 1: -k_r - 1]


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
        print("mode is wrong")
        return None


def Overlap_CPU(image, frames, coord_x, coord_y, k_r):

    for q in range(coord_x.shape[0]):
        
        for i in range(coord_x.shape[1]):
            #image[coord_x[q, i] - k_r: coord_x[q, i] + k_r + 1, coord_y[q, i] - k_r: coord_y[q, i] + k_r + 1] += frames[q, i]
            image[coord_x[q, i] - k_r: coord_x[q, i] + k_r + 1, coord_y[q, i] - k_r: coord_y[q, i] + k_r + 1] += frames[q, i]
            #image[coord_x[q, i] : coord_x[q, i] + 2*k_r+1 , coord_y[q, i] - k_r: coord_y[q, i] + k_r + 1] += frames[q, i]

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

def iradon(sinogram_stack, deapodization_factor, S, k_r , hfilter): 
              #iradon(sino, deapodization_factor, S,  k_r, ramlak_filter )
    xp=np
    
    num_slices = sinogram_stack.shape[0]
    num_angles = sinogram_stack.shape[1]
    num_rays   = sinogram_stack.shape[2]

    # we can avoid the initial fftshift by modifying the input filter
    #sinogram_stack = xp.fft.fftshift(sinogram_stack,axes=2)
    
#    start = timer()
#    
#    sinogram_stack = xp.fft.fft(sinogram_stack,axis=2)
#    #end = timer()
#
#    sinogram_stack = xp.fft.fftshift(sinogram_stack,axes=2)
#    
#    #print("iradon fft1=",end - start)
#
#    sinogram_stack *= hfilter
#    sinogram_stack  = xp.reshape(sinogram_stack,(num_slices,num_rays*num_angles)).T
#    #qt=xp.reshape(sinogram_stack,(num_slices,num_rays*num_angles))
#
#    #start = timer()
#    qxy=S*sinogram_stack
#    #end = timer()
#    #print("iradon SpMV=",end - start)
#
#    #start = timer()
#
#    #qxy=S*xp.reshape(sinogram_stack,(num_slices,num_rays*num_angles)).T
#    qxy=xp.reshape(qxy,(num_rays+2*k_r+1,num_rays+2*k_r+1,num_slices))
#    #end = timer()
#    #print("reshaping SpMV=",end - start)
#    #qxy[k_r:-k_r-1, k_r: -k_r-1,:] 
#    tomogram = qxy[k_r:-k_r-1, k_r: -k_r-1,:] 
#    
#    #tomogram = np.moveaxis(tomogram,0,0)
#    tomogram = np.moveaxis(tomogram,2,0)
#    # removing the highest frequency        
#    #tomogram = np.moveaxis(tomogram,0,0)
#    tomogram[:,0,:]=0
#    tomogram[:,:,0]=0
#    #tomogram=xp.fft.fftshift(tomogram,(1,2))
#
#    #tomogram=xp.fft.fftshift(tomogram,(1,2))
#
#    start = timer()
#    #tomogram=xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(tomogram,(1,2))),(1,2))
#    tomogram=xp.fft.fftshift(xp.fft.ifft2(tomogram),(1,2))
#    #tomogram=xp.fft.ifft2(tomogram)
#    end = timer()
#    #print("iradon fft2=",end - start)
#    #tomogram=xp.fft.fftshift(tomogram,(1,2))
#    tomogram*=deapodization_factor
#
#    return tomogram  

    
    tomo_stack = xp.zeros((num_slices, num_rays , num_rays ),dtype=xp.complex64)
    

    for i in range(0,num_slices,2):
        if i > num_slices-2:
            qt=sinogram_stack[i]
        else:
            qt=sinogram_stack[i]+1j*sinogram_stack[i+1]
            
    #for i in range(num_slices):
        
        # from (r,theta) - radon space - to (q,theta)
        #qt = xp.fft.fftshift(sinogram_stack[i],axes=1)
        qt=sinogram_stack[i]
        qt = xp.fft.fft(qt,axis=1)  
        qt = xp.fft.fftshift(qt,axes=1)
        
        # non uniform iifft:
        
        # density compensation
        qt *= hfilter[0,0]
        
        # inverse gridding from (q,theta) to (qx,qy) 
        qxy=S*(qt).flatten()
        qxy=xp.reshape(qxy,(num_rays+2*k_r+1,num_rays+2*k_r+1))
        #qxy=xp.reshape(S*(qt).flatten(),(num_rays+2*k_r+1,num_rays+2*k_r+1))
        tomogram = qxy[k_r:-k_r-1, k_r: -k_r-1] 
        
        # removing the highest frequency
        tomogram[0,:]=0
        tomogram[:,0]=0
        
        # back to (xy) space
        tomogram=xp.fft.fftshift(xp.fft.ifft2((tomogram)))
        #tomo_stack[i] = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(tomogram)))
        #tomo_stack[i] = tomogram
        if i > num_slices-2:
            tomo_stack[i]=tomogram.real
        else:
            tomo_stack[i]=tomogram.real
            tomo_stack[i+1]=tomogram.imag
    

    
    tomo_stack*=deapodization_factor

#    return tomo_stack
  
    
#    tomo_stack1=xp.zeros((num_slices0, num_rays , num_rays), dtype=xp.float32 )
#    tomo_stack1[0::2,:,:]=tomo_stack.real
#    tomo_stack1[1::2,:,:]=tomo_stack[:num_slices0*2,:,:].imag
    return tomo_stack
    


    #return tomogram

def radon(tomo_stack, deapodization_factor, ST, k_r, num_angles ):
    
    xp=np
    num_slices = tomo_stack.shape[0]
    
    num_rays   = tomo_stack.shape[2]

    sinogram_stack = np.zeros((num_slices, num_angles, num_rays),dtype=np.complex64)

    
#    qxy = np.zeros((num_slices,num_rays+2*k_r+1, num_rays+2*k_r+1), dtype=np.complex64)
#    
#    tomo_stack1 = tomo_stack*deapodization_factor
#    tomo_stack1 = np.fft.fft2(np.fft.fftshift(tomo_stack1,axes=(1,2)))
#    qxy[:,k_r+1:-k_r-1, k_r+1: -k_r-1] = tomo_stack1[:,1:,1:]
#    
#    qxy=np.reshape(qxy,(num_slices,(num_rays+2*k_r+1)**2)).T
#    sinogram=np.reshape(ST*qxy,(num_angles,num_rays,num_slices))
#    sinogram = np.moveaxis(sinogram,2,0)
#    sinogram[:,:,0]=0
#    
#    sinogram = xp.fft.fftshift(sinogram,axes=2)
#    
#    sinogram = xp.fft.fft(sinogram,axis=2)
#    #end = timer()
#
#    sinogram = xp.fft.fftshift(sinogram,axes=2)
#    return sinogram

    
#    
#    print("qxy shape",qxy.shape,"S shape",ST.shape)
#    print("sinogram",sinogram.shape)
#    
    qxy = np.zeros((num_rays+2*k_r+1, num_rays+2*k_r+1), dtype=np.complex64)
    
    for i in range(num_slices):
        
        tomo_slice = tomo_stack[i] * deapodization_factor[0]
        # forward 2D fft centered
        
        if i== num_slices//2:
            tomo_slice+=0
            
        
        #tomo_slice = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(tomo_slice)))
        tomo_slice = np.fft.fft2(np.fft.fftshift(tomo_slice))
        #tomo_slice = tomo_stack1[i]
        
        # copy to qxy while removing the highest frequency
        qxy[k_r+1:-k_r-1, k_r+1: -k_r-1] = tomo_slice[1:,1:]
        
        #print("qxy shape",qxy.shape,"S shape",ST.shape)
        
        sinogram=np.reshape(ST*(qxy).flatten(),(num_angles,num_rays))

        sinogram[:,0]=0
        
        # inverse 2D fft centered
        
        sinogram = np.fft.ifftshift(sinogram,axes=1)        
        sinogram = np.fft.ifft(sinogram,axis=1)        

        sinogram = np.fft.ifftshift(sinogram,axes=1)
        
        sinogram_stack[i]=sinogram
    
    return sinogram_stack
        
        

def gridrec(sinogram_stack, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for backward proj
 
    num_angles=theta_array.size
    #tomo_stack = xp.zeros((sinogram_stack.shape[0], num_rays + 1, num_rays + 1),dtype=xp.complex64)
    tomo_stack = xp.zeros((sinogram_stack.shape[0], num_rays , num_rays ),dtype=xp.complex64)
    #tomo_stack = xp.zeros((sinogram_stack.shape[0], num_rays+2 , num_rays+2 ),dtype=xp.complex64)
    
    # includes scaling factors (1.118 fudge to match tomopy- depends on size slightly)
    ramlak_filter = (xp.abs(xp.array(range(num_rays)) - num_rays/2)+1./num_rays)*2./(num_rays**2)/num_angles*1.1184626
        
            
    # removing the highest frequency
    ramlak_filter[0]=0;
   
    ramlak_filter = ramlak_filter.reshape((1, ramlak_filter.size))
     
    
    for i in range(sinogram_stack.shape[0]): #loops through each slice
        
        print("Reconstructing slice " + str(i))
         
        sinogram = xp.fft.fftshift(sinogram_stack[i],axes=1)
        
        sinogram = xp.fft.fft(sinogram,axis=1)  
#        sinogram = xp.fft.rfft(sinogram,axis=1)  

        sinogram = xp.fft.fftshift(sinogram,axes=1)
        sinogram *= ramlak_filter

        start = timer()

        #tomogram = grid_rec_one_slice(sinogram, theta_array, num_rays, k_r, kernel_type, xp, mode)
        #tomogram = grid_rec_one_slice2(sinogram, theta_array, num_rays, k_r, kernel_type, xp, mode)
        #tomogram = grid_rec_one_slice0(sinogram, theta_array, num_rays, k_r, kernel_type, xp, mode)
        tomogram = grid_rec_one_sliceSpMV(sinogram, theta_array, num_rays, k_r, kernel_type, xp, mode)
 
        # removing the highest frequency
        tomogram[0,:]=0
        tomogram[:,0]=0
        
        end = timer()
        print("total regridding time=",end - start)

        tomo_stack[i] = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(tomogram)))
    

    deapodization_factor = deapodization(num_rays, kernel_type, xp, k_r)
    
    tomo_stack*=deapodization_factor
    return tomo_stack


def gridrec_transpose(tomo_stack, theta_array, num_rays, k_r, kernel_type, xp, mode): #use for forward proj

    sinogram_stack = np.zeros((tomo_stack.shape[0], theta_array.shape[0], num_rays),dtype=np.complex64)

    # includes scaling factors - fudge factor to match tomopy 1.0467145 
    deapodization_factor = deapodization(num_rays, kernel_type, xp, k_r)/num_rays/3.5*1.046684
    
    for i in range(tomo_stack.shape[0]): #loops through each slice
        
        tomo_slice = tomo_stack[i] * deapodization_factor
        # forward 2D fft centered

        tomo_slice = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(tomo_slice)))
        # removing the highest frequency
        tomo_slice[0,:]=0
        tomo_slice[:,0]=0

        #sinogram = grid_rec_one_slice_transpose0(tomo_slice, theta_array, num_rays, k_r, kernel_type, xp, mode)
        sinogram = grid_rec_one_slice_transposeSpMV(tomo_slice, theta_array, num_rays, k_r, kernel_type, xp, mode)

        # removing the highest frequency
        sinogram[:,0]=0
        
        # inverse 2D fft centered
        
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
    try:
        import cupy as cp
        for key, value in host_pointers.items():      
            host_pointers[key] = cp.asarray(value)
    except:
        print("Unable to import CuPy")
def memcopy_to_host(device_pointers):
    try:
        import cupy as cp
        for key, value in device_pointers.items():       
            device_pointers[key] = cp.asnumpy(value)
    except:
        print("Unable to import CuPy")
        
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
