#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:25:06 2019

@author: anu
"""

#Gridrec algorithm based on pseudocode using Gaussian kernel
#from numba import jit
#import matplotlib.pyplot as plt
#import h5py
#import dxchange
import tomopy
#from scipy import 
import imageio
import os
from timeit import default_timer as timer
import numpy as np
import scipy
#import cupy as cp

def gaussian_kernel(x, k_r, sigma, xp): #using sigma=2 since this is the default value for sigma
#    kernel1 = xp.exp(-1/2 * (x/sigma)**2 *16)
    kernel1 = xp.exp(-1/2 * 10 * (x/sigma)**2 )
    kernel1 = kernel1* (xp.abs(x) <= k_r)
    return kernel1 
    
def keiser_bessel(x, k_r, beta,xp):
#    kernel1 = np.i0(beta*np.sqrt(1-(2*x/(k_r))**2))/np.abs(np.i0(beta)) 
    kernel1 = (xp.abs(x) <= k_r/2) * xp.i0(beta*xp.sqrt((xp.abs(x) <= k_r/2)*(1-(2*x/(k_r))**2)))/xp.abs(xp.i0(beta)) 
    return kernel1

# general kernel 1D
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
    # we upsample the kernel
    sampling=8
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
    sampling=8
    step=1./sampling
    stencil=np.array([np.arange(-k_r, k_r+1-step*(sampling-1),step),])
   
    ktilde=K1(stencil,k_r, kernel_type, xp,  beta, sigma)
    
    padding_array = ((0,0),(num_rays*sampling//2 - k_r*sampling , num_rays*sampling//2 - k_r*sampling-1))
    ktilde1=np.lib.pad(ktilde, padding_array, 'constant', constant_values=0)
    # skip one fftshift so that we avoid one fftshift in the iteration
    ktilde1 = np.fft.ifftshift(np.fft.ifft(ktilde1))
    
    # since we upsampled in Fourier space, we need to crop in real space
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

    qxy = np.zeros((num_rays + k_r * 2 + 1, num_rays + k_r * 2 + 1), dtype=np.complex64)
    
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
    
    # setting up the sparse array
    S, ST = gridding_setup(num_rays, theta_array, None, xp, kernel_type , k_r)
#    
#    start = timer()
#    num_angles=theta_array.shape[0]
#    
#    stencil=np.array([range(-k_r, k_r+1),])
#    stencilx=np.reshape(stencil,[1,1,2*k_r+1,1])
#    stencily=np.reshape(stencil,[1,1,1,2*k_r+1])
#    
#    # coordinates of the points on the  grid
#    px = - (xp.sin(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
#    py =   (xp.cos(np.reshape(theta_array,[num_angles,1]))) * (xp.array([range(num_rays) ]) - num_rays/2)+ num_rays/2 + k_r  #adding k_r accounts for the padding
#    
#    # input index (where the data comes from)
#    pind = xp.array(range(num_rays*num_angles))
#    
#    # reshape coordinates and index
#    px=np.reshape(px,[num_angles,num_rays,1,1])
#    py=np.reshape(py,[num_angles,num_rays,1,1])
#    pind=np.reshape(pind,[num_angles,num_rays,1,1])
#    
#    # compute kernel
#    kx=K1(stencilx - (px - xp.around(px)),k_r, kernel_type, xp); 
#    ky=K1(stencily - (py - xp.around(py)),k_r, kernel_type, xp); 
#    Kval=(kx*ky)
#
#    # expand coordinates with stencils
#    kx=stencilx+xp.around(px)
#    ky=stencily+xp.around(py)
#
#    # row  index (where the output goes)
#    Krow=((kx)*(num_rays+2*k_r+1)+ky)
#    # column index (where to get the input)
#    Kcol=(pind+kx*0+ky*0)
#    
#    # create sparse array
#    S=scipy.sparse.csr_matrix((Kval.flatten(), (Krow.flatten(), Kcol.flatten())), shape=((num_rays+2*k_r+1)**2, num_angles*num_rays))
#
#    end = timer()
#    
#    print("Spmv setup time=",end - start)

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

def gridding_setup(num_rays, theta_array, center=None, xp=np, kernel_type = 'gaussian', k_r = 2):
    # setting up the sparse array
    # columns are used to pick the input data
    # rows are where the data is added on the output  image
    # values are what we multiply the input with
    
    num_angles=theta_array.shape[0]
    
    if center == None or center == num_rays//2:
        rampfactor=-1   
#        print("not centering")
    else:
#        print("centering")
        rampfactor=np.exp(1j*2*np.pi*center/num_rays)
    
    stencil=xp.array([range(-k_r, k_r+1),])
    stencilx=xp.reshape(stencil,[1,1,2*k_r+1,1])
    stencily=xp.reshape(stencil,[1,1,1,2*k_r+1])

    # 
    # this avoids one fftshift after fft(data)
    qray = xp.fft.fftshift(xp.array([range(num_rays) ]) - num_rays/2)

    # coordinates of the points on the  grid, 
    px = - (xp.sin(np.reshape(theta_array,[num_angles,1]))) * (qray)
    py =   (xp.cos(np.reshape(theta_array,[num_angles,1]))) * (qray) 
    # we'll add the center later to avoid numerical errors
    
    # reshape coordinates and index for the stencil expansion
    px=xp.reshape(px,[num_angles,num_rays,1,1])
    py=xp.reshape(py,[num_angles,num_rays,1,1])
    
    # compute kernels
    kx=K1(stencilx - (px - xp.around(px)),k_r, kernel_type, xp); 
    ky=K1(stencily - (py - xp.around(py)),k_r, kernel_type, xp); 
    
    qray=rampfactor**qray     # this avoid the fftshift(data)
    qray.shape=(1,num_rays,1,1)
    qray[:,num_rays//2,:,:]=0 # removing highest frequency from the data

    Kval=(kx*ky)*(qray)

    
    # check if theta goes to 180, then assign 1/2 weight to each
    theta=theta_array
    if xp.abs(xp.abs(theta[0]-theta[-1])-np.pi)<xp.abs(theta[1]-theta[0])*1e-5:
        tscale=xp.ones(num_angles)
        tscale[([0,-1])]=.5
        tscale.shape=[num_angles,1,1,1]
        Kval*=tscale
    
    
    # move the grid to the middle and add stencils
    kx=stencilx+xp.around(px)+ num_rays/2
    ky=stencily+xp.around(py)+ num_rays/2
    # find points out of bound
    #ii=xp.where((kx>=0) & (ky>=0) & (kx<=num_rays-1) & (ky<=num_rays-1))
    # let's remove the highest frequencies as well (kx=0,ky=0)
    ii=xp.where((kx>=1) & (ky>=1) & (kx<=num_rays-1) & (ky<=num_rays-1))


    # this avoids fftshift2(tomo)  and and fftshift1(sino)
    Kval*=((-1)**kx)*((-1)**ky)
    
    # index (where the output goes on the cartesian grid)
    Krow=((kx)*(num_rays)+ky)
    
    # input index (where the the data on polar grid input comes from) 
    pind = xp.array(range(num_rays*num_angles))
    # we need to replicate for each point of the kernel 
    # reshape index for the stencil expansion
    pind=xp.reshape(pind,[num_angles,num_rays,1,1])
    # column index 
    Kcol=(pind+kx*0+ky*0)
    
    # remove points out of bound
    Krow=Krow[ii]
    Kcol=Kcol[ii]
    Kval=Kval[ii]
    
#    print("Kval type",Kval.dtype,"Krow type",Krow.dtype)
#    Kval=np.float32(Kval)
    Kval=xp.complex64(Kval)
    Krow=xp.longlong(Krow)
    Kcol=xp.longlong(Kcol)
    
    # create sparse array   
    S=scipy.sparse.csr_matrix((Kval.ravel(),(Kcol.ravel(), Krow.ravel())), shape=(num_angles*num_rays, (num_rays)**2))
    
    Kval=np.conj(Kval)
    # note that we swap column and rows to get the transpose
    #ST=scipy.sparse.csr_matrix((Kval.ravel(),(Kcol.ravel(), Krow.ravel())), shape=(num_angles*num_rays, (num_rays)**2))
    ST=scipy.sparse.csr_matrix((Kval.ravel(), (Krow.ravel(), Kcol.ravel())), shape=((num_rays)**2, num_angles*num_rays))

    
#    # create sparse array   
#    S=scipy.sparse.csr_matrix((Kval.flatten(), (Krow.flatten(), Kcol.flatten())), shape=((num_rays+2*k_r+1)**2, num_angles*num_rays))
#
#    # note that we swap column and rows to get the transpose
#    ST=scipy.sparse.csr_matrix((Kval.flatten(),(Kcol.flatten(), Krow.flatten())), shape=(num_angles*num_rays, (num_rays+2*k_r+1)**2))
    
       
    return S, ST

def masktomo(num_rays,xp,width=.65):
    xx=xp.array([range(-num_rays//2, num_rays//2)])
    msk_sino=xp.float32(np.abs(xx)<(num_rays//2*width))
    msk_sino.shape=(1,1,num_rays)
    
    xx=xx**2
    rr2=xx+xx.T
    msk_tomo=rr2<(num_rays//2*width*1.02)**2
    msk_tomo.shape=(1,num_rays,num_rays)
    return msk_tomo, msk_sino


def radon_setup(num_rays, theta_array, center=None,xp=np, kernel_type = 'gaussian', k_r = 2):

    #print("setting up gridding")
    start = timer()

    S, ST = gridding_setup(num_rays, theta_array, center, xp, kernel_type , k_r)
    end = timer()
    #print("gridding setup time=",end - start)

    
    #deapodization_factor = deapodization(num_rays, kernel_type, xp, k_r)
    deapodization_factor = deapodization_shifted(num_rays, kernel_type, xp, k_r)
    deapodization_factor=xp.reshape(deapodization_factor,(1,num_rays,num_rays))
    
    # get the filter
    num_angles=theta_array.shape[0]
    #print("num_angles", num_angles)
    #ramlak_filter = (xp.abs(xp.array(range(num_rays)) - num_rays/2)+1./num_rays)/(num_rays**2)/num_angles/9.8
    ramlak_filter = (xp.abs(xp.array(range(num_rays)) - num_rays/2)+1./num_rays)/(num_rays**3)/num_angles
    
    # this is to avoid one fftshift
    #ramlak_filter*=(-1)**np.arange(num_rays);
    
    
    # removing the highest frequency
    ramlak_filter[0]=0;
    
    
    # we shifted the sparse matrix output so we work directly with shifted fft
    ramlak_filter=xp.fft.fftshift(ramlak_filter)

    # reshape so that we can broadcast to the whole stack
    ramlak_filter = ramlak_filter.reshape((1, 1, ramlak_filter.size))
    none_filter=ramlak_filter*0+1./(num_rays**3)/num_angles
    

    # mask out outer tomogram
    msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
    
    
    
    deapodization_factor/=num_rays
    deapodization_factor*=msk_tomo

    deapodization_factor*=0.14652085
    deapodization_factor=xp.float32(deapodization_factor)
    
    
    
    R  = lambda tomo:  radon(tomo, deapodization_factor , ST, k_r, num_angles )
    # the conjugate transpose (for least squares solvers):
    RT = lambda sino: iradon(sino, dpr, S,  k_r, none_filter)
    
    # inverse Radon (pseudo inverse)
    dpr= deapodization_factor*num_rays*154.10934
    IR = lambda sino: iradon(sino, dpr, S,  k_r, ramlak_filter)
    
    
    return R,IR,RT
    

    
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
    with open("./src/convolve.cu", 'r') as myfile:
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
    
    
    """
    #-----------------------------------------------------------#  
    #          vectorized code                                  #  
    
    
    nslice2=np.int(np.ceil(num_slices/2))     
    if num_slices==1:
        qts=sinogram_stack
    elif np.mod(num_slices,2) ==1:
        #print('odd slides other than 1 not working yet')
        qts=sinogram_stack[0::2,:,:]
        qts[0:-1]+=1j*sinogram_stack[1::2,:,:]
    else:
        qts=sinogram_stack[0::2,:,:]+1j*sinogram_stack[1::2,:,:]
        #qts=sinogram_stack

    nslice=qts.shape[0]
    qts = xp.fft.fft(qts,axis=2)
    qts *= hfilter
    qts.shape = (nslice,num_rays*num_angles)

    qxy = qts*S

    qxy.shape=(nslice,num_rays,num_rays)

    qxy=xp.fft.ifft2(qxy)
    qxy*=deapodization_factor

 
    if num_slices==1:
        return qxy
    else:
        tomo_stack = xp.empty((num_slices, num_rays , num_rays ),dtype=xp.float32)
        tomo_stack[0::2,:,:]=qxy.real
        tomo_stack[1::2,:,:]=qxy[:nslice2*2,:,:].imag
        return tomo_stack
        qxy=np.concatenate((qxy.real,qxy.imag),axis=0)
        qxy.shape=(2,nslice2,num_rays,num_rays)
        qxy=np.moveaxis(qxy,1,0)
        qxy=np.reshape(qxy,(nslice2*2,num_rays,num_rays))
        return qxy[:num_slices,:,:]

    #         end vectorized code                                  #  
    #-----------------------------------------------------------#  
    """
    
    tomo_stack = xp.empty((num_slices, num_rays , num_rays ),dtype=xp.float32)
    
    # two slices at once    
    for i in range(0,num_slices,2):
        # merge two sinograms into one complex
        if i > num_slices-2:
            qt=sinogram_stack[i]
        else:
            qt=sinogram_stack[i]+1j*sinogram_stack[i+1]
            
       
        # radon (r-theta) to Fourier (q-theta) space        
        qt = xp.fft.fft(qt)  
        
         ###################################
        # non uniform IFFT: Fourier polar (q,theta) to cartesian (x,y):
       
        # density compensation
        qt *= hfilter[0,0]
        
        # inverse gridding (polar (q,theta) to cartesian (qx,qy))
        qt.shape=(-1)        
        tomogram=qt*S #SpMV
        tomogram.shape=(num_rays,num_rays)

        # Fourier cartesian (qx,qy) to real (xy) space

        tomogram=xp.fft.ifft2(tomogram)*deapodization_factor[0]
        
        
        # end of non uniform FFT 
        ###################################
        # extract two slices out of complex
        if i > num_slices-2:
            tomo_stack[i]=tomogram.real
        else:
            tomo_stack[i]=tomogram.real
            tomo_stack[i+1]=tomogram.imag
    

    return tomo_stack
    


def radon(tomo_stack, deapodization_factor, ST, k_r, num_angles ):
    
    xp=np
    num_slices = tomo_stack.shape[0]
    num_rays   = tomo_stack.shape[2]

    sinogram_stack = np.empty((num_slices, num_angles, num_rays),dtype=np.float32)
    
    deapodization_factor.shape=(num_rays,num_rays)

    #go through each slice
    #for i in range(0,num_slices):
    #    tomo_slice = tomo_stack[i] 
        
    # two slices at once by merrging into a complex   
    for i in range(0,num_slices,2):
        
        # merge two slices into complex
        if i > num_slices-2:
            tomo_slice = tomo_stack[i]
            
        else:
            tomo_slice = tomo_stack[i]+1j*tomo_stack[i+1]
        
        #sinogram = radon_oneslice(tomo_slice)
        ###################################
        # non uniform FFT cartesian (x,y) to Fourier polar (q,theta):
        
        tomo_slice = np.fft.fft2(tomo_slice*deapodization_factor)
        
        # gridding from cartiesian (qx,qy) to polar (q-theta)
        # tomo_slice.shape = (num_rays **2)        
        tomo_slice.shape = (-1)        
        sinogram=tomo_slice * ST #SpMV
        sinogram.shape=(num_angles,num_rays)

        # end of non uniform FFT
        ###################################
        
        # (q-theta) to radon (r-theta) :       
        sinogram = np.fft.ifft(sinogram)  
        
        
        # put the sinogram in the stack
        #sinogram_stack[i]=sinogram        
        # extract two slices out of complex
        if i > num_slices-2:
            print("ratio gridrec_transpose i/r=",  np.max(np.abs(sinogram.imag)/np.max(np.abs(sinogram.real))))

            sinogram_stack[i]=sinogram.real
        else:
            sinogram_stack[i]=sinogram.real
            sinogram_stack[i+1]=sinogram.imag
        
     
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
