#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
#import cupy as cp

eps=np.finfo(np.float32).eps
pi=np.pi



def gaussian_kernel(x, k_r, sigma, xp): #using sigma=2 since this is the default value for sigma
#    kernel1 = xp.exp(-1/2 * (x/sigma)**2 *16)
    #xx=xp.exp(-x)
    kernel1 = xp.exp(-1/2 * 10 * (x/sigma)**2 )
    kernel1 = kernel1* (xp.abs(x) < k_r)
    return kernel1 
    
def keiser_bessel(x, k_r, beta,xp):
    #kernel1 = xp.i0(beta*xp.sqrt(1-(2*x/(k_r))**2))/xp.abs(xp.i0(beta)) 
    #kernel1 = kernel1* (xp.abs(x) < k_r)
    #print(beta)
    beta=beta*1.85
    kernel1 =  xp.i0(beta*xp.sqrt((xp.abs(x) <= k_r)*(1-(x/(k_r))**2)))/xp.abs(xp.i0(beta))
    kernel1 = xp.reshape(kernel1, x.shape)
    kernel1 = kernel1 * (xp.abs(x) < k_r) 
    #kernel1 = (xp.abs(x) < k_r) * xp.i0(beta*xp.sqrt((xp.abs(x) <= k_r)*(1-(x/(k_r))**2)))/xp.abs(xp.i0(beta)) 
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
        
def iradon_filter(num_rays,num_angles,filter_type,xp):
    
    qq = xp.abs(xp.array(range(num_rays)) - num_rays/2)
    density_comp_f = xp.abs(qq+1./num_rays)/(num_rays**3)/num_angles
    
    if filter_type == 'ram-lak':
        pass
    elif filter_type == 'hamming':
        density_comp_f *= 0.54-0.46 * xp.cos((qq+num_rays//2)*2*xp.pi/num_rays)
    elif filter_type == 'hann':
        density_comp_f *= 0.5-0.5 * xp.cos((qq+num_rays//2)*2*xp.pi/num_rays)
    elif filter_type == 'cosine':
        density_comp_f *= xp.cos((qq)*xp.pi/num_rays)
    elif filter_type == 'sinc' or filter_type == '':
         density_comp_f *= xp.sinc((qq)/num_rays)
    elif filter_type == None:
        density_comp_f = xp.zeros(num_rays)+1./(num_rays**3)/num_angles
        
    # removing the highest frequency
    density_comp_f[0]=0;

    return density_comp_f

        
"""    
def K2(x, y, kernel_type, xp, k_r=2, beta=1, sigma=2): #k_r and beta are parameters for keiser-bessel
    if kernel_type == 'gaussian':
        kernel2 = gaussian_kernel(x,k_r,sigma,xp) * gaussian_kernel(y,k_r,sigma,xp)
        return kernel2
    elif kernel_type == 'kb':
        kernel2 = keiser_bessel(x,k_r,beta,xp) * keiser_bessel(y,k_r,beta,xp)
        return kernel2
    else:
        print("Invalid kernel type")

def deapodization_simple(num_rays,kernel_type,xp,k_r=2, beta=1, sigma=2):
    ktilde=K1(xp.array([range(-k_r, k_r+1),] ), k_r, kernel_type, xp,  beta, sigma)
    padding_array = ((0,0),(num_rays//2 - k_r , num_rays//2 - k_r-1))
    ktilde = xp.lib.pad(ktilde, padding_array, 'constant', constant_values=0)
    ktilde = xp.fft.fftshift(xp.fft.ifft(xp.fft.ifftshift(ktilde)))

    deapodization_factor = ktilde * ktilde.T 
#    print("ratio i/r=", xp.max(abs(deapodization_factor.imag))/xp.max(abs(deapodization_factor.real)))
 
    return 1./deapodization_factor.real

def deapodization(num_rays,kernel_type,xp,k_r=2, beta=1, sigma=2):
    # we upsample the kernel
    sampling=8
    step=1./sampling
    stencil=xp.array([xp.arange(-k_r, k_r+1-step*(sampling-1),step),])
   
    ktilde=K1(stencil,k_r, kernel_type, xp,  beta, sigma)
    
    padding_array = ((0,0),(num_rays*sampling//2 - k_r*sampling , num_rays*sampling//2 - k_r*sampling-1))
    ktilde1=xp.lib.pad(ktilde, padding_array, 'constant', constant_values=0)
    ktilde1 = xp.fft.fftshift(xp.fft.ifft(xp.fft.ifftshift(ktilde1)))
    #print("ratio i/r=", xp.max(abs(deapodization_factor.imag))/xp.max(abs(deapodization_factor.real)))

    ktilde2=(ktilde1[:,num_rays*(sampling)//2-num_rays//2:num_rays*(sampling)//2+num_rays//2]).real
    
    apodization_factor = ktilde2 * ktilde2.T
    
    return 1./apodization_factor
"""



def deapodization_shifted(num_rays,kernel_type,xp,k_r=2, beta=1, sigma=2):
    #stencil=xp.array([range(-k_r, k_r+1),]
    sampling=8
    step=1./sampling
    stencil=xp.array(xp.arange(-k_r, k_r+1-step*(sampling-1),step))

   
    ktilde=K1(stencil,k_r, kernel_type, xp,  beta, sigma)
    
    padding_array = ((num_rays*sampling//2 - k_r*sampling ),( num_rays*sampling//2 - k_r*sampling-1))

    ktilde1=xp.pad(ktilde, padding_array, 'constant', constant_values=0)
    # skip one fftshift so that we avoid one fftshift in the iteration
    ktilde1 = xp.fft.ifftshift(xp.fft.ifft(ktilde1))
    
    ktilde1=xp.reshape(ktilde1,(1,-1))

    # since we upsampled in Fourier space, we need to crop in real space
    ktilde2=(ktilde1[:,num_rays*(sampling)//2-num_rays//2:num_rays*(sampling)//2+num_rays//2]).real
    
    apodization_factor = ktilde2 * ktilde2.T
    return 1./apodization_factor


"""
def generate_Shepp_Logan(cube_shape):

   return tomopy.misc.phantom.shepp3d(size=cube_shape, dtype='float32')
"""

def forward_project(cube, theta):

    #return tomopy.sim.project.project(cube, theta, pad = False, sinogram_order=False)
    return tomopy.sim.project.project(cube, theta, pad = False, sinogram_order=True)
   

def gridding_setup(num_rays, theta_array, center=None, xp=np, kernel_type = 'gaussian', k_r = 2):
    # setting up the sparse array
    # columns are used to pick the input data
    # rows are where the data is added on the output  image
    # values are what we multiply the input with
    
    num_angles=theta_array.shape[0]
    
    # phase ramp to move center
    if center == None or center == num_rays//2:
        rampfactor=-1   
    else:
        rampfactor=xp.exp(1j*2*xp.pi*center/num_rays)
    
    stencil=xp.array([range(-k_r, k_r+1),])
    stencilx=xp.reshape(stencil,[1,1,2*k_r+1,1])
    stencily=xp.reshape(stencil,[1,1,1,2*k_r+1])

    # 
    # this avoids one fftshift after fft(data)
    qray = xp.fft.fftshift(xp.array(xp.arange(num_rays) ) - num_rays/2)
    qray= xp.reshape(qray,(1,num_rays))
    
    theta_array = theta_array.astype('float64')
    theta_array+=eps*10 # gives better result if theta[0] is not exactly 0
    
    # coordinates of the points on the  grid, 
    px = - (xp.sin(xp.reshape(theta_array,[num_angles,1]))) * (qray)
    py =   (xp.cos(xp.reshape(theta_array,[num_angles,1]))) * (qray) 
    # we'll add the center later to avoid numerical errors
    
    # reshape coordinates and index for the stencil expansion
    px=xp.reshape(px,[num_angles,num_rays,1,1])
    py=xp.reshape(py,[num_angles,num_rays,1,1])
    
    # compute kernels
    kx=K1(stencilx - (px - xp.around(px)),k_r, kernel_type, xp); 
    ky=K1(stencily - (py - xp.around(py)),k_r, kernel_type, xp); 
    
    qray=rampfactor**qray     # this takes care of the center
    qray.shape=(1,num_rays,1,1)
    qray[:,num_rays//2,:,:]=0 # removing highest frequency from the data

    Kval=(kx*ky)*(qray)
    
    # move the grid to the middle and add stencils
    kx=stencilx+xp.around(px)+ num_rays/2
    ky=stencily+xp.around(py)+ num_rays/2


    # this avoids fftshift2(tomo)
    Kval*=((-1)**kx)*((-1)**ky)
    # the complex conjugate
    Kval_conj=xp.conj(Kval)+0.

    # index (where the output goes on the cartesian grid)
    Krow=((kx)*(num_rays)+ky)
    
    tscale=xp.ones(num_angles)
    tscale.shape=[num_angles,1,1,1]
    # check if theta goes to 180, then assign 1/2 weight to each
    theta=theta_array
    theta_repeat=xp.abs(xp.abs(theta[0]-theta[-1])-xp.pi)<xp.abs(theta[1]-theta[0])*1e-5
    if theta_repeat:
        tscale[([0,-1])]=.5
        
    
    Kval*=tscale
        
    # input index (where the the data on polar grid input comes from) 
    pind = xp.array(range(num_rays*num_angles))
    # we need to replicate for each point of the kernel 
    # reshape index for the stencil expansion
    pind=xp.reshape(pind,[num_angles,num_rays,1,1])
    # column index 
    Kcol=(pind+kx*0+ky*0)

    # find points out of bound
    # let's remove the highest frequencies as well (kx=0,ky=0)
    ii=xp.where((kx>=1) & (ky>=1) & (kx<=num_rays-1) & (ky<=num_rays-1))

    # remove points out of bound
    Krow=Krow[ii]
    Kcol=Kcol[ii]
    Kval=Kval[ii]
    Kval_conj=Kval_conj[ii]
    
    Krow=Krow.astype('long')
    Kcol=Kcol.astype('long')
    Kval=Kval.astype('complex64')
    Kval_conj=Kval_conj.astype('complex64')
    
    # create sparse array   
    if xp.__name__=='cupy':
        import cupyx
        #S=scipy.sparse.coo_matrix((Kval.ravel(),(Kcol.ravel(), Krow.ravel())), shape=(num_angles*num_rays, (num_rays)**2))
        #ST=cupyx.scipy.sparse.coo_matrix((Kval_conj.ravel(), (Krow.ravel(), Kcol.ravel())), shape=((num_rays)**2, num_angles*num_rays))
        S=cupyx.scipy.sparse.coo_matrix((Kval.ravel(),(Krow.ravel(), Kcol.ravel())), shape=((num_rays)**2,num_angles*num_rays))
        S=cupyx.scipy.sparse.csr_matrix(S)
        ST=cupyx.scipy.sparse.coo_matrix((Kval_conj.ravel(), (Kcol.ravel(), Krow.ravel())), shape=(num_angles*num_rays, (num_rays)**2))
        ST=cupyx.scipy.sparse.csr_matrix(ST)
        #print("size of S in gb", (8*(S.data).size+4*(S.indptr).size+4*(S.indices).size+5*4)/((2**10)**3))
        #print("size of S",8*(S.data).size,"ind ptr",4*(S.indptr).size,'ind',4*(S.indices).size)

    else:
        import scipy

        #S=scipy.sparse.csr_matrix((Kval.ravel(),      (Kcol.ravel(), Krow.ravel())), shape=(num_angles*num_rays, (num_rays)**2))
        #ST=scipy.sparse.csr_matrix((Kval_conj.ravel(), (Krow.ravel(), Kcol.ravel())), shape=((num_rays)**2, num_angles*num_rays))
        S=scipy.sparse.csr_matrix((Kval.ravel(),      (Krow.ravel(), Kcol.ravel())), shape=((num_rays)**2,num_angles*num_rays))
        ST=scipy.sparse.csr_matrix((Kval_conj.ravel(), (Kcol.ravel(), Krow.ravel())), shape=(num_angles*num_rays, (num_rays)**2))
        

    
    
       
    return S, ST

def masktomo(num_rays,xp,width=.95):
    xx=xp.array([range(-num_rays//2, num_rays//2)])
    msk_sino=(xp.abs(xx)<(num_rays//2*width*.98)).astype('float32')

    msk_sino.shape=(1,1,num_rays)
    
    xx=xx**2
    rr2=xx+xx.T
    msk_tomo=rr2<(num_rays//2*width)**2
    msk_tomo.shape=(1,num_rays,num_rays)
    return msk_tomo, msk_sino

import matplotlib.pyplot as plt
def radon_setup(num_rays, theta_array, xp=np, center=None, filter_type = 'hamming', kernel_type = 'gaussian', k_r = 1, width=.5):

    #print("setting up gridding")
    #start = timer()

    S, ST = gridding_setup(num_rays, theta_array, center, xp, kernel_type , k_r)
    #end = timer()

    #print("gridding setup time=",end - start)
    if xp.__name__=='cupy':
        import fft
        #import cupy.fft as fft
    else:
        import numpy.fft as fft
        
    
    
    #deapodization_factor = deapodization(num_rays, kernel_type, xp, k_r)
    deapodization_factor = deapodization_shifted(num_rays, kernel_type, xp, k_r=k_r)
    deapodization_factor=xp.reshape(deapodization_factor,(1,num_rays,num_rays))
    
    # get the filter
    num_angles=theta_array.shape[0]
    #print("num_angles", num_angles)
    #density_comp_f = (xp.abs(xp.array(range(num_rays)) - num_rays/2)+1./num_rays)/(num_rays**2)/num_angles/9.8

    density_comp_f = iradon_filter(num_rays,num_angles,filter_type,xp)
    
    # we shift the sparse matrix output so we work directly with shifted fft
    density_comp_f=xp.fft.fftshift(density_comp_f)

    # reshape so that we can broadcast to the whole stack
    density_comp_f = density_comp_f.reshape((1, 1, density_comp_f.size))
    
    none_filter=density_comp_f*0+1./(num_rays**3)/num_angles
    

    # mask out outer tomogram
    msk_tomo,msk_sino=masktomo(num_rays,xp,width=width)
    
    
    deapodization_factor/=num_rays
    deapodization_factor*=msk_tomo

    deapodization_factor*=0.14652085
    deapodization_factor=(deapodization_factor).astype('complex64')
    
        

    R  = lambda tomo:  radon(tomo, deapodization_factor , ST, k_r, num_angles,xp,fft )

    dpr= deapodization_factor*num_rays*154.10934

    # the conjugate transpose (for least squares solvers):
    RT = lambda sino: iradon(sino, dpr, S,  k_r, none_filter,xp,fft)
    
    # inverse Radon (pseudo inverse)
    IR = lambda sino: iradon(sino, dpr, S,  k_r, density_comp_f,xp,fft)
    
    
    return R,IR,RT
    

def iradon(sinogram_stack, deapodization_factor, S, k_r , hfilter,xp,fft): 
    
    num_slices = sinogram_stack.shape[0]
    num_angles = sinogram_stack.shape[1]
    num_rays   = sinogram_stack.shape[2]
    
    
    """
    #-----------------------------------------------------------#  
    #          vectorized code                                  #  
    
    
    nslice2=xp.int(xp.ceil(num_slices/2))     
    if num_slices==1:
        qts=sinogram_stack
    elif xp.mod(num_slices,2) ==1:
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
        qxy=xp.concatenate((qxy.real,qxy.imag),axis=0)
        qxy.shape=(2,nslice2,num_rays,num_rays)
        qxy=xp.moveaxis(qxy,1,0)
        qxy=xp.reshape(qxy,(nslice2*2,num_rays,num_rays))
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
        
        #qt = xp.fft.fft(qt)
        #qt = fft.fft(qt)
        qt = fft.fft(qt)
        
        
         ###################################
        # non uniform IFFT: Fourier polar (q,theta) to cartesian (x,y):
       
        # density compensation
        qt *= hfilter[0,0]
        
        # inverse gridding (polar (q,theta) to cartesian (qx,qy))
        #qt.shape=(-1)        
        #tomogram=qt.ravel()*S #SpMV
        tomogram=S*qt.ravel() #SpMV
        
        tomogram.shape=(num_rays,num_rays)

        # Fourier cartesian (qx,qy) to real (xy) space
        tomogram = fft.ifft2(tomogram)
#        tomogram=fft.ifft2(tomogram)*deapodization_factor[0]
        tomogram*=deapodization_factor[0]
        
        # end of non uniform FFT 
        ###################################
        # extract two slices out of complex
        if i > num_slices-2:
            tomo_stack[i]=tomogram.real
        else:
            tomo_stack[i]=tomogram.real
            tomo_stack[i+1]=tomogram.imag
    
    
    return tomo_stack
    


def radon(tomo_stack, deapodization_factor, ST, k_r, num_angles,xp,fft ):
    
    
    num_slices = tomo_stack.shape[0]
    num_rays   = tomo_stack.shape[2]

    sinogram_stack = xp.empty((num_slices, num_angles, num_rays),dtype=xp.float32)
    
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
        
        ###################################
        # non uniform FFT cartesian (x,y) to Fourier polar (q,theta):
        tomo_slice=tomo_slice*deapodization_factor
        tomo_slice = fft.fft2(tomo_slice)
        
        # gridding from cartiesian (qx,qy) to polar (q-theta)
        sinogram=(ST)*tomo_slice.ravel() #SpMV
        sinogram.shape=(num_angles,num_rays)

        # end of non uniform FFT
        ###################################
        
        # (q-theta) to radon (r-theta) :       
        sinogram = fft.ifft(sinogram)  
        
        # put the sinogram in the stack
        # extract two slices out of complex
        if i > num_slices-2:
            #print("ratio gridrec_transpose i/r=",  xp.max(xp.abs(sinogram.imag)/xp.max(xp.abs(sinogram.real))))

            sinogram_stack[i]=sinogram.real
        else:
            sinogram_stack[i]=sinogram.real
            sinogram_stack[i+1]=sinogram.imag
        
     
    return sinogram_stack
        
        
