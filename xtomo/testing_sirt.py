import gridrec
import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os

from timeit import default_timer as timer


def masktomo(num_rays,xp,width=.65):
    
    xx=xp.array([range(-num_rays//2, num_rays//2)])
    msk_sino=np.float32(np.abs(xx)<(num_rays//2*width))
    msk_sino.shape=(1,1,num_rays)
    
    xx=xx**2
    rr2=xx+xx.T
    msk_tomo=rr2<(num_rays//2*width*1.02)**2
    msk_tomo.shape=(1,num_rays,num_rays)
    return msk_tomo, msk_sino

def sirtMcalc(radon,radont,shape,xp):
    # compute the tomogram and sinogram of all ones
    # if R (radon) is a matrix sum_rows R = R*1= radon(1) 
    #                  sum_cols R = RT*1 = radont(1) 
    # we need to clip outside a mask of interest
    
    num_rays=shape[2]
    num_angles=shape[1]
    
    msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
    
    eps=xp.finfo(xp.float32).eps
    #t1=tomo0*0+1.
    S1=radon(np.ones((1,num_rays,num_rays),dtype='float32'))
    S1=1./np.clip(S1,eps*10,None)*msk_sino
    T1=radont(np.ones((1,num_angles,1),dtype='float32')*msk_sino)
    T1=1./np.clip(T1,eps,None)*msk_tomo
    return T1,S1


def sirt(radon, radont, sino_data, max_iter=30, factor=1, verbose=0):

    xp=np
    #image = np.zeros((sino_data.shape[0], sino_data.shape[1], sino_data.shape[2]))
    
    
#    if gpu_accelerated == False:
#        mode = "python"
#        xp = __import__("numpy")
        
    
    nrm0 = np.linalg.norm(sino_data)

    # prepare the left and right matrix 
    
    T1,S1=sirtMcalc(radon,radont,xp.shape(sino_data),xp)

    radont1= lambda x: T1*radont(S1*x)

#    radont1=radont
    
    tomo = radont1(sino_data)
    
    #print("image", image.shape)

    for i in range(max_iter):
        

        sino_sim = radon(tomo)
        
        residual = sino_data - sino_sim
        rnrm=np.linalg.norm(residual)/nrm0
        
        if verbose >0:
            title = "SIRT iter=%d, rnrm=%g" %(i, rnrm)
            print(title )
            if verbose >1:
                plt.imshow(t2i(tomo))
                plt.title(title)
                plt.show()

                
        
        step = radont1(residual)
        #step = step[:,:step.shape[1],:step.shape[2]]
        step *= factor
        
        tomo +=  step
        
        

#    return image[sino_data.shape[0]//2]
    return tomo,rnrm

xp=np
scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

from testing_setup import setup_tomo

size = 64*2
num_slices = size//2
num_angles = 27
num_rays   = size
# tomo to image
t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
# vector to tomo
v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))


radon, iradon, radont, true_obj, data, theta=setup_tomo(num_slices, num_angles, num_rays, xp)


#
#xp=np
#k_r = 1
#size = 64*2
#num_slices = size
#num_angles = size//2
#num_rays   = size
#
#t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
#
#base_folder = gridrec.create_unique_folder("shepp_logan")
#
#true_obj_shape = (num_slices, num_rays, num_rays)
#true_obj = gridrec.generate_Shepp_Logan(true_obj_shape)
#
#pad_1D        = size//2
#padding_array = ((0, 0), (pad_1D, pad_1D), (pad_1D, pad_1D))
#num_rays      = num_rays + pad_1D*2
#
#print("True obj shape", true_obj.shape)
#
#true_obj = np.lib.pad(true_obj, padding_array, 'constant', constant_values = 0)
#theta    = np.arange(0, 180., 180. / num_angles)*np.pi/180.
#
#t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real
#
##data = gridrec.forward_project(true_obj, theta)
##data = np.swapaxes(data,0,1)
#
##xp              = __import__("numpy")
##mode            = 'python'
#kernel_type     = "gaussian"
#gpu_accelerated = False
#
#radon,iradon,radont = gridrec.radon_setup(num_rays, theta, center=num_rays//2, xp=np, kernel_type = 'gaussian', k_r =1)
#
#data = radon(true_obj)

tomo0=iradon(data)

#scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
#factor=scale(tomo0,true_obj)
#
#msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
#
#T1,S1=sirtMcalc(radon,radont,msk_sino,msk_tomo,xp.shape(data),xp)
##
#radont1= lambda x: T1*radont(S1*x)
 
 
#image = sirt(data, theta, num_rays, k_r, kernel_type, gpu_accelerated, 100, factor=factor)
start = timer()
tomo_sirt,rnrm = sirt(radon, radont, data, max_iter=30, factor=1,verbose=0)
end = timer()
time_sirt=end-start
print("sirt data time=", time_sirt, "rnrm",rnrm)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
scalem   = lambda x,y: xp.max(y)/xp.max(x *x)

ssirt=scalem(t2i(tomo_sirt),t2i(tomo0))
plt.imshow(np.concatenate((t2i(tomo0),ssirt*t2i(tomo_sirt)),axis=1))
plt.title('iradon vs sirt')
plt.show()

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

print("sirt    time=", time_sirt, "snr=", ssnr(true_obj,tomo_sirt))




#plt.imshow(np.concatenate((t2i(tomo0),t2i(tomo_sirt)),axis=1))
#plt.title('iradon vs sirt')

