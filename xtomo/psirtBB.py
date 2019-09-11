import gridrec
import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os
import gridrec
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

def sirtMcalc(radon,radont,msk_sino,msk_tomo,shape,xp):
    
    
    num_rays=shape[2]
    num_angles=shape[1]
    
    eps=xp.finfo(xp.float32).eps
    #t1=tomo0*0+1.
    S1=radon(np.ones((1,num_rays,num_rays),dtype='float32'))
    S1=1./np.clip(S1,eps*10,None)*msk_sino
    T1=radont(np.ones((1,num_angles,1),dtype='float32')*msk_sino)
    T1=1./np.clip(T1,eps,None)*msk_tomo
    return T1,S1

def sirtBB(radon, iradon, sino_data, max_iter=30, alpha=1, verbose=0):

    #image = np.zeros((sino_data.shape[0], sino_data.shape[1], sino_data.shape[2]))
    xp=np
    
    if gpu_accelerated == False:
        mode = "python"
        xp = __import__("numpy")
        
    
    nrm0 = xp.linalg.norm(sino_data)
    tomo = iradon(sino_data)
    
    num_slices=np.shape(sino_data)[0]
    num_angles=np.shape(sino_data)[1]
    num_rays=np.shape(sino_data)[2]
    
    #print("image", image.shape)

    for i in range(max_iter):
        

        
        residual =  radon(tomo) - sino_data 
        rnrm=np.linalg.norm(residual)/nrm0
        
        if verbose >0:
            title = "SIRT iter=%d, rnrm=%g" %(i, rnrm)
            print(title )
            if verbose >1:
                plt.imshow(t2i(tomo))
                plt.title(title)
                plt.show()

       
        grad = iradon(residual)

        # BB step  (alternating)       
        if i>0:
            if np.mod(i,6)<3:
                alpha=xp.linalg.norm(tomo-tomo_old)**2/xp.inner((tomo-tomo_old).ravel(),(grad-grad_old).ravel())
            else:
                alpha=xp.inner((tomo-tomo_old).ravel(),(grad-grad_old).ravel())/xp.linalg.norm(grad-grad_old)**2
#
        tomo_old=tomo+0
        
        
        tomo -=  grad*alpha
        
        grad_old=grad+0
        
        
        #plt.imshow(np.real(image[size//2]))
        #plt.imshow(t2i(tomo))
        #plt.show()

#    return image[sino_data.shape[0]//2]
    return tomo,rnrm

xp=np
k_r = 1
size = 64
num_slices = size
num_angles = 25 # size//2
num_rays   = size

t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real

base_folder = gridrec.create_unique_folder("shepp_logan")

true_obj_shape = (num_slices, num_rays, num_rays)
true_obj = gridrec.generate_Shepp_Logan(true_obj_shape)

pad_1D        = size//2
padding_array = ((0, 0), (pad_1D, pad_1D), (pad_1D, pad_1D))
num_rays      = num_rays + pad_1D*2

print("True obj shape", true_obj.shape)

true_obj = np.lib.pad(true_obj, padding_array, 'constant', constant_values = 0)
theta    = np.arange(0, 180., 180. / num_angles)*np.pi/180.

t2i = lambda x: x[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3].real

#simulation = gridrec.forward_project(true_obj, theta)
#simulation = np.swapaxes(simulation,0,1)

#xp              = __import__("numpy")
#mode            = 'python'
kernel_type     = "gaussian"
gpu_accelerated = False

radon,iradon,radont = gridrec.radon_setup(num_rays, theta, center=num_rays//2, xp=np, kernel_type = 'gaussian', k_r =1)

simulation = radon(true_obj)

tomo0=iradon(simulation)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
factor=scale(tomo0,true_obj)

msk_tomo,msk_sino=masktomo(num_rays,xp,width=.65)
T1,S1=sirtMcalc(radon,radont,msk_sino,msk_tomo,xp.shape(simulation),xp)

iradon1= lambda x: T1*radont(S1*x)
 
#image = sirt(simulation, theta, num_rays, k_r, kernel_type, gpu_accelerated, 100, factor=factor)
start = timer()
tomo_sirtBB,rnrm = sirtBB(radon, iradon1, simulation, max_iter=30, alpha=1.,verbose=0)
end = timer()
time_sirtBB=end-start

start = timer()
tomo_psirtBB,rnrmp = sirtBB(radon, iradon, simulation, max_iter=30, alpha=1.,verbose=0)
end = timer()
time_psirtBB=end-start

print("sirtBB  time=", time_sirtBB, "rnrm=", rnrm)
print("psirtBB time=", time_psirtBB, "rnrm=", rnrmp)

scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
scalem   = lambda x,y: xp.max(y)/xp.max(x *x)

ssirt=scalem(t2i(tomo_sirtBB),t2i(tomo0))
ssirtp=scalem(t2i(tomo_psirtBB),t2i(tomo0))

plt.imshow(np.concatenate((t2i(tomo0),ssirt*t2i(tomo_sirtBB),ssirtp*t2i(tomo_psirtBB)),axis=1))
plt.title('iradon vs sirt')
plt.show()


scale   = lambda x,y: xp.sum(x * y)/xp.sum(x *x)
rescale = lambda x,y: scale(x,y)*x
ssnr2   = lambda x,y: xp.sum(y**2)/xp.sum((y-rescale(x,y))**2)
ssnr    = lambda x,y: xp.sqrt(ssnr2(x,y))

print("sirtBB  time=", time_sirtBB, "snr=", ssnr(true_obj,tomo_sirtBB))
print("psirtBB time=", time_psirtBB, "snr=", ssnr(true_obj,tomo_psirtBB))