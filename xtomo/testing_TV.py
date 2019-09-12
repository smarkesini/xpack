#import gridrec
import radon

import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os

from solvers import Grad
from solvers import solveTV


from timeit import default_timer as timer

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
#num_rays=num_rays


print("=========TV regularization=========")
#
# reconstruct by iradon
start = timer()
tomo0=iradon(data)
end = timer()
time_iradon=(end - start)

plt.imshow(tomo0[num_slices//2,:,:])
plt.title("init")
plt.show()



# TV parameters selection
# let's scale the Radon trasform so that RT (data)~1 in the central slice
Rsf=1./np.mean(t2i(radont(data)))
data*=Rsf
# set up thresholding tau
p=Grad(tomo0)
#tau=0.008*np.max(np.abs(p))*Rsf # soft thresholding 
tau=0.05

print("τ=",tau)

r = .8     # regularization weight 



start=timer()
tomotv=solveTV(radon,radont, data, r, tau, x0=tomo0, tol=1e-2, maxiter=10, verbose=1)
end = timer()
time_TV=(end - start)



#tomotv=v2t(u)

#plt.imshow(t2i(tomotv))
nn=xp.ones((num_rays//2,1))*xp.nan
plt.imshow(np.concatenate((t2i(tomotv),nn,t2i(tomo0)),axis=1))
plt.title("TV vs iradon")
plt.show()

fig, axs = plt.subplots(1,3)


axs[0].imshow(t2i(tomo0))
axs[0].set_title('iradon')
axs[1].imshow(t2i(v2t(tomotv)))
axs[1].set_title('tv')
axs[2].imshow(t2i(true_obj))
axs[2].set_title('truth')
plt.show()
    

# cropped slice    
tomotvc=t2i(tomotv)
tomo0c=t2i(tomo0)
truth=t2i(true_obj)

##scalingtv=(np.sum(tomo0c * tomotvc))/np.sum(tomotvc *tomotvc)
#scalingtv=scale(tomotvc,truth)
#scaling0=scale(tomo0c,truth)

#snr_TV  =np.sum(tomo0c**2)/np.sum(abs(tomotvc*scalingtv-tomo0c)**2)
snr_TV  = ssnr2(tomotvc,truth)#np.sum(tomo0c**2)/np.sum(abs(tomotvc-tomo0c)**2)
snr_iradon  = ssnr2(tomo0c,truth)#np.sum(tomo0c**2)/np.sum(abs(tomotvc-tomo0c)**2)


print("TV     rec time =%3.4f, \t snr=%3.3g"% ( time_TV, snr_TV))
print("iradon rec time =%3.4f, \t snr=%3.3g"% ( time_iradon, snr_iradon))
