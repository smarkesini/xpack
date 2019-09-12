import radon
import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os



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


print("=========iradon with ramlak filter=========")



start=timer()
tomo0=iradon(data)
end = timer()
time_iradon=(end - start)

########################################
# plotting

tomo0c=t2i(tomo0)
truth=t2i(true_obj)

scaling_iradon=scale(tomo0c,truth)
#(np.sum(truth * tomowcgc))/np.sum(tomowcgc *tomowcgc)
snr_iradon  = ssnr(tomo0c,truth) 



#plt.imshow(t2i(tomo0))
#plt.title('iradon')
#plt.show()

nn=xp.ones((num_rays//2,1))*xp.nan
plt.imshow(np.concatenate((truth,nn,tomo0c*scaling_iradon),axis=1))
plt.title("truth vs iradon")
plt.show()


