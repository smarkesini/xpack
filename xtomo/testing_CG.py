import gridrec
import numpy as np
import matplotlib.pyplot as plt
#import tomopy
#import imageio
#import os

#from solvers import Grad
#from solvers import solveTV
from solvers import cgs
from solvers import cg

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


print("=========GC Least Square by CGS=========")


# we are solving min_x ||R x-data||^2
# the gradient w.r.t x is:  R.T R x -R.T data
# setting the gradient to 0:  R.T R x = R.T data    

# let's setup R.T R to act on a vector
#RTR = lambda x: radont(radon(v2t(x)))

RTR = lambda x: xp.reshape(radont(radon(v2t(x))),(-1))

# R.T data  (as a long vector)
RTdata =np.reshape(radont(data),(-1))
#RTdata =radont(data)
tolerance = 2e-2


print("solving CG-LS")
# initial guess (as a vector)
# tomo0=np.reshape(iradon(data),(-1))

start = timer()
tomocg,info, imax, resnrm = cgs(RTR, RTdata, x0=0, maxiter=100, tol=tolerance)
tomocg.shape=(num_slices,num_rays,num_rays)
end = timer()
cgls_time=(end - start)

tomocgc=t2i(tomocg)
truth=t2i(true_obj)
scalingcgls=scale(tomocgc,truth) #(np.sum(truth * tomocgc))/np.sum(tomocgc *tomocgc)
snr_cgls  = ssnr2(tomocgc,truth)


tomo0=iradon(data)
#tomo0c=v2t(tomo0)
plt.imshow(t2i(tomo0))
plt.title('iradon')
plt.show()

plt.imshow(tomocgc)
plt.title(' cgls')
plt.show()



########################################
# we can use the ramlak filter as a weight to get a better condition number
# we are solving min_x ||filter (R x-data)||^2
# wich is a weighted least squares problem. Setting the gradient to 0 gives:
# (R.T filter) R x = (R.T filter) data 
# and we have iradon = (R.T filter)



RTR = lambda x: xp.reshape(iradon(radon(v2t(x))),(-1))
RTdata =np.reshape(iradon(data),(-1))

print("solving  CG-weighted LS")

start = timer()
tomowcg,info, imax, resnrm = cg(RTR, RTdata, x0=0, maxiter=100, tol=tolerance)
# reshape the output
tomowcg.shape=(num_slices,num_rays,num_rays)
end = timer()

wcgls_time=(end - start)

tomowcgc=t2i(tomowcg)#[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
truth=t2i(true_obj)

scalingwcgls=scale(tomowcgc,truth)
#(np.sum(truth * tomowcgc))/np.sum(tomowcgc *tomowcgc)
snr_wcgls  = ssnr2(tomowcgc,truth) 
#np.sum(truth**2)/np.sum(abs(tomowcgc*scalingwcgls-truth)**2)

plt.imshow(tomowcgc)
plt.title('weighted cgls')
plt.show()


fig, axs = plt.subplots(1,3)
axs[0].imshow(t2i(tomocg))
axs[0].set_title('cg-ls')
axs[1].imshow(t2i(tomowcg))
axs[1].set_title('weighted-cg-ls')
axs[2].imshow(t2i(true_obj))
axs[2].set_title('truth')
plt.show()
    

print("cgls   rec time =%3.4g, \t snr=%3.3g"% ( cgls_time, snr_cgls))
print("wcgls   rec time =%3.4g, \t snr=%3.3g"% ( wcgls_time, snr_wcgls))



