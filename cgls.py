#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:47:38 2019

@author: smarchesini
"""
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cgs as cgs

v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))

# we are solving min_x ||R x-data||^2
# the gradient w.r.t x is:  R.T R x -R.T data
# setting the gradient to 0:  R.T R x = R.T data    

# let's setup R.T R as an operator
# setting radont(radon(x)) as a linear operator


def RTR_setup(radon,radont,num_slices, num_rays):
    
    # reshape vector to tomogram 
    v2t= lambda x: xp.reshape(x,(num_slices, num_rays, num_rays))
    mradon2 = lambda x: xp.reshape(radont(radon(v2t(x))),(-1))
    
    # now let's setup the operator
    
    RRshape = (num_slices*num_rays*num_rays,num_slices*num_rays*num_rays)
    RTR = LinearOperator(RRshape, dtype='float32', matvec=mradon2, rmatvec=mradon2)
    return RTR


RTR=RTR_setup(radon,radont,num_slices,num_rays)
# R.T data  (as a long vector)
RTdata =np.reshape(radont(data),(-1))

# initial guess (as a vector)
tomo0=np.reshape(iradon(data),(-1))
tolerance=np.linalg.norm(tomo0)*1e-4

tomo0c=v2t(tomo0)[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
plt.imshow(tomo0c)
plt.title('iradon')
plt.show()


print("solving CG-LS")

start = timer()
#tomocg,info = cgs(RTR,RTdata,x0=tomo0,tol=tolerance) 
tomocg,info = cgs(RTR,RTdata,tol=tolerance) 
tomocg.shape=(num_slices,num_rays,num_rays)

end = timer()
cgls_time=(end - start)

tomocgc=tomocg[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
scalingcgls=(np.sum(tomo_stack0c * tomocgc))/np.sum(tomocgc *tomocgc)
snr_cgls  =np.sum(tomo_stack0c**2)/np.sum(abs(tomocgc*scalingcgls-tomo_stack0c)**2)

plt.imshow(tomocgc)
plt.title(' cgls')

plt.show()


########################################
# we can use the ramlak filter as a weight to get a better condition number
# we are solving min_x ||filter (R x-data)||^2
# wich is a weighted least squares problem. Setting the gradient to 0 gives:
# (R.T filter) R x = (R.T filter) data 
# and we have iradon = (R.T filter)


RTR=RTR_setup(radon,iradon,num_slices,num_rays)
# R.T data  (as a long vector)
RTdata =np.reshape(iradon(data),(-1))

# initial guess (as a vector)
tomo0=np.reshape(iradon(data),(-1))
tolerance=np.linalg.norm(tomo0)*1e-5


print("solving  CG-weighted LS")

start = timer()
#tomowcg,info = cgs(RTR,RTdata,x0=tomo0,tol=tolerance) 
tomowcg,info = cgs(RTR,RTdata,tol=tolerance) 
# reshape the output
tomowcg.shape=(num_slices,num_rays,num_rays)

end = timer()
wcgls_time=(end - start)

tomowcgc=tomowcg[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
scalingwcgls=(np.sum(tomo_stack0c * tomowcgc))/np.sum(tomowcgc *tomowcgc)
snr_wcgls  =np.sum(tomo_stack0c**2)/np.sum(abs(tomowcgc*scalingwcgls-tomo_stack0c)**2)

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



