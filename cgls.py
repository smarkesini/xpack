#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:47:38 2019

@author: smarchesini
"""


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
tomocg,info, imax, resnrm = cgs(RTR, RTdata, x0=0, maxit=100, tol=tolerance)
tomocg.shape=(num_slices,num_rays,num_rays)
end = timer()
cgls_time=(end - start)

tomocgc=t2i(tomocg)
scalingcgls=scale(tomocgc,tomo_stack0c) #(np.sum(tomo_stack0c * tomocgc))/np.sum(tomocgc *tomocgc)
snr_cgls  = ssnr2(tomocgc,tomo_stack0c)


tomo0=iradon(data)
#tomo0c=v2t(tomo0)
plt.imshow(t2i(tomo0c))
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
tomowcg,info, imax, resnrm = cgs(RTR, RTdata, x0=0, maxit=100, tol=tolerance)
# reshape the output
tomowcg.shape=(num_slices,num_rays,num_rays)
end = timer()

wcgls_time=(end - start)

tomowcgc=t2i(tomowcg)#[num_slices//2,num_rays//4:num_rays//4*3,num_rays//4:num_rays//4*3]
scalingwcgls=scale(tomowcgc,tomo_stack0c)
#(np.sum(tomo_stack0c * tomowcgc))/np.sum(tomowcgc *tomowcgc)
snr_wcgls  = ssnr2(tomowcgc,tomo_stack0c) 
#np.sum(tomo_stack0c**2)/np.sum(abs(tomowcgc*scalingwcgls-tomo_stack0c)**2)

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



