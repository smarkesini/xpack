import numpy as np
from timeit import default_timer as timer
#import time

def wrap(sshape,theta,rot_center,algo,xp=np, obj_width=.98, max_iter=10, tol=1e-3, reg=1., tau=0.05, ncore=64, verbose=1, Positivity = True):   

    num_rays=sshape[2]
    if xp.__name__=='cupy': 
        GPU=True
    else:     GPU=False


    if algo=='tomopy-gridrec':
        import tomopy
        rnrm=None
        def reconstruct(data,verbose):
            start1 = timer()
            tomo_t = tomopy.recon(data, theta, center=rot_center, sinogram_order=True, algorithm="gridrec",ncore=ncore)
            t=timer()-start1
            return tomo_t, rnrm, t
    elif algo=='tomopy-sirt':
        import tomopy
        rnrm=None
        def reconstruct(data,verbose):
            tomo_t = tomopy.recon(data, theta, center=rot_center, sinogram_order=True, algorithm="sirt", num_iter=max_iter, ncore=ncore)
            return tomo_t, rnrm, 0.
    elif algo[0:5] =='astra':
        #print('using astra')
        import tomopy
        #import astra
        rnrm=None
        number_of_iterations= max_iter
        options = {'method': 'CGLS', 'num_iter': int(number_of_iterations)}
        if len(algo)==10:
            if algo[6:10]=='cuda':
                options = {'proj_type': 'cuda', 'method': 'CGLS_CUDA', 'num_iter': int(number_of_iterations)}
        #print('    Doing reconstruction...')
        def reconstruct(data,verbose):
#            tomo_t = tomopy.recon(np.swapaxes(data,0,1), theta, center=rot_center, sinogram_order=True, options=options, algorithm=tomopy.astra)
#            return np.swapaxes(tomo_t,0,1), rnrm, 0.
            #print("test: astra using ncore")
            tomo_t = tomopy.recon(data, theta, center=rot_center, sinogram_order=True, options=options, algorithm=tomopy.astra,ncore=ncore)
            return tomo_t, rnrm, 0.

        
    else:
        from .fubini import radon_setup as radon_setup
        if algo=='iradon' or algo=='iradon':
            
            iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width,iradon_only=True)
            rnrm=0
            def reconstruct(data,verbose):
                tomo_t=iradon(data)
                if GPU:
                    start1 = timer()
                    tomo= xp.asnumpy(tomo_t)
                    t=timer()-start1
                    return tomo,rnrm,t
                else: return tomo_t,None,0.
                
                
        elif algo == 'sirt' or algo == 'SIRT':
    
            radon,iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
    
            #import solve_sirt 
            from .solve_sirt import init 
            from .solve_sirt import sirtBB 
            init(xp)
            #solve_sirt.init(xp)
            #sirtBB=solve_sirt.sirtBB
            #t=0.
            def reconstruct(data,verbose):
                tomo_t,rnrm=sirtBB(radon, iradon, data, xp, max_iter=max_iter, alpha=1.,verbose=verbose,  Positivity = Positivity)
                if GPU:
                    start1 = timer()
                    tomo= xp.asnumpy(tomo_t)
                    t=timer()-start1
                    return tomo,rnrm,t
                else: return tomo_t,rnrm, 0.
            
        elif algo == 'CGLS' or algo == 'cgls':
            radon,iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
            #from solvers import solveTV
            from .solvers import solveCGLS

            def reconstruct(data,verbose):
                
                tomo_t, rnrm = solveCGLS(radon,iradon, data, x0=0, tol=tol, maxiter=max_iter, verbose=verbose)
                #tomo_t,rnrm = solveTV(radon, iradon, data, r, tau,  tol=1e-2, maxiter=10, verbose=verbose)
                if GPU:
                    start1 = timer()
                    tomo= xp.asnumpy(tomo_t)
                    t=timer()-start1
                    return tomo,rnrm,t
                else: return tomo_t,rnrm,0.

        elif algo == 'tv' or algo =='TV':
            #print("solving tv !!!!!!!!!!")
            algo = 'tv'
            radon,iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
            if tau==None: 
                tau=0.05
            if reg==None:
                reg=.8
            
            #print("τ=",tau, "reg",reg)
            #r = .8   
            #from solvers import Grad
            from .solvers import solveTV

            def reconstruct(data,verbose):
                tomo_t,rnrm = solveTV(radon, iradon, data, reg, tau,  tol=5e-3, maxiter=max_iter, verbose=verbose)
                if GPU:
                    start1 = timer()
                    tomo= xp.asnumpy(tomo_t)
                    t=timer()-start1
                    return tomo,rnrm,t
                else: return tomo_t,rnrm,0.

        elif algo == 'tvrings' or algo =='TVRINGS':
            algo = 'tvrings'
            radon,iradon = radon_setup(num_rays, theta, xp=xp, center=rot_center, filter_type='hamming', kernel_type = 'gaussian', k_r =1, width=obj_width)
            if tau==None: 
                tau=0.05
            if reg==None:
                reg=.8
            
            print("τ=",tau, "reg",reg)
            #r = .8   
            #from solvers import Grad
            from .solvers import solveTV_ring
            # set up the tv with missing pixels
            # fradon=lambda x: deadpix*radon(x)
            # fradont=lambda x: radont(x*deadpix)    
            #print("solving tv_rings")
            def reconstruct(data,verbose):
                tomo_t,rnrm = solveTV_ring(radon, iradon, data, reg, tau,  tol=tol, maxiter=max_iter, verbose=verbose)
                if GPU:
                    start1 = timer()
                    tomo= xp.asnumpy(tomo_t)
                    t=timer()-start1
                    return tomo,rnrm,t
                else: return tomo_t,rnrm,0.
        else:
            print('algorithm can be iradon, sirt, cgls, tv, tvrings, tomopy-gridrec, tomopy-sirt, astra, astra-cuda')
    
    return reconstruct
