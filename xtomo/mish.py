import numpy as np

bold='\033[1m'
endb= '\033[0m'

def compare_tomo(tomo,true_obj,chunks):
    
    if type(true_obj) == type(None): 
        print("no tomogram to compare")
        return None
    
    num_rays=tomo.shape[2]
    from .fubini import masktomo
    msk_tomo,msk_sino=masktomo(num_rays, np, width=.95)
    
    print('\r**norm*****'+'*'*60)
    nrm=np.linalg.norm(np.reshape(tomo,(-1)))
    print('tomo norm',nrm)
    print('\r**norm done'+'*'*60)
    
    #scale   = lambda x,y: np.dot(x.ravel(), y.ravel())/np.linalg.norm(x)**2
    scale   = lambda x,y: np.dot(np.reshape(x,(-1)), np.reshape(y,(-1)))/np.linalg.norm(x)**2
    #scale   = lambda x,y: np.dot(x.flat(), y.flat())/np.linalg.norm(x)**2
    rescale = lambda x,y: scale(x,y)*x
    ssnr   = lambda x,y: np.linalg.norm(y)/np.linalg.norm(y-rescale(x,y))
    #ssnr2    = lambda x,y: ssnr(x,y)**2
    
    print('\r snr ...', end=' ')
    snr=ssnr(true_obj,tomo)
    print('\r done computing',end=' ')

    print('\r',' '*50+'\r' +bold+"snr=", snr,endb,'\n')
    return snr, nrm
