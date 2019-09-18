from radon import generate_Shepp_Logan as generate_Shepp_Logan

def setup_tomo (num_slices, num_angles, num_rays, xp, k_r=1, kernel_type = 'gaussian'):

    num_rays=num_rays//2
    true_obj_shape = (num_slices, num_rays, num_rays)
    true_obj = generate_Shepp_Logan(true_obj_shape)
    
    
    pad_1D        = num_rays//2
    padding_array = ((0, 0), (pad_1D, pad_1D), (pad_1D, pad_1D))
    num_rays      = num_rays + pad_1D*2
    
    print("True obj shape", true_obj.shape)
    
    true_obj = xp.lib.pad(true_obj, padding_array, 'constant', constant_values = 0)
    theta    = xp.arange(0, 180., 180. / num_angles)*xp.pi/180.
    
    kernel_type     = "gaussian"
    gpu_accelerated = False
    
    ############################
    # generate data
    return true_obj,  theta
