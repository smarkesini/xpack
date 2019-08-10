#include "cupy/complex.cuh"



extern "C" __global__ void 
Convolve(thrust::complex< float >  * image, 
	thrust::complex< float >  * frames, 
	int * coord_x, 
	int * coord_y, 
	int k_r, 
        int n_angles,
        int n_rays,  
        int img_x, 
        int img_y)
{

        int tid = threadIdx.x + blockIdx.x*blockDim.x;
 
        if(tid >= n_rays*n_angles) return;


        int kernel_width = k_r * 2 + 1;

        int input_index = kernel_width * kernel_width * tid;

        int output_index = coord_y[tid] - k_r + (coord_x[tid] - k_r) * img_y;


	for(int x = 0; x < k_r*2 + 1; x++){

		for(int y = 0; y < k_r*2 + 1; y++){

 
			atomicAdd((float*)&(image[output_index + y]), frames[input_index + y].real());
			atomicAdd((float*)&(image[output_index + y]) + 1, frames[input_index + y].imag());

		}

                input_index += kernel_width;
                output_index += img_y;
	}			
}		
	

