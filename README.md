## Description:

Distributed heterogeneous iterative solver for tomography. 
Solvers are: iradon (non-iterative), sirt (with BB-step [1]), CGLS, TV (split bregman[2]), tvrings[4], 
tomopy-gridrec [5], tomopy-astra using cgls [6]

## Installation:
Clone the repo and enjoy!

## Requirements:
Python, numpy, either scipy (for CPU) or cupy (for GPU).

Optional: 
mpi4py for distributed jobs.
h5py (for reading/saving). 
tifffile (for saving).

For simulations: tomopy. 
To use tomopy-gridrec and tomopy-astra: tomopy and astra.

## Usage

From the command line: 
> python recon.py 

without any input it will generate a simulation (using tomopy) and start reconstructing.

> python recon.py -h 

for further information on the options. 
The following will reconstruct on 2 GPUs, using TV denoising, and write the output onto the same folder as the input.

>$ mpirun -n 2 recon.py -f file_in.h5 -o '*' --GPU 1 -a 'tv'

the input file is assumed in sinogram order, with data in \exchange\data and angles in \exchange\theta
rotation center and other parameters are optionals and described by using the -h flag.


From Python, given some projection data e.g.:
> from fubini import radon_setup

> radon, iradon = radon_setup(num_rays, theta, xp=xp, center=None,filter_type='hamming')

> tomogram=iradon(data)

set xp=numpy or xp=cupy for GPU. The data should be in sinogram order, e.g.:

> data=np.ascontiguousarray(np.swapaxes(data_normalized,0,1))

> num_rays=data.shape[0]

> num_angles=data.shape[1]

More advanced algorithms. e.g. for tv:

> from wrap_algorithms import wrap

> reconstruct=wrap(sino_shape,theta,rot_center,'tv' ,xp=np)

> tomogram = reconstruct(sinogram_stack, verbose)

If things don't fit in memory, 

> from loop_sino import recon

> tomo, times_loop= recon(sino, theta, algo = 'tv', ...)


## Bibliography

[1] J. Barzilai and J. Borwein. Two-point step size gradient method. IMA J. Numerical Analysis 8, 141–148, 1988

[2] Goldstein, Tom, and Stanley Osher. "The split Bregman method for L1-regularized problems." SIAM journal on imaging sciences 2.2 (2009): 323-343.

[3] Maia, F. R. N. C., MacDowell, A., Marchesini, S., Padmore, H. A., Parkinson, D. Y., Pien, J., ... & Yang, C. (2010, September). Compressive phase contrast tomography. In Image Reconstruction from Incomplete Data VI (Vol. 7800, p. 78000F). International Society for Optics and Photonics.

[4] Gürsoy D, De Carlo F, Xiao X, and Jacobsen C. Tomopy: a framework for the analysis of synchrotron tomographic data. Journal of Synchrotron Radiation, 21(5):1188–1193, 2014.

[5] W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers, “Fast and Flexible X-ray Tomography Using the ASTRA Toolbox”, Optics Express, 24(22), 25129-25147, (2016)
