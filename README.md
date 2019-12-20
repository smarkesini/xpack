## Description:

Distributed heterogeneous iterative solver for tomography. 
Solvers are: iradon (non-iterative), preconditioned sirt (with BB-step [1], Ram-Lak-Hamming preconditioner), CGLS (using CG-squared [2]), TV (split bregman[3]), tvrings[4], tomopy-gridrec [5a] [5b], tomopy-astra [6a] using cgls [6b].

## Installation:
Clone the repo and enjoy!

## Requirements:
Python (>=3.7), numpy (>=1.15.0), either scipy (>=1.3.1)  (for CPU) or [cupy >=7.0](https://docs-cupy.chainer.org/en/stable/index.html) (for GPU).

**Optional**:  
[mpi4py](https://mpi4py.readthedocs.io/en/stable/) for multicore and distributed jobs. It uses shared memory if the mpi framework supports it.  
h5py (for reading/saving, with preferred parallel version [hdf5-parallel](https://anaconda.org/Clawpack/hdf5-parallel)).  
[tifffile](https://pypi.org/project/tifffile/) (for saving).  

For simulations: [tomopy](https://tomopy.readthedocs.io/en/latest/)
To use tomopy-gridrec and tomopy-astra: tomopy and tomopy-[astra](https://www.astra-toolbox.com/).

## Usage

From the command line: 
> python recon.py 

without any input it will generate a simulation (using tomopy) and start reconstructing.

>$ python recon.py -h 

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

If things don't fit in memory: 

> from loop_sino import recon  
> tomo, times_loop= recon(sino, theta, algo = 'tv', ...)  

**Contributors:** S. Marchesini, Sigray Inc.; Anu Trivedi, Virginia Tech.; Pablo Enfedaque, LBNL


## Bibliography

[1] J. Barzilai and J. Borwein. Two-point step size gradient method. IMA J. Numerical Analysis 8, 141–148, 1988

[2] T. Goldstein and S. Osher. "The split Bregman method for L1-regularized problems." SIAM journal on imaging sciences 2.2 (2009): 323-343.

[3] Barrett, Richard, Michael W. Berry, Tony F. Chan, James Demmel, June Donato, Jack Dongarra, Victor Eijkhout, Roldan Pozo, Charles Romine, and Henk Van der Vorst. Templates for the solution of linear systems: building blocks for iterative methods. Vol. 43. Siam, 1994.

[4] Maia, F. R. N. C., MacDowell, A., Marchesini, S., Padmore, H. A., Parkinson, D. Y., Pien, J., ... & Yang, C. (2010, September). Compressive phase contrast tomography. In Image Reconstruction from Incomplete Data VI (Vol. 7800, p. 78000F). International Society for Optics and Photonics. (arxiv:1009.1380)[https://arxiv.org/abs/1009.1380]

[5b] Dowd BA, Campbell GH, Marr RB, Nagarkar VV, Tipnis SV, Axe L, and Siddons DP. Developments in synchrotron x-ray computed microtomography at the national synchrotron light source. In Proc. SPIE, volume 3772, 224–236. 1999.

[5a] Gürsoy D, De Carlo F, Xiao X, and Jacobsen C. Tomopy: a framework for the analysis of synchrotron tomographic data. Journal of Synchrotron Radiation, 21(5):1188–1193, 2014.

[6a] Pelt D, Gürsoy D, Palenstijn WJ, Sijbers J, De Carlo F, and Batenburg KJ. Integration of tomopy and the astra toolbox for advanced processing and reconstruction of tomographic synchrotron data. Journal of Synchrotron Radiation, 23(3):842–849, 2016.

[6b] W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers, “Fast and Flexible X-ray Tomography Using the ASTRA Toolbox”, Optics Express, 24(22), 25129-25147, (2016)
