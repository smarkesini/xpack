## Description:

Xtomo provides high performance reconstructions of tomography data. 
Reference (Sparse Matrix-Based HPC Tomography [^1]) provides an overview of the strategy, performance, features and solvers.
Performance is achieved using a distributed (using mpi) and heterogeneous (multi-GPUs or multicore) iterative (or direct) solvers 
and non-uniform FFTs. The non-uniform FFT uses a gridding and inverse gridding operation performed by Sparse-Matrix Vector multiplication.
   
Solvers are: iradon (non-iterative, also known as gridrec), SIRT (preconditioned by e.g. Ram-Lak-Hamming and with BB-step [^2]), CGLS (using CG-squared [^3]), TV (split Bregman[^4]), tvrings[^5] to remove rings within the iteration, tomopy-gridrec [^5a],[^5b], tomopy-astra [^6a],[^6b]...

If you make use of the library for your publication, pleace cite [^1].


**Contributors:** S. Marchesini, SLAC; Anu Trivedi, Virginia Tech.; Pablo Enfedaque, LBNL

**Known Issues** Don't use an odd number of pixels in first dimension (orthogonal to the rotation axis). There is a bug whereby odd numbers give the wrong results. TV does not use halos, therefore the slice near the border of chunk are a bit corrupted. Assuming the regularization parameter is kept small, it is not a major problem.

**Possible enhancement (contribution welcome)**: half precision arithmetic, GPU streaming. Halos for TV regularization, other solvers, positivity constraints in SIRT-BB, TV and CG,   Fan beam geometry, and more.



## Installation:
Clone the repo and run:
```
cd xpack
pip install -e .
```


## Requirements:
Python (>=3.7), numpy (>=1.15.0), either scipy (>=1.3.1)  (for CPU) or [cupy >=7.0](https://docs-cupy.chainer.org/en/stable/index.html) (for GPU).

**Recommended**:  
[mpi4py](https://mpi4py.readthedocs.io/en/stable/) for multicore and distributed jobs. It uses shared memory if the mpi framework supports it and cuda aware mpi see [^7] (thanks to [@LeoFang](https://github.com/leofang)!). If using conda, the feedstock version has cuda-aware mpi support: https://github.com/conda-forge/openmpi-feedstock , https://github.com/conda-forge/mpi4py-feedstock. 

h5py (for reading/saving, with preferred parallel version [hdf5-parallel](https://anaconda.org/lcls-ii/hdf5-parallel).  
[tifffile](https://pypi.org/project/tifffile/) (for saving).  
 
[tomopy](https://tomopy.readthedocs.io/en/latest/) To use tomopy-gridrec and tomopy-astra: tomopy and tomopy-[astra](https://www.astra-toolbox.com/). Also [dxchange](https://dxchange.readthedocs.io/en/latest/source/install.html) for reading general tomography data.


**Optional**:  
[bm3d_streak](https://pypi.org/project/bm3d-streak-removal/) to pre-process data using [^10].



**Notes**
*Installation:* The order of installation should be (dxchange) Tomopy, cupy, mpi4py, then this package. Tomopy comes with its own libraries that override others.
*Cache:* the library saves the Sparse matrices or SpMV each time you change geometry (number of pixels, angles, rotation center, ...). To clear them, use xtomo.sparse_plan.clean_cache()


## Usage (command line)

(**First install** with e.g. pip see above).   

From the command line (terminal), data (from file) and reconstruction (to file) using (e.g. GPU and tv algorithm):

>$ python recon.py [-f input_file.h5] [-o output_file.h5] [--GPU 1] [-a 'tv']

you can use recon.py file in the main folder '[xpack/recon.py](https://github.com/smarkesini/xpack/blob/master/recon.py)', or just copy it to your path.

The input file is assumed in sinogram order, with data in '\exchange\data' and angles in '\exchange\theta'. 

For MPI use for example using 2 GPU, TV algorithm, saving to an auto-generated file name:

>$ mpirun -n 2 recon.py -f file_in.h5 -o '*' --GPU 1 -a 'tv'

If the input file is not given it will generate its own simulation (using tomopy) and save it to file. The simulation size is adjustable.   The output file name is auto-generated if not set from the command line. The file name can be '*.h5' or '*.tif'. 

The full list of options type:

>$ python recon.py -h 


```
usage: recon.py [-h] [-f FILE_IN] [-o FILE_OUT] [-rot_center ROT_CENTER] [-a ALGO] [-G GPU] [-S SHMEM] [-maxiter MAXITER] [-tol TOL] [-max_chunk MAX_CHUNK_SLICE] [-chunks CHUNKS [CHUNKS ...]] [-time_file TIME_FILE]
                [-reg REG] [-tau TAU] [-v VERBOSE] [-sim SIMULATE] [-sim_shape SIM_SHAPE [SIM_SHAPE ...]] [-sim_width SIM_WIDTH] [-opts OPTIONS] [-fopts FOPTIONS] [-ncore NCORE] [-rb RING_BUFFER]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE_IN, --file_in FILE_IN
                        h5 file in
  -o FILE_OUT, --file_out FILE_OUT
                        file out, default 0, 0: autogenerate name -tif, 0.tif or 0.h5; -1: skip saving
  -rot_center ROT_CENTER, --rot_center ROT_CENTER
                        rotation center, float 
  -a ALGO, --algo ALGO  algorithm: 'iradon' (default), 'sirt', 'cgls', 'tv', 'tvrings' 'tomopy-gridrec', 'tomopy-sirt', 'astra'
  -G GPU, --GPU GPU     turn on GPU, bool
  -S SHMEM, --shmem SHMEM
                        turn on shared memory MPI, bool
  -maxiter MAXITER, --maxiter MAXITER
                        maxiter, default 10
  -tol TOL, --tol TOL   tolerance, default 5e-3
  -max_chunk MAX_CHUNK_SLICE, --max_chunk_slice MAX_CHUNK_SLICE
                        max chunks per mpi rank
  -chunks CHUNKS [CHUNKS ...], --chunks CHUNKS [CHUNKS ...]
                        chunks to reconstruct: [number of slices around center | first and last slice]
  -time_file TIME_FILE, --time_file TIME_FILE
                        1: save timings to a txt file
  -reg REG, --reg REG   regularization parameter
  -tau TAU, --tau TAU   soft thresholding parameter
  -v VERBOSE, --verbose VERBOSE
                        verbose float between (0-1), default 1, a smaller value reduces outputs/loop
  -sim SIMULATE, --simulate SIMULATE
                        use simulated data, bool
  -sim_shape SIM_SHAPE [SIM_SHAPE ...], --sim_shape SIM_SHAPE [SIM_SHAPE ...]
                        simulate shape nslices,nangles,nrays
  -sim_width SIM_WIDTH, --sim_width SIM_WIDTH
                        object width within the FOV, between 0-1
  -opts OPTIONS, --options OPTIONS
                        e.g. '{"algo":"iradon", "maxiter":10, "tol":1e-2, "reg":1, "tau":.05} ' 
  -fopts FOPTIONS, --foptions FOPTIONS
                        file with json options  
  -ncore NCORE, --ncore NCORE
                        ncore for tomopy reconstruction algorithms
  -rb RING_BUFFER, --ring_buffer RING_BUFFER
                        ring buffer 0 none,1:input,2=output,4=MPI,3=1+2 (both), 7=1+2+4

+-------------------------------------------------------------+
| option precedence (highest on the left):                    |
| individual options, 'opts' dictionary, 'fopts' file options |
+-------------------------------------------------------------+

```
## Usage (Python)

From Python, the most general interface (using mpi, etc) can be used as (e.g. 2 mpi workers, using GPUs, iradon algorithm): 

>  import xtomo

>  tomo2=xtomo.recon(data,theta, rot_center, Dopts = None)

where the data should be in sinogram order, e.g.:

> num_rays=data.shape[0]  
> num_angles=data.shape[1]  
> num_slices=data.shape[2]  

theta is the angle in radians, rot_center is the position of the center of rotation, 
and Dopts is an optional set of parameters to define the type of solvers, parallelization, GPUs, etc. 
for example to use TV regularization on 2 GPUS:

>  Dopts={ 'algo':'TV', 'GPU': True, 'n_workers' : 2 }


See example '[examples/tomobank_rec.py](https://github.com/smarkesini/xpack/blob/master/xtomo/examples/tomobank_rec.py)', that will process tomo_00001 from [tomobank](https://tomobank.readthedocs.io/en/latest/) [^8] using stripe-removal from [^9] or [^10] for pre-processing. 


Dopts can be used to change various solvers, regularization parameters, chunking (to proces piece by piece), max_iterations,  GPU or CPU, number of mpi jobs spawned. Example options to apply TV regularization, with 2 GPUs, chunking 16 slices at a time:
 Dopts={ 'algo':'TV', 'GPU': True, 'n_workers' : 2 , 'reg': .02, 'max_chunk_slice': 16}


There are other interfaces to the solvers that don't use mpi, don't chunk the data (to fit in memory), or don't iterate. Essentially starting from (1) the core functions, (2) to the iterative solvers, (3) looping chunks of data, and mpi:

[1] The core one-step inverse radon transform using the SPmV operation can be used as follows:

> from xtomo.fubini import radon_setup  
> radon, iradon = radon_setup(num_rays, theta, xp=[cupy or numpy], center=None,filter_type='hamming')  
> tomogram=iradon(data)  

[2] Algorithms (sirt, cgls, tv, ...), e.g. for tv on GPU:

> from xtomo.wrap_algorithms import wrap  
> reconstruct=wrap(sino_shape,theta,rot_center,'tv' ,xp=cupy)  
> tomogram = reconstruct(sinogram_stack, verbose)  

[3] Chunking, if things don't fit in memory : 

> from xtomo.loop_sino import recon  
> tomo, times_loop= recon(sino, theta, algo = 'iradon', ...)  



## Contents

1. recon.py: reconstruct from file input to file output, parsing options

10. examples: examples using mpi, e.g. with [tomobank](https://tomobank.readthedocs.io/en/latest/) [^8] data. It calls [^9] for pre-processor.

14. prep: pre-process raw data

2. fubini.py: high performance CPU and GPU forward and backward operators  using non-uniform FFT.

3. solvers.py: iterative solvers.

4. solve_sirt.py: SIRT with BB-step.

5. wrap.py: wraps solvers such as non-iterative 'iradon'

6. loop_sino.py: process chunks of data.

7. IO.py: handles files, memory mapping.

8. communicator.py: mpi communication

9. devmanager: selects GPUs.

10. spawn.py: spawns mpi jobs.

11. sparse_plan.py: saves/load Sparse Matrix to cache.

12. fft.py: handles ffts (plans, GPU,...).

13. mpi_workers: handle spawned workers (working from files or numpy arrays)

 


## Bibliography
[^1]: S. Marchesini, A. Trivedi, P. Enfedaque, T. Perciano, D. Parkinson. Sparse Matrix-Based HPC Tomography, 
["Lecture Notes in Computer Science", vol 12137, 248--261, 2020.](https://doi.org/10.1007/978-3-030-50371-0_18)

[^2]: J. Barzilai and J. Borwein. Two-point step size gradient method. IMA J. Numerical Analysis 8, 141–148, 1988

[^3]: T. Goldstein and S. Osher. "The split Bregman method for L1-regularized problems." SIAM journal on imaging sciences 2.2 (2009): 323-343.

[^4]: Barrett, Richard, Michael W. Berry, Tony F. Chan, James Demmel, June Donato, Jack Dongarra, Victor Eijkhout, Roldan Pozo, Charles Romine, and Henk Van der Vorst. Templates for the solution of linear systems: building blocks for iterative methods. Vol. 43. Siam, 1994.

[^5]: Maia, F. R. N. C., MacDowell, A., Marchesini, S., Padmore, H. A., Parkinson, D. Y., Pien, J., ... & Yang, C. (2010, September). Compressive phase contrast tomography. In Image Reconstruction from Incomplete Data VI (Vol. 7800, p. 78000F). International Society for Optics and Photonics. [arxiv:1009.1380](https://arxiv.org/abs/1009.1380)

[^5b]: Dowd BA, Campbell GH, Marr RB, Nagarkar VV, Tipnis SV, Axe L, and Siddons DP. Developments in synchrotron x-ray computed microtomography at the national synchrotron light source. In Proc. SPIE, volume 3772, 224–236. 1999.


[^5a]: Gürsoy D, De Carlo F, Xiao X, and Jacobsen C. Tomopy: a framework for the analysis of synchrotron tomographic data. Journal of Synchrotron Radiation, 21(5):1188–1193, 2014.

[^6a]: Pelt D, Gürsoy D, Palenstijn WJ, Sijbers J, De Carlo F, and Batenburg KJ. Integration of tomopy and the astra toolbox for advanced processing and reconstruction of tomographic synchrotron data. Journal of Synchrotron Radiation, 23(3):842–849, 2016.


[^6b]: W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers, “Fast and Flexible X-ray Tomography Using the ASTRA Toolbox”, Optics Express, 24(22), 25129-25147, (2016)

[^7]: Dalcin, Lisandro, and Yao-Lung L. Fang. "mpi4py: Status Update After 12 Years of Development." Computing in Science & Engineering 23.4 (2021): 47-54.

[^8]: Francesco De Carlo, Doğa Gürsoy, Daniel J Ching, K Joost Batenburg, Wolfgang Ludwig, Lucia Mancini, Federica Marone, Rajmund Mokso, Daniël M Pelt, Jan Sijbers, and Mark Rivers. Tomobank: a tomographic data repository for computational x-ray science. Measurement Science and Technology, 29(3):034004, 2018. [URL](http://stacks.iop.org/0957-0233/29/i=3/a=034004).

[^9]: N. T. Vo,  R. C. Atwood, M. Drakopoulos,  Opt. Express, 26, 28396–28412 (2018). 

[^10]: [Y. Mäkinen,  S. Marchesini, A. Foi,  "Ring artifact reduction via multiscale nonlocal collaborative filtering of spatially correlated noise", J. Synchrotron Rad. 28(3), pages 876-888, 2021.](http://doi.org/10.1107/S1600577521001910)
