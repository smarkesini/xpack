#!/usr/bin/env python

import subprocess
import operator
import os
import sys
import socket
#from .communicator import mpi_allGather, size
from .communicator import mpi_allGather, size, printv
import numpy as np
#from .misc import printd, printv


#Setup local GPU device based on a gpu_priority integer. The lower the priority, the more available gpu this assigns.
def set_visible_device(gpu_priority):

    cmd = "nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used,pstate,utilization.gpu --format=csv"
 
    result = str(subprocess.check_output(cmd.split(" ")))

    if sys.version_info <= (3, 0):
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split()[1:] # use this for python2
    else:
        result = result.replace(" ", "").replace("MiB", "").replace("%", "").split("\\n")[1:] # use this for python3


    result = [res.split(",") for res in result]
  
    if sys.version_info >= (3, 0):
        result = result[0:-1]

    result = [[int(r[0]), int(r[3])] for r in result]
    result = sorted(result, key=operator.itemgetter(1))

    n_devices = len(result)

    nvidia_device_order = [res[0] for res in result]

    visible_devices = str(nvidia_device_order[(gpu_priority % len(nvidia_device_order))])
      
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    return nvidia_device_order, visible_devices, n_devices


#not working, needs some extra work
def get_core_list(my_rank):

    pid = os.getpid()

    cmd = "awk '{print $39}' /proc/" + str(pid) + "/stat"

    core = str(subprocess.check_output(cmd))

    core_list = []

    core_list = mpi_allGather(core, mode = "python")

    return core_list


def get_rank_in_host(my_rank):

    hostname = socket.gethostname()

    hostname_list = []

    hostname_list = mpi_allGather(hostname, hostname_list, mode = "python")

    printv("List of node's hostnames for each MPI rank: [" + ", ".join(str(e) for e in hostname_list) + "]" )

    #this gives the index of the first hostname occurence on the list
    first_hostname = hostname_list.index(hostname)

    total_ranks_in_host = hostname_list.count(hostname)
    
    rank_in_host_index = my_rank - first_hostname

    return rank_in_host_index, total_ranks_in_host


def get_tile_distribution(my_rank, total_ranks, gpu_enabled, gpu_weight = 4):

    heterogeneous_computing = False
    tile_id = 0
    n_local_tiles = 1
    n_total_tiles = 1
    
    tiles_per_cpu = 1
    tiles_per_gpu = 1

    gpu_enabled_list = []

    gpu_enabled_list = mpi_allGather(gpu_enabled, gpu_enabled_list, mode = "python")

    if size > 1: printv("List of GPU-enabled flags for each MPI rank: [" + ", ".join(str(e) for e in gpu_enabled_list) + "]" )

    #If there is no heterogeneous computing all weights are 1
    if True in gpu_enabled_list and False in gpu_enabled_list:
        heterogeneous_computing = True
        tiles_per_gpu = gpu_weight

        weight_list = [ tiles_per_gpu if a is True else tiles_per_cpu for a in gpu_enabled_list]
    
        weight_array = np.array(weight_list)

        if my_rank == 0: 
            tile_id = 0
        else: 
            tile_id = np.sum(weight_array[:my_rank])

        n_total_tiles = np.sum(weight_array)

    else:

        tile_id = my_rank
        n_total_tiles = total_ranks

    if gpu_enabled: 
        n_local_tiles = tiles_per_gpu
    else:
        n_local_tiles = tiles_per_cpu

    return heterogeneous_computing, tile_id, n_local_tiles, n_total_tiles

def backend(GPU,rank):
     if GPU:
         try:
             
             from .devmanager import set_visible_device
             
             do,vd,nd=set_visible_device(rank)
 
             try:
                 import cupy as xp
                 #device_gbsize=xp.cuda.Device().mem_info[1]/((2**10)**3)
                 #printv("gpu memory:",device_gbsize, 
                 #       "GB, chunk sino memory:",max_chunk_slice*num_rays*num_angles*4/(2**10)**2,'MB',
                 #       ", chunk tomo memory:",max_chunk_slice*(num_rays**2)*4/(2**10)**2,'MB')
                 #xp.cuda.profiler.start()
             except:
                 pass
         except:
             xp=np
             GPU=False
     else:
         xp=np
     return xp, GPU
 
  