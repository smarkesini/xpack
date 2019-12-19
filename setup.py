#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:47:12 2019

@author: anu
"""

import os
import sys
from setuptools import find_packages

current_python = sys.version_info[:2]
required_python = (3, 7)

if current_python < required_python:
    sys.stderr.write('This version of xpack requires Python {}.{}, but you are trying to install it on Python {}.{}.'.format(*(required_python + current_python)))
    sys.exit(1)

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()
    
print("Found packages")
print(find_packages())

setup_info = dict(
        name = 'xpack',
        version = '0.1',
        packages = ['xpack'],
        python_requires = '>={}.{}'.format(*required_python),
        
        description = 'Distributed heterogeneous iterative solver for tomography.',
        long_description = read(README.md),
        
        url = 'https://github.com/smarkesini/xpack',        
        author = 'Stefano Marchesini',
        author_email = 'smarchesini@sigray.com',
        
        install_requires = ['numpy >= 1.15.0', 'scipy >= 1.3.1'],
        
        extras_require = {
                'cupy': ['cupy >= 7.0'],
                'mpi4py': ['mpi4py'],
                'h5py': ['h5py'],
                'hdf5-parallel': ['hdf5-parallel'],
                'tifffile': ['tifffile'],
                'tomopy': ['tomopy'],
                'astra': ['astra-toolbox']
                },
        
        classifiers = [
                # put in some classifiers
                ]
        )    




