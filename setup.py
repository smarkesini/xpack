#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
#from setuptools import find_packages
from setuptools import setup, find_packages

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
        name = 'xtomo',
        version = '0.1',
        packages = ['xtomo'],
        python_requires = '>={}.{}'.format(3,7),
        
        description = 'Distributed heterogeneous iterative solver for tomography.',
        long_description = read('README.md'),
        
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
            'Development Status :: 2 - Pre-Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research', 
            'License :: OSI Approved :: OSS License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3 :: Only',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics'
            'Topic :: Software Development :: Libraries :: Python Modules',
                ]
        )    

setup(**setup_info)


