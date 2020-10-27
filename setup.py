#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

poisson_ext = Extension('poisson', 
    ['poisson/colamd.c',
    'poisson/dasum.c',
    'poisson/daxpy.c',
    'poisson/dcolumn_bmod.c',
    'poisson/dcolumn_dfs.c',
    'poisson/dcopy.c',
    'poisson/dcopy_to_ucol.c',
    'poisson/dgscon.c',
    'poisson/dgsequ.c',
    'poisson/dgsrfs.c',
    'poisson/dgssvx.c',
    'poisson/dgstrf.c',
    'poisson/dgstrs.c',
    'poisson/dlacon.c',
    'poisson/dlamch.c',
    'poisson/dlangs.c',
    'poisson/dlaqgs.c',
    'poisson/dmemory.c',
    'poisson/dmyblas2.c',
    'poisson/dpanel_bmod.c',
    'poisson/dpanel_dfs.c',
    'poisson/dpivotgrowth.c',
    'poisson/dpivotL.c',
    'poisson/dpruneL.c',
    'poisson/dsnode_bmod.c',
    'poisson/dsnode_dfs.c',
    'poisson/dsp_blas2.c',
    'poisson/dsp_blas3.c',
    'poisson/dtrsv.c',
    'poisson/dutil.c',
    'poisson/get_perm_c.c',
    'poisson/heap_relax_snode.c',
    'poisson/idamax.c',
    'poisson/lsame.c',
    'poisson/memory.c',
    'poisson/mmd.c',
    'poisson/poisson.cpp',
    'poisson/relax_snode.c',
    'poisson/sp_coletree.c',
    'poisson/sp_ienv.c',
    'poisson/sp_preorder.c',
    'poisson/superlu_timer.c',
    'poisson/util.c',
    'poisson/xerbla.c'], 
	include_dirs=[numpy.get_include()], 
	define_macros=[('NO_TIMER', None)])

vof_ext = Extension('vof', ['vof/main.cpp', 'vof/vof.cpp'], 
    include_dirs=[numpy.get_include()])

setup(name='mpfs', 
    version='1.0',
    author='Subhajit Kar',
    author_email='subhajitkar19@gmail.com',
    description="A framework for solving 2D Navier-Stokes equation.",
    url='https://github.com/subhk/multi-phase-flow-solver',
    py_modules=['kmkns.NavierStokes', 
		'kmkns.Keyboard', 
		'kmkns.csf'],
    ext_package='kmkns',
    ext_modules=[poisson_ext, vof_ext],
    )

