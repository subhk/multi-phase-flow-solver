#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy as np

poisson_ext = Extension('matsolver', 
    ['matsolver/colamd.c',
    'matsolver/dasum.c',
    'matsolver/daxpy.c',
    'matsolver/dcolumn_bmod.c',
    'matsolver/dcolumn_dfs.c',
    'matsolver/dcopy.c',
    'matsolver/dcopy_to_ucol.c',
    'matsolver/dgscon.c',
    'matsolver/dgsequ.c',
    'matsolver/dgsrfs.c',
    'matsolver/dgssvx.c',
    'matsolver/dgstrf.c',
    'matsolver/dgstrs.c',
    'matsolver/dlacon.c',
    'matsolver/dlamch.c',
    'matsolver/dlangs.c',
    'matsolver/dlaqgs.c',
    'matsolver/dmemory.c',
    'matsolver/dmyblas2.c',
    'matsolver/dpanel_bmod.c',
    'matsolver/dpanel_dfs.c',
    'matsolver/dpivotgrowth.c',
    'matsolver/dpivotL.c',
    'matsolver/dpruneL.c',
    'matsolver/dsnode_bmod.c',
    'matsolver/dsnode_dfs.c',
    'matsolver/dsp_blas2.c',
    'matsolver/dsp_blas3.c',
    'matsolver/dtrsv.c',
    'matsolver/dutil.c',
    'matsolver/get_perm_c.c',
    'matsolver/heap_relax_snode.c',
    'matsolver/idamax.c',
    'matsolver/lsame.c',
    'matsolver/memory.c',
    'matsolver/mmd.c',
    'matsolver/poisson.cpp',
    'matsolver/relax_snode.c',
    'matsolver/sp_coletree.c',
    'matsolver/sp_ienv.c',
    'matsolver/sp_preorder.c',
    'matsolver/superlu_timer.c',
    'matsolver/util.c',
    'matsolver/xerbla.c'], 
	include_dirs=[np.get_include()], 
	define_macros=[('NO_TIMER', None)])

setup(name='mpfs', 
    version='1.0',
    author='Subhajit Kar',
    author_email='subhajitkar19@gmail.com',
    url='https://github.com/subhk/multi-phase-flow-solver/',
    py_modules=['mpfs.bc', 
	 	'mpfs.domain', 
	 	'mpfs.force',
        'mpfs.ns2d',
        'mpfs.tools',
        'mpfs.writer'],
    ext_package='mpfs',
    ext_modules=[poisson_ext],
    )

