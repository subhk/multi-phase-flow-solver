import sys
import os
import time
import numpy as np
import h5py

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
import matplotlib as mpl

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
plt.rcParams.update({'font.size': 14})


def reading_hdf5(filename):
    
    hFile = h5py.File( filename + '.h5', mode='r' )
    
    t   = np.array( hFile['t'] )
    u   = np.array( hFile['u'] )
    w   = np.array( hFile['w'] )
    p   = np.array( hFile['p'] )
    psi = np.array( hFile['psi'] )

    x = np.array( hFile['x'] )
    z = np.array( hFile['z'] )
    
    if hFile.__bool__():
        hFile.close()
        print( 'done -> reading ...', filename + '.h5' )
    
    return t, u, w, p, psi, x, z


def Contouplot( var, x, z ):

    X, Z = np.meshgrid( x, z, indexing='ij' )

    fig, ax = plt.subplots()
    maxi = max( np.max(var), abs(np.min(var)) )

    # color contour plot
    cf = ax.contourf( X, Z, var, vmin=-maxi, vmax=maxi, levels=30, cmap=cm.bwr ) 
    fig.colorbar(cf, ax=ax)
    
    # line contour plot
    colours = ['w' if level<0 else 'k' for level in cf.levels]
    lc = ax.contour( X, Z, var, levels=30, colors=colours )
    ax.clabel(lc, fontsize=12, colors=colours)
    ax.set_title(r'$\psi$')
        
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$z$', fontsize=18)
    plt.show()


if __name__ == "__main__":

    t, u, w, p, psi, x, z = reading_hdf5('snapshots')

    print('files saved: ', len(t))
    
    snaps = 46
    var2plot = psi[ snaps,:,: ]
    print( 'max-var: ', np.max(abs(var2plot)) )

    Contouplot(var2plot, x[:], z[:])
