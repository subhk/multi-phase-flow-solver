import os
import h5py
import numpy as np

from .tools import utility 

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Writer(object):

    def __init__(self, solver, grid):

        self.solver = solver
        self.grid = grid


    def FileWrite(self, filename, u, w, p, psi, time):

        if self.solver.sim_time >= self.solver.stop_sim_time or \
            self.solver.iteration >= self.solver.stop_iteration:

            u     = np.array(u)
            w     = np.array(w)
            p     = np.array(p)
            time  = np.array(time)
            psi   = np.array(psi)

            hf = h5py.File( filename + '.h5', 'w' )

            hf.create_dataset( 't',   data=time )
            hf.create_dataset( 'u',   data=u )
            hf.create_dataset( 'w',   data=w )
            hf.create_dataset( 'p',   data=p )
            hf.create_dataset( 'psi', data=psi )
            
            x = np.linspace( self.grid.x_coord[0], self.grid.x_coord[1], self.grid.sze[0]+1 )
            z = np.linspace( self.grid.z_coord[0], self.grid.z_coord[1], self.grid.sze[1]+1 )

            hf.create_dataset( 'x', data=x )
            hf.create_dataset( 'z', data=z )

            if hf.__bool__():
                hf.close()
                print( 'done -> writing ... ', filename + ',h5' )




        
        

