import os
import h5py
import pathlib
import numpy as np
from collections import defaultdict
import re
import shutil
import uuid

from 

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Writer(object):

    def __init__(self, solver, grid):

        self.solver = solver
        self.grid = grid

    


    def file_handler(self, filename, iter=None):

        if iter is None:
            raise ValueError('Invalid file saves.')

        u_tmp = []
        w_tmp = []
        p_tmp = []
        time = []

        if self.solver.iteration%iter == 0:
            
            time.extend(self.solver.sim_time)
            u_tmp.extend(self.solver.u)
            w_tmp.extend(self.solver.w)
            p_tmp.extend(self.solver.p)

        
        if self.solver.sim_time >= self.solver.stop_sim_time:

            u_tmp = np.array(u_tmp)
            w_tmp = np.array(w_tmp)
            p_tmp = np.array(p_tmp)
            time  = np.array(time)

            hf = h5py.File( filename + '.h5', 'w' )

            hf.create_dataset( 't', data=time )
            hf.create_dataset( 'u', data=u_tmp )
            hf.create_dataset( 'w', data=w_tmp )
            hf.create_dataset( 'p', data=p_tmp )
            
            hf.create_dataset( 'x', data=self.grid.x )
            hf.create_dataset( 'z', data=self.grid.z )




        
        

