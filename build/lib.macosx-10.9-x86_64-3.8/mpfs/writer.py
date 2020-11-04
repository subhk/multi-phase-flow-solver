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

    def file_handler(self, filename, iter=None):

        if iter is None:
            raise ValueError('Invalid file saves.')

        if self.solver.iteration == 0:
            u_tmp = []
            w_tmp = []
            p_tmp = []
            time = []
            strm_fun_tmp = []

        # velocity will be written on pressure nodes.
        if self.solver.iteration%iter == 0:
            
            util = utility(self.solver, self.grid)

            ( u_cell_center, w_cell_center ) = util.avg_velocity()
            strm_fun = util._cal_streamfunction()

            time.extend(self.solver.sim_time)
            u_tmp.extend(u_cell_center)
            w_tmp.extend(w_cell_center)
            p_tmp.extend(self.solver.p)
            strm_fun_tmp.extend(strm_fun)

        
        if self.solver.sim_time >= self.solver.stop_sim_time:

            u_tmp = np.array(u_tmp)
            w_tmp = np.array(w_tmp)
            p_tmp = np.array(p_tmp)
            time  = np.array(time)
            strm_fun_tmp = np.array(strm_fun_tmp)

            hf = h5py.File( filename + '.h5', 'w' )

            hf.create_dataset( 't', data=time )
            hf.create_dataset( 'u', data=u_tmp )
            hf.create_dataset( 'w', data=w_tmp )
            hf.create_dataset( 'p', data=p_tmp )
            hf.create_dataset( 'psi', data=strm_fun_tmp)
            
            hf.create_dataset( 'x', data=self.grid.x )
            hf.create_dataset( 'z', data=self.grid.z )

            if hf.__bool__():
                hf.close()
                print( 'done → writing ... ', filename + ',h5' )




        
        

