"""
example code: solves the square cavity problem
with no-slip boundary condition except at the top
boundary where only horizontal velocity is specified.
"""

import sys, getopt
import numpy as np
import time

from mpfs.domain import Domain 
from mpfs.bc import bc_ns2d 
from mpfs.ns2d import NS2Dsolver 

import logging
#logger = logging.getLogger(__name__)


grid = Domain( [0, 1], [0, 1], [100, 100] )

print('grid generation is done!')

# setting up the boundary condition
bc_2d = bc_ns2d(grid)
bc_2d._set_bc_( 'left+right', u=0, v=0 )
bc_2d._set_bc_( 'down', u=0, v=0 )
bc_2d._set_bc_( 'up', u=0.1, v=0 )

# setting up the NS2D-solver
solver = NS2Dsolver( grid, bc_2d, nu=1.e-4 )
print('solver is done!')

# Integration parameters
solver.stop_sim_time = 10.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

print('I am getting inside the loop!')

# Main-loop:
try:
    logging.info('Starting loop')
    start_run_time = time.time()

    while solver.ok:

        print('I am here!')

        dt = solver.compute_cfl_dt_()
        solver.ns2d_simuation( dt )

        if (solver.iteration-1) % 10 == 0:
            logging.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

except:
    logging.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logging.info('Iterations: %i' %solver.iteration)
    logging.info('Sim end time: %f' %solver.sim_time)
    logging.info('Run time: %.2f sec' %(end_run_time - start_run_time))




