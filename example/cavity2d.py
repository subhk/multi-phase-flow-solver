"""
example code: solves the square cavity problem
with no-slip boundary condition except at the top
boundary where only horizontal velocity is specified.
"""

from numpy.lib.function_base import gradient
from mpfs.src import ns2d as nst
from mpfs.src import domain as de
from mpfs.src import bc

import numpy as np
import time
import logging
logger = logging.getLogger(__name__)


# Domain should be defined as:
# first argument  : domain size in x-direction: x ∈ [x_min, x_max]
# second argument : domain size in z-direction: z ∈ [z_min, z_max]
# third argument  : no. of grid points in x & z-direction.
grid = de.Domain( [0, 1], [0, 1], [100, 100] )

# setting up the boundary condition
bc_2d = bc.bc_ns2d(grid)
bc_2d._set_bc_( 'left+right', u=0, v=0 )
bc_2d._set_bc_( 'down', u=0, v=0 )
bc_2d._set_bc_( 'up', u=0.1, v=0 )

# setting up the NS2D-solver
solver = nst.NS2Dsolver( grid, bc_2d, ν=1.e-4 )

# Integration parameters
solver.stop_sim_time = 10.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf


# Main-loop:
try:
    logger.info('Starting loop')
    start_run_time = time.time()

    while solver.ok:

        dt = solver.compute_cfl_dt_()
        solver.ns2d_simuation( dt )

        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

        

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))




