"""
example code: solves the square cavity problem
with no-slip boundary condition except at the top
boundary where only horizontal velocity is specified.
"""

import numpy as np
import time

from mpfs.domain import Domain 
from mpfs.bc import bc_ns2d 
from mpfs.ns2d import NS2Dsolver 

import logging
log = logging.getLogger(__name__)


grid = Domain( [0, 1], [0, 1], [200, 200] )

print('grid generation is done!')

# setting up the boundary condition
bc_2d = bc_ns2d(grid)

# setting up the NS2D-solver
solver = NS2Dsolver( grid, bc_2d, mu=1./1e3, rho=1. )
solver._set_bc_( 'left+right', u=0., v=0. )
solver._set_bc_( 'down', u=0., v=0. )
solver._set_bc_( 'up', u=-1., v=0. )

print('solver is done!')

# Integration parameters
solver.stop_sim_time = 10.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

print('I am getting inside the loop!')
log.info("Starting loop")

# Main-loop:
try:
    log.info('Starting loop')
    start_run_time = time.time()

    while solver.ok:

        #print('I am here!')

        dt = solver.compute_cfl_dt_()
        #print('dt = ', dt)
        solver.ns2d_simuation( dt )

        if (solver.iteration-1) % 10 == 0:
            print('solver.iteration: %i, Time: %e, dt: %e', (solver.iteration, solver.sim_time, dt))
            mid = solver.u.shape[1]/2
            print "Centre velocities (u):", solver.u[-1,mid-2:mid+3]

except:
    log.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    print('Iterations: %i' %solver.iteration)
    print('Sim end time: %f' %solver.sim_time)
    print('Run time: %.2f sec' %(end_run_time - start_run_time))




