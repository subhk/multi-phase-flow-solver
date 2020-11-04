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
from mpfs.writer import Writer 

import logging
log = logging.getLogger(__name__)


grid = Domain( [0, 1], [0, 1], [500, 500] )

# object to handle boundary condition for velocity, pressure,... etc
bc_2d = bc_ns2d(grid)

# setting up the NS2D-solver
solver = NS2Dsolver( grid, bc_2d, mu=1./1e3, rho=1. )
solver._set_bc_( 'left+right', u=0., v=0. )
solver._set_bc_( 'down', u=0., v=0. )
solver._set_bc_( 'up', u=-1., v=0. )

# Integration parameters
solver.stop_sim_time = 10.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf


# Main-loop:
try:
    print('Starting loop')
    start_run_time = time.time()

    snap = Writer(solver, grid)

    while solver.ok:

        dt = solver.compute_cfl_dt_()
        solver.ns2d_simuation( dt )

        snap.file_handler( 'snapshots', iter=20 )

        if (solver.iteration-1) % 10 == 0:
            print('solver.iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            print('Max KE: %e' %(solver.u.max()**2 + solver.w.max()**2))

except:
    log.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    print('Iterations: %i' %solver.iteration)
    print('Sim end time: %f' %solver.sim_time)
    print('Run time: %.2f sec' %(end_run_time - start_run_time))




