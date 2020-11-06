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
from mpfs.tools import utility

import logging
log = logging.getLogger(__name__)

grid = Domain( [0, 1], [0, 1], [100, 100] )

# object to handle boundary condition for velocity, pressure,... etc
bc_2d = bc_ns2d(grid)

# setting up the NS2D-solver
solver = NS2Dsolver( grid, bc_2d, mu=1./1e3, rho=1. )
solver._set_bc_( 'left',  u=0,  w=0 )
solver._set_bc_( 'right', u=0,  w=0 )
solver._set_bc_( 'down',  u=0,  w=0 )
solver._set_bc_( 'up',    u=1, w=0 )

# Integration parameters
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = 50000

# save diagonstic variables
max_write = 400
u_sol = np.zeros( (max_write, solver.u.shape[0]-1, solver.u.shape[1]-2), dtype=np.float )
w_sol = np.zeros( (max_write, solver.w.shape[0]-2, solver.w.shape[1]-1), dtype=np.float )
p_sol = np.zeros( (max_write, solver.p.shape[0], solver.p.shape[1]), dtype=np.float )
strm_fun_sol = np.zeros( (max_write, solver.u.shape[0], solver.u.shape[1]-1), dtype=np.float )

Time = []

snap = Writer( solver, grid )
save_rate = 1000
util = utility( solver, grid )

# Main-loop:
try:
    print('Starting loop')
    start_run_time = time.time()

    cnt = 0
    while solver.ok:

        dt = solver.compute_cfl_dt_()
        solver.ns2d_simuation( dt )

        if solver.iteration%save_rate == 0:
            u_cell_center, w_cell_center = util.avg_velocity()
            strm_fun = util._cal_streamfunction()

            u_sol[cnt,:,:] = u_cell_center
            w_sol[cnt,:,:] = w_cell_center
            p_sol[cnt,:,:] = solver.p 
            strm_fun_sol[cnt,:,:] = strm_fun
            Time.append( solver.sim_time )

            cnt += 1

        snap.FileWrite('snapshots', u_sol, w_sol, p_sol, strm_fun_sol, Time)

        if (solver.iteration-1) % 10 == 0:
            print('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            print('Velocity variance: %e' %np.sqrt(solver.u[1:-1,1:-1].max()**2) )

except:
    log.error('Exception raised, triggering end of main loop.')
    raise
finally:
    print('Cnt = ', cnt)
    end_run_time = time.time()
    print('Iterations: %i' %solver.iteration)
    print('Sim end time: %f' %solver.sim_time)
    print('Run time: %.2f sec' %(end_run_time - start_run_time))




