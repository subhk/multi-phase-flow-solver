"""
example code: solves the square cavity problem
with no-slip boundary condition except at the top
boundary where only horizontal velocity is specified.
"""

from numpy.lib.function_base import gradient
from src.ns2d import NS2Dsolver
from src.domain import Domain
from src.bc import bc_ns2d


# Domain should be defined as:
# first argument  : domain size in x-direction: x ∈ [x_min, x_max]
# second argument : domain size in z-direction: z ∈ [z_min, z_max]
# third argument  : no. of grid points in x & z-direction.
grid = Domain( [0, 1], [0, 1], [100, 100] )

# setting up the boundary condition
bc_2d = bc_ns2d(grid)
bc_2d._set_bc_( 'left+right', u=0, v=0 )
bc_2d._set_bc_( 'down', u=0, v=0 )
bc_2d._set_bc_( 'up', u=0.1, v=0 )

# setting up the NS2D-solver
solver = NS2Dsolver( grid, bc_2d, ν=1.e-4 )

