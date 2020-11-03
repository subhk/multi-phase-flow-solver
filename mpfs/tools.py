import os
import numpy as np

def _lerp(a, b, s):
    return a * (1 - s) + b * s

class tools(object):

    def __init__(self, solver, grid):

        self.solver = solver
        self.grid = grid
    

    def _interpolate_u(self, pos):
        """
        Return the u-velocity at the given position in the domain by using
        bilinear interpolation.
        """
        min_coord = np.array( [ self.grid.x.min(), self.grid.z.min() ]  )
        max_coord = np.array( [ self.grid.x.max(), self.grid.z.max() ]  )

        pos = ( pos - min_coord)/(max_coord - min_coord )
        pos = np.array( [pos[0] * self.grid.sze[0], pos[1] * self.grid.sze[1] + 0.5] )
        low = np.array( pos, int )

        if low[0] < 0: low[0] = 0
        if low[0] > self.solver.u.shape[0]-2: low[0] = self.solver.u.shape[0]-2
        if low[1] < 0: low[1] = 0
        if low[1] > self.solver.u.shape[1]-2: low[1] = self.solver.u.shape[1]-2
        frac = pos - low

        left  = _lerp( self.solver.u[low[0],   low[1]], self.solver.u[low[0],   low[1]+1], frac[1] )
        right = _lerp( self.solver.u[low[0]+1, low[1]], self.solver.u[low[0]+1, low[1]+1], frac[1] )

        return _lerp(left, right, frac[0])

    
    def _interpolate_v(self, pos):
        """
        Return the v-velocity at the given position in the domain by using
        bilinear interpolation.
        """
        min_coord = np.array( [ self.grid.x.min(), self.grid.z.min() ]  )
        max_coord = np.array( [ self.grid.x.max(), self.grid.z.max() ]  )

        pos = (pos - min_coord)/(max_coord - min_coord)
        pos = np.array([pos[0] * self.grid.sze[0] + 0.5, pos[1] * self.grid.sze[1]])
        low = np.array(pos, int)

        if low[0] < 0: low[0] = 0
        if low[0] > self.v.shape[0]-2: low[0] = self.v.shape[0]-2
        if low[1] < 0: low[1] = 0
        if low[1] > self.v.shape[1]-2: low[1] = self.v.shape[1]-2
        frac = pos - low

        left  = _lerp( self.solver.v[low[0],   low[1]], self.solver.v[low[0],   low[1]+1], frac[1] )
        right = _lerp( self.solver.v[low[0]+1, low[1]], self.solver.v[low[0]+1, low[1]+1], frac[1] )

        return _lerp(left, right, frac[0])

    
    def _interpolate_velocity(self, pos):
        return np.array( [self._interpolate_u(pos), self._interpolate_v(pos)] )

    
    def avg_velocity(self, middle=True):
        """
        Calculate the velocity in the middle of the cells using linear
        interpolation if 'middle' is true. If 'middle' is false, the
        average velocity is calculated on grid cell corners.
        """
        if middle:
            u_avg = 0.5 * ( self.solver.u[:-1,1:-1] + self.solver.u[1:,1:-1] )
            v_avg = 0.5 * ( self.solver.v[1:-1,:-1] + self.solver.v[1:-1,1:] )
        else:
            u_avg = 0.5 * ( self.solver.u[:,1:] + self.solver.u[:,:-1] )
            v_avg = 0.5 * ( self.solver.v[1:,:] + self.solver.v[:-1,:] )

        return (u_avg, v_avg)


    def divergence(self):
        """
        Return the divergence field. The nodes coincide with pressure nodes.
        """
        d = self.grid.d
        div = np.zeros(self.p.shape, float)
        u = self.solver.u
        v = self.solver.v
        div[1:-1,1:-1] = (u[1:,1:-1]-u[:-1,1:-1])/delta[0] + (v[1:-1,1:]-v[1:-1,:-1])/delta[1]
        return f