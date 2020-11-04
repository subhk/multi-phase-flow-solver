import os
import numpy as np

def _lerp(a, b, s):
    return a * (1 - s) + b * s

class tools(object):

    def __init__(self, solver, grid):

        self.solver = solver
        self.grid = grid
        self.u = solver.u
        self.w = solver.w
    
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
        if low[0] > self.u.shape[0]-2: low[0] = self.u.shape[0]-2
        if low[1] < 0: low[1] = 0
        if low[1] > self.u.shape[1]-2: low[1] = self.u.shape[1]-2
        frac = pos - low

        left  = _lerp( self.u[low[0],   low[1]], self.u[low[0],   low[1]+1], frac[1] )
        right = _lerp( self.u[low[0]+1, low[1]], self.u[low[0]+1, low[1]+1], frac[1] )

        return _lerp(left, right, frac[0])

    
    def _interpolate_w(self, pos):
        """
        Return the w-velocity at the given position in the domain by using
        bilinear interpolation.
        """
        min_coord = np.array( [ self.grid.x.min(), self.grid.z.min() ]  )
        max_coord = np.array( [ self.grid.x.max(), self.grid.z.max() ]  )

        pos = (pos - min_coord)/(max_coord - min_coord)
        pos = np.array([pos[0] * self.grid.sze[0] + 0.5, pos[1] * self.grid.sze[1]])
        low = np.array(pos, int)

        if low[0] < 0: low[0] = 0
        if low[0] > self.w.shape[0]-2: low[0] = self.w.shape[0]-2
        if low[1] < 0: low[1] = 0
        if low[1] > self.w.shape[1]-2: low[1] = self.w.shape[1]-2
        frac = pos - low

        left  = _lerp( self.w[low[0],   low[1]], self.w[low[0],   low[1]+1], frac[1] )
        right = _lerp( self.w[low[0]+1, low[1]], self.w[low[0]+1, low[1]+1], frac[1] )

        return _lerp(left, right, frac[0])

    
    def _interpolate_velocity(self, pos):
        return np.array( [self._interpolate_u(pos), self._interpolate_w(pos)] )

    
    def avg_velocity(self, middle=True):
        """
        Calculate the velocity in the middle of the cells using linear
        interpolation if 'middle' is true. If 'middle' is false, the
        average velocity is calculated on grid cell corners.
        """

        if middle:
            u_avg = 0.5 * ( self.u[:-1,1:-1] + self.u[1:,1:-1] )
            w_avg = 0.5 * ( self.w[1:-1,:-1] + self.w[1:-1,1:] )
        else:
            u_avg = 0.5 * ( self.u[:,1:] + self.u[:,:-1] )
            w_avg = 0.5 * ( self.w[1:,:] + self.w[:-1,:] )

        return (u_avg, w_avg)


    def _cal_divergence(self):
        """
        Return the divergence field. The nodes coincide with pressure nodes.
        """
        d = self.grid.d
        div = np.zeros(self.p.shape, float)

        div[1:-1,1:-1] = (self.u[1:,1:-1] - self.u[:-1,1:-1])/d[0] \
                        + (self.w[1:-1,1:] - self.w[1:-1,:-1])/d[1]

        return div

    
    def _cal_vorticity(self):
        """
        Return the vorticity or curl field. The nodes are located
        at the grid cell corners.
        """
        d = self.grid.d
        
        vor = (self.w[1:,:] - self.w[:-1,:])/d[0] \
                - (self.u[:,1:] - self.u[:,:-1])/d[1]
        
        return vor

    
    def _cal_streamfunction(self):
        """
        Return the stream function field. The nodes are located
        at the grid cell corners.
        """
        psi = np.zeros(self.grid.shape, float)
        
        mask = (self.solver.mask[:-1,1:-1] | self.solver.mask[1:,1:-1]) & 1

        psi[1:,0] = -(w[1:-1,0] * self.grid.d[0]).cumsum(0)
        psi[:,1:] = mask * u[:,1:-1] * self.grid.d[1]
        psi = psi.cumsum(1)
        
        return psi

    
