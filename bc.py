"""
This file contains the boundary condition for the
velocities and pressure for NS2D solver
"""
from functools import partial
import numpy as np


class _bc_ns2d_(object):

    LEFT    = 'left'
    RIGHT   = 'right'
    UP      = 'up'
    DOWN    = 'down'

    def __init__(self, u, w, p, grid, **args):

        self._bc = {self.LEFT:{}, self.RIGHT:{}, self.UP:{}, self.DOWN:{}}
        

    def _set_bc_(self, bounadries, **args):
        """
        Set bonadry conditions for the computational domain.
        The 'bounadries' should be 'east', 'west', 'north', 'south', 
        or a combination with '+' sign, for exmaple: 'east'+'west'+'south'
        with a common boundary conditions. 
        The keywords should be 'u', 'w', 'p', 'χ', 
        The condition should be either constant, or can be function of
        (x,y,t).
        """

        for bounadry in bounadries.split('+'):

            if bounadry.self._bc:
                self._bc[bounadry].update(args)

            if 'p' in args:
                if bounadry == self.UP:
                    self.mask[:,-1] = 3
                elif bounadry == self.DOWN:
                    self.mask[:,0]= 3
                elif bounadry == self.LEFT:
                    self.mask[0,:] = 3
                elif bounadry == self.RIGHT:
                    self.mask[-1,:] = 3

            if 'w' in args:
                if bounadry in (self.UP, self.DOWN):
                    self._bc[bounadry]['dpdn'] = 0.

            if 'u' in args:
                if bounadry in (self.LEFT, self.RIGHT):
                    self._bc[bounadry]['dpdn'] = 0.

    
    def _cal_νΔu_(self, μ, ρ, dt):
        """
        calculate viscous force 
        """

        δu = np.zeros( self.u.shape, np.float )
        δw = np.zeros( self.w.shape, np.float )

        μ_avg = 0.25 * ( μ[1:,1:] + μ[1:,:-1] + μ[:-1,1:] + μ[:-1,:-1] )

        δ = self.grid.δ
        u, w = self.u, self.w

        # viscous terms in u-equation: (1/ρ)∂/∂x(2μ∂u/∂x) + (1/ρ)∂/∂y(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂x(2μ∂u/∂x):
        #  
        δu[1:-1,1:-1] += ( μ[2:-1,1:-1] * (u[2:,1:-1] - u[1:-1,1:-1] ) - \
                    μ[1:-2,1:-1] * ( u[1:-1,1:-1] - u[:-2,1:-1]) ) / \
                    ( 0.5 * (δ[0]**2) * ρ[1:-1,1:-1] )

        # ∂/∂y(μ(∂u/∂z+∂w/∂x)):
        δu[1:-1,1:-1] += ( μ_avg[:,1:] * ( (u[:,2:] - u[:,1:-1]) / (δ[1]**2) + \
                    (w[1:,1:] - w[:-1,1:]) / (δ[0] * δ[1]) ) - \
                    μ_avg[:,:-1] * ( (u[:,1:-1] - u[:,:-2]) / (δ[1]**2) + \
                    (w[1:,:-1] - w[:-1,:-1]) / (δ[0]*δ[1]) ) ) / \
                    ρ[:,1:-1]

        # // TODO need to evaulate the viscous terms it at the boundaries.
        
        # viscous terms in w-equation: (1/ρ)∂/∂y(2μ∂w/∂z) + (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        # (1/ρ)∂/∂y(2μ∂w/∂z):
        #
        δw[1:-1,1:-1] += (μ[1:-1,2:-1] * (w[1:-1,2:] - w[1:-1,1:-1]) - \
                        μ[1:-1,1:-2] * (w[1:-1,1:-1] - w[1:-1,:-2])) / \
                        (0.5 * (δ[1]**2) * ρ[1:-1,1:-1])

        # (1/ρ)∂/∂x(μ(∂u/∂z+∂w/∂x))
        δw[1:-1,1:-1] += (μ_avg[1:,:] * ( (w[2:,:] - w[1:-1,:])/(δ[0]**2) + \
                      (u[1:,1:] - u[1:,:-1]) / (δ[0]*δ[1]) ) - \
                      μ_avg[:-1,:] * ( (w[1:-1,:] - w[:-2,:]) / (δ[0]**2) + \
                      (u[:-1,1:] - u[:-1,:-1])/(δ[0]*δ[1]) ) ) / \
                      ρ[1:-1,:]

        # // TODO need to evaulate the viscous terms it at the boundaries.

        return δu * dt, δw * dt


    def _update_vel_bc_(self):

        u, w = self.u, self.w

        # top-boundary condition
        _bc_ = self._bc[self.UP]
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[:,-1] = w[:,-2] # extrapolate
    
        # TODO : require to handle 'w'-condition
        
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[:,-1] = u[:,-2] # extrapolate
        elif 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[0]):
                    node = self.grid[i, u.shape[1]-2]
                    u[i,-1] = 2. * fun_(node[0], node[1], self.time) - u[i,-2]
            else:
                u[:,-1] = 2. * fun_ - u[:,-2]

        # bottom-boundary condition
        _bc_ = self._bc[self.DOWN]
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[:,0] = w[:,1]
        
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[:,0] = u[:,1]
        elif 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[0]):
                    node = self.grid[i, 0]
                    u[i,0] = 2. * fun_(node[0], node[1], self.time) - u[i,1]
            else:
                u[:,0] = 2. * fun_ - u[:,1]

        # left-boundary condition
        _bc_ = self._bc[self.LEFT]
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[-1,:] = u[-2,:]
        
        if 'dwdn' in _bc_:
            if _bc_['dvdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[-1,:] = w[-2,:]
        elif 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(w.shape[1]):
                    node = self.grid[w.shape[0]-2, i]
                    w[-1,i] = 2. * fun_(node[0], node[1], self.time) - w[-2,i]
            else:
                w[-1,:] = 2. * fun_ - w[-2,:]

        # right-boundary condition
        _bc_ = self._bc[self.RIGHT]
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError, '∂u/∂n must be zero.'
            u[0,:] = u[1,:]
        
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError, '∂w/∂n must be zero.'
            w[0,:] = w[1,:]
        elif 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(v.shape[1]):
                    node = self.grid[0, i]
                    w[0,i] = 2. * fun_(node[0], node[1], self.time) - w[1,i]
            else:
                w[0,:] = 2. * fun_ - w[1,:]


    def _update_pressure_bc_(self):
        p = self.p

        # top-boundary condition
        _bc_ = self._bc[self.UP]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[:,-1] = p[:,-2]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self._grid[i-0.5, p.shape[1]-2]
                    p[i,-1] = 2. * fun_(node[0], node[1], self.time) - p[i,-2]
            else:
                p[:,-1] = 2. * fun_ - p[:,-2]

        # bottom-boundary condition
        _bc_ = self._bc[self.DOWN]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[:,0] = p[:,1]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, 0]
                    p[i,0] = 2. * fun_(node[0], node[1], self.time) - p[i,1]
            else:
                p[:,0] = 2. * fun_ - p[:,1]

        # left-boundary condition
        _bc_ = self._bc[self.LEFT]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[-1,:] = p[-2,:]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[p.shape[0]-2, i-0.5]
                    p[-1,i] = 2. * fun_(node[0], node[1], self.time) - p[-2,i]
            else:
                p[-1,:] = 2. * fun_ - p[-2,:]

        # right-boundary condition
        _bc_ = self._bc[self.RIGHT]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError, '∂p/∂n must be zero.'
            p[0,:] = p[1,:]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[0, i-0.5]
                    p[0,i] = 2 * fun_(node[0], node[1], self.time) - p[1,i]
            else:
                p[0,:] = 2. * fun_ - p[1,:]


    def _update_Ξ_bc_(self, Ξ, β ):

        p = self.p
        
        # top-boundary condition
        _bc_ = self._bc[self.UP]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, p.shape[1]-2]
                    Ξ[i,-1] = 2. * fun_(node[0], node[1], self.time) - β * (p[i,-2] + p[i,-1])
            else:
                Ξ[:,-1] = 2 * fun_ - β * (p[:,-2] + p[:,-1])
        
        # bottom-boundary condition
        _bc_ = self._bc[self.DOWN]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, 0]
                    Ξ[i,0] = 2. * fun_(node[0], node[1], self.time) - β * (p[i,1] + p[i,0])
            else:
                Ξ[:,0] = 2. * fun_ - β * (p[:,1] + p[:,0])        

        # left-boundary condition
        _bc_ = self._bc[self.LEFT]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[p.shape[0]-2, i-0.5]
                    Ξ[-1,i] = 2. * f(node[0], node[1], self.time) - β * (p[-2,i] + p[-1,i])
            else:
                Ξ[-1,:] = 2 * fun_ - β * (p[-2,:] + p[-1,:])        

        # right-boundary condition
        _bc_ = self._bc[self.RIGHT]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[0, i-0.5]
                    Ξ[0,i] = 2. * fun_(node[0], node[1], self.time) - β * (p[1,i] + p[0,i])
            else:
                Ξ[0,:] = 2 * fun_ - β * (p[1,:] + p[0,:])        
