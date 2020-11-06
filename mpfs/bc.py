"""
This file contains the boundary condition for the
velocities and pressure for NS2D solver
"""
#from functools import partial
import numpy as np


class bc_ns2d(object):

    LEFT    = 'left'
    RIGHT   = 'right'
    UP      = 'up'
    DOWN    = 'down'

    def __init__(self, grid):

        self.grid = grid


    def _update_vel_bc_(self, u, w, mask, time, _bc):

        self._update_intermediate_vel_bc_(u, w, mask, time, _bc)

        # top-boundary condition
        _bc_ = _bc[self.UP]
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError('dw/dn must be zero.')
            w[:,-1] = w[:,-2] # extrapolate
    
        # done: TODO : require to handle 'w'-condition: it is done in 
        # "_update_intermediate_vel_bc_()"
        
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError('du/dn must be zero.') 
            u[:,-1] = u[:,-2] # extrapolate
        elif 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[0]):
                    node = self.grid[i, u.shape[1]-2]
                    u[i,-1] = 2. * fun_(node[0], node[1], time) - u[i,-2]
            else:
                u[:,-1] = 2. * fun_ - u[:,-2]

        # bottom-boundary condition
        _bc_ = _bc[self.DOWN]
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError('dw/dn must be zero.') 
            w[:,0] = w[:,1]
        
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError('du/dn must be zero.') 
            u[:,0] = u[:,1]
        elif 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[0]):
                    node = self.grid[i, 0]
                    u[i,0] = 2. * fun_(node[0], node[1], time) - u[i,1]
            else:
                u[:,0] = 2. * fun_ - u[:,1]

        # left-boundary condition
        _bc_ = _bc[self.LEFT]
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError('du/dn must be zero.') 
            u[-1,:] = u[-2,:]
        
        if 'dwdn' in _bc_:
            if _bc_['dvdn'] != 0:
                raise ValueError('dw/dn must be zero.') 
            w[-1,:] = w[-2,:]
        elif 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(w.shape[1]):
                    node = self.grid[w.shape[0]-2, i]
                    w[-1,i] = 2. * fun_(node[0], node[1], time) - w[-2,i]
            else:
                w[-1,:] = 2. * fun_ - w[-2,:]

        # right-boundary condition
        _bc_ = _bc[self.RIGHT]
        if 'dudn' in _bc_:
            if _bc_['dudn'] != 0:
                raise ValueError('du/dn must be zero.') 
            u[0,:] = u[1,:]
        
        if 'dwdn' in _bc_:
            if _bc_['dwdn'] != 0:
                raise ValueError('dw/dn must be zero.') 
            w[0,:] = w[1,:]
        elif 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(w.shape[1]):
                    node = self.grid[0, i]
                    w[0,i] = 2. * fun_(node[0], node[1], time) - w[1,i]
            else:
                w[0,:] = 2. * fun_ - w[1,:]


    def _update_pressure_bc_(self, p, time, _bc):
        #p = self.p

        # top-boundary condition
        _bc_ = _bc[self.UP]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError('dp/dn must be zero.') 
            p[:,-1] = p[:,-2]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, p.shape[1]-2]
                    p[i,-1] = 2. * fun_(node[0], node[1], time) - p[i,-2]
            else:
                p[:,-1] = 2. * fun_ - p[:,-2]

        # bottom-boundary condition
        _bc_ = _bc[self.DOWN]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError('dp/dn must be zero.') 
            p[:,0] = p[:,1]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, 0]
                    p[i,0] = 2. * fun_(node[0], node[1], time) - p[i,1]
            else:
                p[:,0] = 2. * fun_ - p[:,1]

        # left-boundary condition
        _bc_ = _bc[self.LEFT]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError('dp/dn must be zero.') 
            p[-1,:] = p[-2,:]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[p.shape[0]-2, i-0.5]
                    p[-1,i] = 2. * fun_(node[0], node[1], time) - p[-2,i]
            else:
                p[-1,:] = 2. * fun_ - p[-2,:]

        # right-boundary condition
        _bc_ = _bc[self.RIGHT]
        if 'dpdn' in _bc_:
            if _bc_['dpdn'] != 0:
                raise ValueError('dp/dn must be zero.') 
            p[0,:] = p[1,:]
        elif 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[0, i-0.5]
                    p[0,i] = 2 * fun_(node[0], node[1], time) - p[1,i]
            else:
                p[0,:] = 2. * fun_ - p[1,:]


    def _update_xi_bc_(self, p, xi, beta, time, _bc ):
        
        # top-boundary condition
        _bc_ = _bc[self.UP]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, p.shape[1]-2]
                    xi[i,-1] = 2. * fun_(node[0], node[1], time) - beta * (p[i,-2] + p[i,-1])
            else:
                xi[:,-1] = 2 * fun_ - beta * (p[:,-2] + p[:,-1])
        
        # bottom-boundary condition
        _bc_ = _bc[self.DOWN]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[0]):
                    node = self.grid[i-0.5, 0]
                    xi[i,0] = 2. * fun_(node[0], node[1], time) - beta * (p[i,1] + p[i,0])
            else:
                xi[:,0] = 2. * fun_ - beta * (p[:,1] + p[:,0])        

        # left-boundary condition
        _bc_ = _bc[self.LEFT]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[p.shape[0]-2, i-0.5]
                    xi[-1,i] = 2. * fun_(node[0], node[1], time) - beta * (p[-2,i] + p[-1,i])
            else:
                xi[-1,:] = 2 * fun_ - beta * (p[-2,:] + p[-1,:])        

        # right-boundary condition
        _bc_ = _bc[self.RIGHT]
        if 'p' in _bc_:
            fun_ = _bc_['p']
            if callable(fun_):
                for i in range(p.shape[1]):
                    node = self.grid[0, i-0.5]
                    xi[0,i] = 2. * fun_(node[0], node[1], time) - beta * (p[1,i] + p[0,i])
            else:
                xi[0,:] = 2 * fun_ - beta * (p[1,:] + p[0,:])        


    def _update_intermediate_vel_bc_(self, u, w, mask, time, _bc):
        """
        update bcs for intermediate velocities.
        """

        # Interior boundaries
        # Apply no-slip boundary conditions to obstacles.
        # Setup masks that are 0 where velocities need to be updated,
        # and 1 where they stay unmodified.
        # Note that (mask & 1) has 1 in the ghost cells.
        u_mask = ( mask[:-1,:] | mask[1:,:] ) & 1
        w_mask = ( mask[:,:-1] | mask[:,1:] ) & 1

        # zero velocity inside and on the boundary of obstacles
        u[:,:] *= ( mask[:-1,:] & mask[1:,:] & 1 )
        # negate velocities inside obstacles
        u[:,1:-2] -= ( 1 - u_mask[:,1:-2] ) * u[:,2:-1]
        u[:,2:-1] -= ( 1 - u_mask[:,2:-1] ) * u[:,1:-2]

        # zero velocity inside and on the boundary of obstacles
        w[:,:] *= ( mask[:,:-1] & mask[:,1:] & 1 )
        # nullify velocities inside obstacles
        w[1:-2,:] -= ( 1 - w_mask[1:-2,:] ) * w[2:-1,:]
        w[2:-1,:] -= ( 1 - w_mask[2:-1,:] ) * w[1:-2,:] 

        # top boundary
        _bc_ = _bc[self.UP]
        if 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(w.shape[0]):
                    node = self.grid[i-0.5, w.shape[1]-1]
                    w[i,-1] = fun_(node[0], node[1], time) * (mask[i,-2] & 1)
            else:
                w[:,-1] = fun_     

        # bottom boundary
        _bc_ = _bc[self.DOWN]
        if 'w' in _bc_:
            fun_ = _bc_['w']
            if callable(fun_):
                for i in range(w.shape[0]):
                    node = self.grid[i-0.5, 0]
                    w[i,0] = fun_(node[0], node[1], time) * (mask[i,1] & 1)
            else:
                w[:,0] = fun_ 

        # left boundary
        _bc_ = _bc[self.LEFT]
        if 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[1]):
                    node = self.grid[u.shape[0]-1, i-0.5]
                    u[-1,i] = fun_(node[0], node[1], time) * (mask[-2,i] & 1)
            else:
                u[-1,:] = fun_

        # west boundary
        _bc_ = _bc[self.RIGHT]
        if 'u' in _bc_:
            fun_ = _bc_['u']
            if callable(fun_):
                for i in range(u.shape[1]):
                    node = self.grid[0, i-0.5]
                    u[0,i] = fun_(node[0], node[1], time) * (mask[1,i] & 1)
            else:
                u[0,:] = fun_

